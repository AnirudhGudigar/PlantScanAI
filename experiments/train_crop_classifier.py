"""
PlantScanAI - Crop Species Classifier Training
Trains a MobileNetV2 10-crop species classifier using balanced samples from:
    datasets/model/<crop>/{train,validation,test}

Outputs:
    ml/weights/crop_classifier/crop_classifier.keras
    ml/weights/crop_classifier/crop_classifier.tflite
    ml/weights/crop_classifier/classes.json
    datasets/reports/model/crop_classifier/evaluation_report.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "model"
WEIGHTS_DIR = PROJECT_ROOT / "ml" / "weights" / "crop_classifier"
REPORTS_DIR = PROJECT_ROOT / "datasets" / "reports" / "model" / "crop_classifier"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
MAX_SAMPLES_PER_CROP_TRAIN = 150
MAX_SAMPLES_PER_CROP_VAL = 30
MAX_SAMPLES_PER_CROP_TEST = 30

SUPPORTED_CROPS = [
    "apple",
    "banana",
    "bell_pepper",
    "cotton",
    "grape",
    "maize",
    "mango",
    "potato",
    "rice",
    "tomato",
]


def set_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_crop_filepaths(split: str, max_per_crop: int) -> Tuple[List[str], List[int]]:
    file_paths = []
    labels = []
    random.seed(SEED)

    for crop_idx, crop in enumerate(SUPPORTED_CROPS):
        split_dir = MODEL_DATASET_ROOT / crop / split
        if not split_dir.exists():
            continue

        crop_files = []
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        crop_files.append(str(f))

        random.shuffle(crop_files)
        selected = crop_files[:max_per_crop]
        file_paths.extend(selected)
        labels.extend([crop_idx] * len(selected))
        print(f"  {crop:12} ({split}): {len(selected)} images")

    return file_paths, labels


def parse_image(file_path, label):
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


def create_dataset(file_paths: List[str], labels: List[int], shuffle: bool = True) -> tf.data.Dataset:
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=SEED)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def build_crop_classifier_model(num_classes: int) -> tf.keras.Model:
    input_layer = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")
    
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomRotation(factor=0.1),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    ], name="data_augmentation")
    
    x = augmentation(input_layer)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="crop_predictions")(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output, name="PlantScanAI_CropClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def convert_to_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"TFLite crop classifier saved: {output_path}")


def train_crop_classifier(epochs: int = DEFAULT_EPOCHS):
    set_reproducibility()
    print("=" * 60)
    print("PlantScan AI - Training 10-Crop Species Classifier")
    print("=" * 60)
    
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nCollecting balanced datasets per crop...")
    train_files, train_labels = collect_crop_filepaths("train", MAX_SAMPLES_PER_CROP_TRAIN)
    val_files, val_labels = collect_crop_filepaths("validation", MAX_SAMPLES_PER_CROP_VAL)
    test_files, test_labels = collect_crop_filepaths("test", MAX_SAMPLES_PER_CROP_TEST)
    
    print(f"\nTotal dataset: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    train_ds = create_dataset(train_files, train_labels, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, shuffle=False)
    test_ds = create_dataset(test_files, test_labels, shuffle=False)
    
    print("\nBuilding MobileNetV2 Crop Classifier...")
    model = build_crop_classifier_model(len(SUPPORTED_CROPS))
    
    keras_path = WEIGHTS_DIR / "crop_classifier.keras"
    tflite_path = WEIGHTS_DIR / "crop_classifier.tflite"
    classes_path = WEIGHTS_DIR / "classes.json"
    report_path = REPORTS_DIR / "evaluation_report.json"
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(keras_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]
    
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    if keras_path.exists():
        model = tf.keras.models.load_model(keras_path)
    
    print("\nEvaluating on Test dataset...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    
    y_true, y_pred = [], []
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(lbls.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    report = classification_report(y_true, y_pred, labels=list(range(len(SUPPORTED_CROPS))), target_names=SUPPORTED_CROPS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SUPPORTED_CROPS))))
    
    results = {
        "model": "MobileNetV2-CropClassifier",
        "num_classes": len(SUPPORTED_CROPS),
        "crops": SUPPORTED_CROPS,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved: {report_path}")
    
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(SUPPORTED_CROPS, f, indent=2)
    print(f"Classes manifest saved: {classes_path}")
    
    print("\nConverting model to TFLite...")
    convert_to_tflite(model, tflite_path)
    
    print("\n" + "=" * 60)
    print(f"CROP CLASSIFIER TRAINING COMPLETE")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlantScan AI Crop Species Classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    args = parser.parse_args()
    train_crop_classifier(args.epochs)
