"""
PlantScan AI - Train Tomato Disease Classifier

Model Architecture: MobileNetV2 (Transfer Learning & Fine-tuning)
Dataset: datasets/model/tomato/ (train, validation, test)
Output Artifacts:
  - ml/weights/disease_classifier/tomato/tomato_disease_model.keras
  - ml/weights/disease_classifier/tomato/tomato_disease_model.tflite
  - ml/weights/disease_classifier/tomato/class_indices.json
  - ml/weights/disease_classifier/tomato/training_history.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "model" / "tomato"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "weights" / "disease_classifier" / "tomato"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def get_data_augmentation():
    """Build data augmentation sequential model."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")


def load_datasets(dataset_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Load train, validation, and test datasets from directory structure."""
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Missing train or validation split in {dataset_dir}")

    print(f"Loading training data from: {train_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    class_names = train_ds.class_names
    print(f"Detected {len(class_names)} classes: {class_names}")

    print(f"Loading validation data from: {val_dir}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = None
    if test_dir.exists():
        print(f"Loading test data from: {test_dir}")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
        )

    # Configure dataset performance
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes, img_size=IMG_SIZE):
    """Build MobileNetV2 classification model with transfer learning."""
    inputs = layers.Input(shape=(*img_size, 3), name="input_image")
    
    # Augmentation & MobileNetV2 Preprocessing [-1, 1]
    x = get_data_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Pre-trained MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="tomato_mobilenetv2")
    return model, base_model


def convert_to_tflite(keras_model, output_tflite_path):
    """Convert Keras model to TensorFlow Lite format with dynamic range quantization."""
    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {output_tflite_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train PlantScan AI Tomato Disease Classifier")
    parser.add_argument("--epochs-phase1", type=int, default=5, help="Epochs for feature extraction phase")
    parser.add_argument("--epochs-phase2", type=int, default=5, help="Epochs for fine-tuning phase")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PlantScan AI - Tomato Disease Classifier Training")
    print("=" * 80)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Output Directory  : {OUTPUT_DIR}")

    train_ds, val_ds, test_ds, class_names = load_datasets(DATASET_DIR, batch_size=args.batch_size)

    # Save class indices mapping
    class_indices = {idx: name for idx, name in enumerate(class_names)}
    class_indices_path = OUTPUT_DIR / "class_indices.json"
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)
    print(f"Saved class indices mapping to {class_indices_path}")

    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    # ----------------------------------------------------
    # Phase 1: Train Top Classification Head
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 50)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase1,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )

    # ----------------------------------------------------
    # Phase 2: Fine-Tuning Top Layers of MobileNetV2
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 2: Fine-Tuning MobileNetV2 Base Layers")
    print("=" * 50)

    base_model.trainable = True
    # Freeze bottom layers, fine-tune top layers (from layer 100 onwards)
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    keras_model_path = OUTPUT_DIR / "tomato_disease_model.keras"

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase1 + args.epochs_phase2,
        initial_epoch=history1.epoch[-1] + 1,
        callbacks=[
            callbacks.ModelCheckpoint(filepath=str(keras_model_path), monitor="val_accuracy", save_best_only=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ],
    )

    # Save final Keras model
    model.save(keras_model_path)
    print(f"\nSaved full Keras model to: {keras_model_path}")

    # Convert to TFLite format
    tflite_model_path = OUTPUT_DIR / "tomato_disease_model.tflite"
    convert_to_tflite(model, tflite_model_path)

    # Save complete training history
    combined_history = {
        "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()},
    }
    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    print("\nSTATUS: TRAINING COMPLETE")


if __name__ == "__main__":
    main()
