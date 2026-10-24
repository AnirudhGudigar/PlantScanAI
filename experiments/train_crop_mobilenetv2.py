"""
PlantScanAI - Crop-wise MobileNetV2 Training

Trains one MobileNetV2 classifier per crop using:
    datasets/model/<crop>/{train,validation,test}

Outputs:
    ml/weights/<crop>/<crop>_mobilenetv2.keras
    ml/weights/<crop>/<crop>_model.tflite
    datasets/reports/model/<crop>/evaluation_report.json

This version intentionally does NOT use class weights.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DATASET_ROOT = PROJECT_ROOT / "datasets" / "model"
WEIGHTS_DIR = PROJECT_ROOT / "ml" / "weights"
REPORTS_DIR = PROJECT_ROOT / "datasets" / "reports" / "model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DEFAULT_EPOCHS = 10
LEARNING_RATE = 1e-3

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_reproducibility(seed: int = SEED) -> None:
    """
    Set random seeds for reproducible training.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ============================================================
# DATASET UTILITIES
# ============================================================

def get_crop_classes(crop: str) -> List[str]:
    """
    Discover class names from:

        datasets/model/<crop>/train/

    Classes are sorted alphabetically to guarantee the same
    class-to-index mapping across train/validation/test.
    """

    train_dir = MODEL_DATASET_ROOT / crop / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    class_names = sorted(
        [
            directory.name
            for directory in train_dir.iterdir()
            if directory.is_dir()
        ]
    )

    if not class_names:
        raise ValueError(
            f"No class directories found in: {train_dir}"
        )

    return class_names


def load_dataset(
    crop: str,
    split: str,
    class_names: List[str],
) -> tf.data.Dataset:
    """
    Load a train/validation/test dataset.
    """

    directory = MODEL_DATASET_ROOT / crop / split

    if not directory.exists():
        raise FileNotFoundError(
            f"{split} directory not found: {directory}"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=SEED if split == "train" else None,
    )

    return dataset


# ============================================================
# MODEL
# ============================================================

def build_mobilenetv2_model(
    num_classes: int,
) -> tf.keras.Model:
    """
    Build a MobileNetV2 classifier.

    Stage 1:
        MobileNetV2 ImageNet backbone is frozen.

    Classification head:
        GlobalAveragePooling2D
        Dropout
        Sigmoid for binary classification
        Softmax for multi-class classification
    """

    input_layer = tf.keras.Input(
        shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        name="image",
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal"
            ),
            tf.keras.layers.RandomRotation(
                factor=0.15
            ),
            tf.keras.layers.RandomZoom(
                height_factor=0.1,
                width_factor=0.1,
            ),
        ],
        name="data_augmentation",
    )

    x = augmentation(input_layer)

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # --------------------------------------------------------
    # MobileNetV2 backbone
    # --------------------------------------------------------

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze backbone for initial training.
    base_model.trainable = False

    x = base_model(
        x,
        training=False,
    )

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(
        0.2
    )(x)

    output = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output,
        name="PlantScanAI_MobileNetV2",
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model


# ============================================================
# TFLITE CONVERSION
# ============================================================

def convert_to_tflite(
    model: tf.keras.Model,
    output_path: Path,
) -> None:
    """
    Convert Keras model to optimized TensorFlow Lite model.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    tflite_model = converter.convert()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        tflite_model
    )

    print(
        f"TFLite model saved: {output_path}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
) -> dict:
    """
    Evaluate the trained model on the test dataset.

    Produces:
        accuracy
        weighted precision
        weighted recall
        weighted F1
        macro precision
        macro recall
        macro F1
        classification report
        confusion matrix
    """

    print("\nEvaluating test dataset...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1,
    )

    y_true = []
    y_pred = []

    # --------------------------------------------------------
    # Collect predictions
    # --------------------------------------------------------

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0,
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1,
        )

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predicted_classes.tolist()
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --------------------------------------------------------
    # Weighted metrics
    # --------------------------------------------------------

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),

        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),

        "classification_report": report,

        "confusion_matrix": cm.tolist(),
    }

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n========== TEST RESULTS ==========")

    print(
        f"Test Loss       : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy   : {test_accuracy:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print("==================================\n")

    # --------------------------------------------------------
    # Print per-class F1
    # --------------------------------------------------------

    print("Per-class F1 scores:")

    for class_name in class_names:

        class_f1 = report[class_name]["f1-score"]

        print(
            f"  {class_name:<40} "
            f"{class_f1:.4f}"
        )

    print()

    return results


# ============================================================
# TRAINING
# ============================================================

def train_and_evaluate_crop(
    crop: str,
    epochs: int,
) -> None:
    """
    Train and evaluate one crop model.
    """

    print("\n")
    print("=" * 70)
    print(f"PlantScanAI - Training crop: {crop}")
    print("=" * 70)

    # --------------------------------------------------------
    # Discover classes
    # --------------------------------------------------------

    class_names = get_crop_classes(crop)

    num_classes = len(class_names)

    print(
        f"\nClasses ({num_classes}):"
    )

    for index, class_name in enumerate(class_names):

        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_ds = load_dataset(
        crop,
        "train",
        class_names,
    )

    val_ds = load_dataset(
        crop,
        "validation",
        class_names,
    )

    test_ds = load_dataset(
        crop,
        "test",
        class_names,
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        autotune
    )

    val_ds = val_ds.prefetch(
        autotune
    )

    test_ds = test_ds.prefetch(
        autotune
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding MobileNetV2 model...")

    model = build_mobilenetv2_model(
        num_classes
    )

    model.summary()

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    crop_weights_dir = (
        WEIGHTS_DIR / crop
    )

    crop_report_dir = (
        REPORTS_DIR / crop
    )

    crop_weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    keras_path = (
        crop_weights_dir
        / f"{crop}_mobilenetv2.keras"
    )

    tflite_path = (
        crop_weights_dir
        / f"{crop}_model.tflite"
    )

    report_path = (
        crop_report_dir
        / "evaluation_report.json"
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(keras_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # No class_weight argument here.
    # --------------------------------------------------------

    print("\nStarting training...")
    print("Class weights: DISABLED")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # --------------------------------------------------------
    # Restore best model from checkpoint
    # --------------------------------------------------------

    if keras_path.exists():

        print(
            "\nLoading best checkpoint..."
        )

        model = tf.keras.models.load_model(
            keras_path
        )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_model(
        model,
        test_ds,
        class_names,
    )

    # --------------------------------------------------------
    # Add training metadata
    # --------------------------------------------------------

    results["crop"] = crop

    results["class_names"] = class_names

    results["num_classes"] = num_classes

    results["image_size"] = list(
        IMG_SIZE
    )

    results["batch_size"] = BATCH_SIZE

    results["epochs_requested"] = epochs

    results["class_weights_used"] = False

    results["model"] = "MobileNetV2"

    results["seed"] = SEED

    # --------------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------------

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"Evaluation report saved: "
        f"{report_path}"
    )

    # --------------------------------------------------------
    # Convert to TFLite
    # --------------------------------------------------------

    print("\nConverting model to TFLite...")

    convert_to_tflite(
        model,
        tflite_path,
    )

    # --------------------------------------------------------
    # Save classes manifest
    # --------------------------------------------------------

    classes_json_path = crop_weights_dir / "classes.json"
    with open(classes_json_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Classes list saved: {classes_json_path}")

    # --------------------------------------------------------
    # Tomato compatibility output
    # --------------------------------------------------------

    if crop == "tomato":

        (WEIGHTS_DIR / "classes.json").write_text(
            json.dumps(class_names, indent=2),
            encoding="utf-8",
        )

        disease_classifier_path = (
            WEIGHTS_DIR
            / "disease_classifier"
            / "tomato"
            / "disease_classifier.tflite"
        )

        disease_classifier_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        disease_classifier_path.write_bytes(
            tflite_path.read_bytes()
        )

        print(
            "Tomato disease classifier copied to:"
        )

        print(
            f"  {disease_classifier_path}"
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(f"TRAINING COMPLETE: {crop}")
    print("=" * 70)

    print(
        f"Test Accuracy : "
        f"{results['test_accuracy']:.4f}"
    )

    print(
        f"Weighted F1   : "
        f"{results['weighted_f1']:.4f}"
    )

    print(
        f"Macro F1      : "
        f"{results['macro_f1']:.4f}"
    )

    print(
        f"Keras model   : {keras_path}"
    )

    print(
        f"TFLite model  : {tflite_path}"
    )

    print(
        f"Report        : {report_path}"
    )

    print("=" * 70)


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train PlantScanAI MobileNetV2 "
            "models crop-wise."
        )
    )

    parser.add_argument(
        "--crop",
        type=str,
        default="all",
        choices=SUPPORTED_CROPS + ["all"],
        help=(
            "Crop to train. "
            "Use 'all' to train every supported crop."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            f"Number of training epochs "
            f"(default: {DEFAULT_EPOCHS})"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_reproducibility()

    args = parse_arguments()

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be greater than 0."
        )

    if args.crop == "all":

        crops = SUPPORTED_CROPS

    else:

        crops = [args.crop]

    print("\n")
    print("=" * 70)
    print("PlantScanAI - Crop Model Training")
    print("=" * 70)

    print(
        f"Selected crops : {', '.join(crops)}"
    )

    print(
        f"Epochs         : {args.epochs}"
    )

    print(
        "Class weights  : DISABLED"
    )

    print(
        f"Image size     : {IMG_SIZE}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    print("=" * 70)

    for crop in crops:

        try:

            train_and_evaluate_crop(
                crop=crop,
                epochs=args.epochs,
            )

        except Exception as error:

            print("\n")
            print("=" * 70)
            print(
                f"ERROR TRAINING CROP: {crop}"
            )
            print("=" * 70)

            print(
                repr(error)
            )

            print("=" * 70)

            raise


if __name__ == "__main__":
    main()