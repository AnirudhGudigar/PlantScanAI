import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PlantScan AI - Tomato Disease Classifier
# MobileNetV2 Baseline
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "datasets" / "model" / "tomato"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "disease_classifier"
    / "tomato"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "disease_classifier"
    / "tomato"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

CLASS_NAMES = [
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_yellow_leaf_curl_virus",
]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# TensorFlow can still use nondeterministic GPU kernels on some
# systems. This setting improves reproducibility where supported.
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# Directories
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(split):
    directory = DATASET_DIR / split

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {directory}"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=SEED,
    )

    return dataset


print("=" * 80)
print("PlantScan AI - Tomato Disease Classifier")
print("MobileNetV2 Baseline")
print("=" * 80)

print("\nTensorFlow:", tf.__version__)
print("Dataset:", DATASET_DIR)
print("Classes:", len(CLASS_NAMES))
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


train_ds = load_dataset("train")
val_ds = load_dataset("validation")
test_ds = load_dataset("test")


# ============================================================
# Dataset performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# Calculate class weights
# ============================================================

print("\nCalculating class weights...")

class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

for _, labels in train_ds:
    values, counts = np.unique(
        labels.numpy(),
        return_counts=True
    )

    for value, count in zip(values, counts):
        class_counts[value] += count


classes = np.arange(len(CLASS_NAMES))

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.repeat(
        classes,
        class_counts
    ),
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(class_weights_array)
}


print("\n--- Training Class Distribution ---")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{index:2} | "
        f"{class_name:40} | "
        f"{class_counts[index]:5} | "
        f"weight={class_weights[index]:.4f}"
    )


# ============================================================
# Data augmentation
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.08
        ),
        tf.keras.layers.RandomZoom(
            0.10
        ),
        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="augmentation",
)


# ============================================================
# MobileNetV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

# Baseline: freeze pretrained backbone.
base_model.trainable = False


inputs = tf.keras.Input(
    shape=IMG_SIZE + (3,),
    name="image",
)

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.25
)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="disease_prediction",
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="PlantScanAI_Tomato_MobileNetV2",
)


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
    ],
)


print("\n--- Model Summary ---")
model.summary()


# ============================================================
# Callbacks
# ============================================================

best_model_path = OUTPUT_DIR / "tomato_mobilenetv2_baseline.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 80)
print("STARTING BASELINE TRAINING")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
)


# ============================================================
# Save training history
# ============================================================

history_path = REPORT_DIR / "training_history.json"

with history_path.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        f,
        indent=2,
    )


# ============================================================
# Load best model
# ============================================================

print("\nLoading best checkpoint...")

model = tf.keras.models.load_model(
    best_model_path
)


# ============================================================
# Test evaluation
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1,
)

print(f"\nTest loss     : {test_loss:.6f}")
print(f"Test accuracy : {test_accuracy:.6f}")


# ============================================================
# Predictions
# ============================================================

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


print("\n--- Overall Metrics ---")
print(f"Accuracy          : {accuracy:.6f}")
print(f"Weighted Precision: {precision:.6f}")
print(f"Weighted Recall   : {recall:.6f}")
print(f"Weighted F1       : {f1:.6f}")


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\n--- Classification Report ---")
print(report)


report_path = REPORT_DIR / "classification_report.txt"

report_path.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
)

cm_path = REPORT_DIR / "confusion_matrix.csv"

np.savetxt(
    cm_path,
    cm,
    delimiter=",",
    fmt="%d",
)


# ============================================================
# Final metrics JSON
# ============================================================

metrics = {
    "model": "MobileNetV2",
    "experiment": "tomato_baseline",
    "tensorflow_version": tf.__version__,
    "seed": SEED,
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "num_classes": len(CLASS_NAMES),
    "classes": CLASS_NAMES,
    "train_images": int(sum(class_counts)),
    "test_images": int(len(y_true)),
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
}

metrics_path = REPORT_DIR / "baseline_metrics.json"

with metrics_path.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# Save final model
# ============================================================

final_model_path = OUTPUT_DIR / "tomato_mobilenetv2_baseline_final.keras"

model.save(final_model_path)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 80)
print("BASELINE TRAINING COMPLETE")
print("=" * 80)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nReports:")
print(REPORT_DIR)

print("\nSTATUS: SUCCESS")