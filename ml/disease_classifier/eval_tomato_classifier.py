"""
PlantScan AI - Evaluate Tomato Disease Classifier

Evaluates the trained Keras (.keras) and TensorFlow Lite (.tflite) models
against the test dataset split (`datasets/model/tomato/test`).
Generates detailed metrics, evaluation report JSON, and confusion matrix plot.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / "model" / "tomato" / "test"
WEIGHTS_DIR = PROJECT_ROOT / "ml" / "weights" / "disease_classifier" / "tomato"

KERAS_MODEL_PATH = WEIGHTS_DIR / "tomato_disease_model.keras"
TFLITE_MODEL_PATH = WEIGHTS_DIR / "tomato_disease_model.tflite"
CLASS_INDICES_PATH = WEIGHTS_DIR / "class_indices.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_test_data():
    """Load test dataset from directory."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Test dataset directory missing: {DATASET_DIR}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return test_ds, test_ds.class_names


def evaluate_keras_model(model, test_ds):
    """Run evaluation for full Keras model."""
    print("\nRunning evaluation on full Keras model...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred_classes)

    return np.array(y_true), np.array(y_pred)


def evaluate_tflite_model(tflite_path, test_ds):
    """Run evaluation using TensorFlow Lite Interpreter engine."""
    print("\nRunning evaluation on TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        images_np = images.numpy().astype(np.float32)
        
        for i in range(len(images_np)):
            single_img = np.expand_dims(images_np[i], axis=0)
            interpreter.set_tensor(input_details[0]['index'], single_img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            y_pred.append(np.argmax(output_data[0]))
            y_true.append(labels.numpy()[i])

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(cm, class_names, output_path):
    """Generate and save confusion matrix heatmap plot."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Tomato Disease Classifier - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {output_path}")


def main():
    print("=" * 80)
    print("PlantScan AI - Tomato Disease Classifier Evaluation")
    print("=" * 80)

    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained Keras model not found at {KERAS_MODEL_PATH}")

    test_ds, class_names = load_test_data()
    
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    # 1. Keras Model Evaluation
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    y_true, y_pred_keras = evaluate_keras_model(model, test_ds)

    acc_keras = np.mean(y_true == y_pred_keras)
    print(f"\n---> Keras Model Test Accuracy: {acc_keras * 100:.2f}%")

    report_dict = classification_report(
        y_true, y_pred_keras, target_names=class_names, output_dict=True
    )
    print("\n--- Classification Report (Keras Model) ---")
    print(classification_report(y_true, y_pred_keras, target_names=class_names))

    # Confusion matrix
    cm_keras = confusion_matrix(y_true, y_pred_keras)
    plot_confusion_matrix(cm_keras, class_names, WEIGHTS_DIR / "confusion_matrix.png")

    # 2. TFLite Model Evaluation
    if TFLITE_MODEL_PATH.exists():
        y_true_tflite, y_pred_tflite = evaluate_tflite_model(TFLITE_MODEL_PATH, test_ds)
        acc_tflite = np.mean(y_true_tflite == y_pred_tflite)
        print(f"---> TFLite Model Test Accuracy : {acc_tflite * 100:.2f}%")
    else:
        acc_tflite = None

    # Save evaluation report JSON
    eval_report = {
        "dataset": "Tomato",
        "num_classes": len(class_names),
        "total_test_samples": int(len(y_true)),
        "keras_model_accuracy": float(acc_keras),
        "tflite_model_accuracy": float(acc_tflite) if acc_tflite is not None else None,
        "classification_metrics": report_dict,
        "class_names": class_names,
    }

    report_json_path = WEIGHTS_DIR / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nEvaluation report saved to: {report_json_path}")
    print("\nSTATUS: EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
