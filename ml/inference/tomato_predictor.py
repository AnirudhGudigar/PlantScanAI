"""
PlantScan AI - Tomato Disease Predictor (Inference Interface)

Provides a unified prediction class `TomatoDiseasePredictor` that handles image input,
preprocessing, model execution (Keras or TFLite engine), and unknown disease thresholding.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
from PIL import Image
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "ml" / "weights" / "disease_classifier" / "tomato"

DEFAULT_KERAS_PATH = DEFAULT_WEIGHTS_DIR / "tomato_disease_model.keras"
DEFAULT_TFLITE_PATH = DEFAULT_WEIGHTS_DIR / "tomato_disease_model.tflite"
DEFAULT_CLASS_INDICES_PATH = DEFAULT_WEIGHTS_DIR / "class_indices.json"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.50


class TomatoDiseasePredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        class_indices_path: Union[str, Path] = None,
        use_tflite: bool = False,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.use_tflite = use_tflite
        self.confidence_threshold = confidence_threshold

        # Determine class indices path
        if class_indices_path is None:
            class_indices_path = DEFAULT_CLASS_INDICES_PATH
        
        self.class_indices_path = Path(class_indices_path)
        if not self.class_indices_path.exists():
            raise FileNotFoundError(f"Class indices file missing: {self.class_indices_path}")

        with open(self.class_indices_path, "r", encoding="utf-8") as f:
            raw_indices = json.load(f)
            # Ensure keys are integers
            self.class_mapping = {int(k): v for k, v in raw_indices.items()}

        self.num_classes = len(self.class_mapping)

        # Load Model Engine
        if self.use_tflite:
            if model_path is None:
                model_path = DEFAULT_TFLITE_PATH
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"TFLite model file missing: {self.model_path}")

            print(f"Loading TFLite Interpreter: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            if model_path is None:
                model_path = DEFAULT_KERAS_PATH
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Keras model file missing: {self.model_path}")

            print(f"Loading Keras Model Engine: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.interpreter = None

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess input image to required (1, 224, 224, 3) tensor format."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        
        # Expand batch dimension
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Run prediction and return structured result dictionary."""
        img_tensor = self.preprocess_image(image_input)

        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], img_tensor)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            # Model includes internal preprocess_input layer if constructed via train_tomato_classifier.py
            preds = self.model.predict(img_tensor, verbose=0)
            probs = preds[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        predicted_class = self.class_mapping.get(top_idx, f"unknown_class_{top_idx}")

        is_unknown = confidence < self.confidence_threshold

        all_probabilities = {
            self.class_mapping[idx]: float(probs[idx])
            for idx in range(self.num_classes)
        }

        return {
            "crop": "Tomato",
            "predicted_class": "unknown_disease" if is_unknown else predicted_class,
            "raw_prediction": predicted_class,
            "confidence": round(confidence, 4),
            "is_unknown": is_unknown,
            "confidence_threshold": self.confidence_threshold,
            "probabilities": all_probabilities,
            "engine": "TFLite" if self.use_tflite else "Keras",
        }


def main():
    parser = argparse.ArgumentParser(description="PlantScan AI - Tomato Disease Predictor CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input plant image")
    parser.add_argument("--tflite", action="store_true", help="Use TFLite interpreter instead of Keras model")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold for unknown detector")
    args = parser.parse_args()

    predictor = TomatoDiseasePredictor(use_tflite=args.tflite, confidence_threshold=args.threshold)
    result = predictor.predict(args.image)

    print("\n--- Prediction Output ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
