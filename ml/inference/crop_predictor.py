"""
PlantScan AI - Multi-Crop Disease Predictor (Inference Interface)

Provides a unified prediction class `CropDiseasePredictor` that handles image input,
preprocessing, multi-crop model execution (Keras or TFLite engine), and confidence thresholding.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

import numpy as np
from PIL import Image
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "ml" / "weights"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70

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


class CropDiseasePredictor:
    def __init__(
        self,
        crop: str = "tomato",
        model_path: Optional[Union[str, Path]] = None,
        use_tflite: bool = True,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self.crop = crop.strip().lower().replace(" ", "_")
        if self.crop not in SUPPORTED_CROPS:
            raise ValueError(f"Unsupported crop '{self.crop}'. Supported: {SUPPORTED_CROPS}")

        self.use_tflite = use_tflite
        self.confidence_threshold = confidence_threshold

        crop_weights_dir = DEFAULT_WEIGHTS_DIR / self.crop

        if use_tflite:
            if model_path is None:
                model_path = crop_weights_dir / f"{self.crop}_model.tflite"
                if not model_path.exists():
                    model_path = DEFAULT_WEIGHTS_DIR / "disease_classifier.tflite"

            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"TFLite model missing for '{self.crop}': {self.model_path}")

            print(f"Loading TFLite Interpreter for {self.crop.upper()}: {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            if model_path is None:
                model_path = crop_weights_dir / f"{self.crop}_mobilenetv2.keras"

            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Keras model missing for '{self.crop}': {self.model_path}")

            print(f"Loading Keras Model Engine for {self.crop.upper()}: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.interpreter = None

        # Load class list from classes.json, falling back to train directory
        classes_json = crop_weights_dir / "classes.json"
        train_dir = PROJECT_ROOT / "datasets" / "model" / self.crop / "train"
        if classes_json.exists():
            with open(classes_json, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
        elif train_dir.exists():
            self.class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        else:
            self.class_names = [f"{self.crop}_class_{i}" for i in range(10)]

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
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
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        img_tensor = self.preprocess_image(image_input)

        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]["index"], img_tensor)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            probs = self.model.predict(img_tensor, verbose=0)[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        if top_idx < len(self.class_names):
            predicted_class = self.class_names[top_idx]
        else:
            predicted_class = f"class_{top_idx}"

        is_healthy = "healthy" in predicted_class
        is_uncertain = confidence < self.confidence_threshold

        return {
            "crop": self.crop.capitalize(),
            "predicted_class": predicted_class if not is_uncertain else "uncertain",
            "display_name": predicted_class.replace("_", " ").title() if not is_uncertain else "Uncertain Diagnosis",
            "confidence": round(confidence, 4),
            "is_healthy": is_healthy,
            "status": "success" if not is_uncertain else "uncertain",
            "confidence_threshold": self.confidence_threshold,
            "top_k": [
                {
                    "class": self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}",
                    "confidence": round(float(probs[idx]), 4),
                }
                for idx in np.argsort(probs)[::-1][:3]
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Run PlantScan AI crop disease prediction.")
    parser.add_argument("--image", required=True, help="Path to input image file.")
    parser.add_argument("--crop", default="tomato", choices=SUPPORTED_CROPS, help="Target crop species.")
    parser.add_argument("--use-tflite", action="store_true", help="Use TFLite model engine.")

    args = parser.parse_args()
    predictor = CropDiseasePredictor(crop=args.crop, use_tflite=args.use_tflite)
    res = predictor.predict(args.image)
    print("\nPrediction Result:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
