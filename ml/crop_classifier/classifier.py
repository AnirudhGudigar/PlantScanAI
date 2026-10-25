"""
PlantScan AI - Automated Crop Species Classifier Interface

Classifies uploaded plant images into one of the 10 supported crops:
    Apple, Banana, Bell pepper, Cotton, Grape, Maize, Mango, Potato, Rice, Tomato

Uses a dedicated MobileNetV2 crop-species classifier (ml/weights/crop_classifier/crop_classifier.tflite).
Prevents automatic hardcoded fallback to Tomato.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "weights" / "crop_classifier" / "crop_classifier.tflite"
CLASSES_PATH = PROJECT_ROOT / "ml" / "weights" / "crop_classifier" / "classes.json"


class PlantCropClassifier:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model_path = MODEL_PATH
        self.classes_path = CLASSES_PATH
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self.classes = []
        self._load()

    def _load(self):
        if self.classes_path.exists():
            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
        else:
            self.classes = [
                "apple", "banana", "bell_pepper", "cotton", "grape",
                "maize", "mango", "potato", "rice", "tomato"
            ]

        if self.model_path.exists():
            try:
                import tensorflow as tf
                self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
            except Exception as e:
                print(f"Warning: Failed loading Crop Classifier TFLite: {e}")
                self._interpreter = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PlantCropClassifier()
        return cls._instance

    @classmethod
    def classify(
        cls,
        image_input: Union[np.ndarray, Image.Image],
        min_confidence: float = 0.45,
    ) -> Tuple[str, float, List[Dict[str, float]], bool]:
        """
        Classifies plant crop species.
        Returns:
            (predicted_crop_display, confidence, top_k_crops, is_confident)
        """
        classifier = cls.get_instance()
        if classifier._interpreter is None:
            return "Unknown", 0.0, [], False

        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB").resize((224, 224))
            arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3:
                arr = np.expand_dims(image_input, axis=0)
            else:
                arr = image_input
            if arr.shape[1:3] != (224, 224):
                pil_tmp = Image.fromarray(arr[0].astype(np.uint8)).resize((224, 224))
                arr = np.expand_dims(np.array(pil_tmp, dtype=np.float32), axis=0)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
        else:
            return "Unknown", 0.0, [], False

        with classifier._lock:
            classifier._interpreter.set_tensor(classifier._input_details[0]["index"], arr)
            classifier._interpreter.invoke()
            probs = classifier._interpreter.get_tensor(classifier._output_details[0]["index"])[0]

        top_indices = np.argsort(probs)[::-1]
        top_idx = int(top_indices[0])
        confidence = float(probs[top_idx])

        raw_crop = classifier.classes[top_idx] if top_idx < len(classifier.classes) else "unknown"
        crop_display = raw_crop.replace("_", " ").capitalize()
        if raw_crop == "bell_pepper":
            crop_display = "Bell pepper"

        top_k = []
        for idx in top_indices[:3]:
            c_name = classifier.classes[idx] if idx < len(classifier.classes) else f"crop_{idx}"
            disp = c_name.replace("_", " ").capitalize()
            if c_name == "bell_pepper":
                disp = "Bell pepper"
            top_k.append({"crop": disp, "confidence": round(float(probs[idx]), 4)})

        # Check margin between top 1 and top 2 predictions:
        # If top 1 is below 0.60 and the runner-up is within 0.05, treat detection as ambiguous/uncertain.
        is_confident = confidence >= min_confidence
        if len(top_k) >= 2 and confidence < 0.60:
            margin = top_k[0]["confidence"] - top_k[1]["confidence"]
            if margin < 0.05:
                is_confident = False

        return crop_display, round(confidence, 4), top_k, is_confident
