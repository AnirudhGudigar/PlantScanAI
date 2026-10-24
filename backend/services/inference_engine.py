import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from ..config import settings


@dataclass
class ModelPrediction:
    predicted_class: str
    crop: str
    confidence: float
    is_healthy: bool
    plant_part: str = "Leaf"
    display_name: str = ""
    top_k: List[Dict[str, float]] = field(default_factory=list)
    raw_probabilities: Optional[List[float]] = None
    crop_confidence: float = 1.0
    crop_detected_automatically: bool = False
    candidate_crops: List[Dict[str, Any]] = field(default_factory=list)
    lesion_measurement: Optional[Any] = None


class BaseModelEngine(ABC):
    """Abstract interface for PlantScan AI model inference engines."""

    @abstractmethod
    def predict(
        self,
        image_array: np.ndarray,
        crop_hint: Optional[str] = None,
        plant_part_hint: Optional[str] = None,
        force_simulation: Optional[Dict[str, any]] = None,
        original_image: Optional[Any] = None,
    ) -> ModelPrediction:
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        pass


class MockModelEngine(BaseModelEngine):
    """
    Deterministic inference engine used for API testing, frontend integration,
    and fallback when trained model weights are not loaded.
    Simulates predictions adhering to the project's 10-crop taxonomy.
    """

    CROP_CLASS_MAP = {
        "apple": [
            ("apple_scab", "Apple Scab", False),
            ("apple_rust", "Cedar Apple Rust", False),
            ("apple_healthy", "Healthy Apple Tree", True),
        ],
        "banana": [
            ("cordana", "Banana Cordana Leaf Spot", False),
            ("pestalotiopsis", "Banana Pestalotiopsis Leaf Spot", False),
            ("sigatoka", "Banana Yellow/Black Sigatoka", False),
            ("banana_healthy", "Healthy Banana Plant", True),
        ],
        "bell_pepper": [
            ("bell_pepper_bacterial_spot", "Bell Pepper Bacterial Spot", False),
            ("bell_pepper_leaf_spot", "Bell Pepper Cercospora Leaf Spot", False),
            ("bell_pepper_healthy", "Healthy Bell Pepper Plant", True),
        ],
        "cotton": [
            ("bacterial_blight", "Cotton Bacterial Blight", False),
            ("curl_virus", "Cotton Leaf Curl Virus", False),
            ("fusarium_wilt", "Cotton Fusarium Wilt", False),
            ("cotton_healthy", "Healthy Cotton Crop", True),
        ],
        "grape": [
            ("grape_black_rot", "Grape Black Rot", False),
            ("grape_healthy", "Healthy Grape Vine", True),
        ],
        "maize": [
            ("maize_gray_leaf_spot", "Maize Gray Leaf Spot", False),
            ("maize_leaf_blight", "Northern Corn Leaf Blight", False),
            ("maize_rust", "Common Rust of Maize", False),
        ],
        "mango": [
            ("anthracnose", "Mango Anthracnose", False),
            ("bacterial_canker", "Mango Bacterial Canker", False),
            ("cutting_weevil", "Mango Leaf Cutting Weevil", False),
            ("die_back", "Mango Die Back", False),
            ("gall_midge", "Mango Leaf Gall Midge", False),
            ("powdery_mildew", "Mango Powdery Mildew", False),
            ("sooty_mould", "Mango Sooty Mould", False),
            ("mango_healthy", "Healthy Mango Tree", True),
        ],
        "potato": [
            ("potato_early_blight", "Potato Early Blight", False),
            ("potato_late_blight", "Potato Late Blight", False),
            ("potato_healthy", "Healthy Potato Plant", True),
        ],
        "rice": [
            ("bacterial_leaf_blight", "Rice Bacterial Leaf Blight", False),
            ("brown_spot", "Rice Brown Spot", False),
            ("leaf_blast", "Rice Leaf Blast", False),
            ("leaf_scald", "Rice Leaf Scald", False),
            ("sheath_blight", "Rice Sheath Blight", False),
            ("healthy_rice_leaf", "Healthy Rice Crop", True),
        ],
        "tomato": [
            ("tomato_early_blight", "Tomato Early Blight", False),
            ("tomato_late_blight", "Tomato Late Blight", False),
            ("tomato_bacterial_spot", "Tomato Bacterial Spot", False),
            ("tomato_septoria_leaf_spot", "Tomato Septoria Leaf Spot", False),
            ("tomato_leaf_mold", "Tomato Leaf Mold", False),
            ("tomato_spider_mites", "Tomato Two-Spotted Spider Mites", False),
            ("tomato_target_spot", "Tomato Target Spot", False),
            ("tomato_yellow_leaf_curl_virus", "Tomato Yellow Leaf Curl Virus", False),
            ("tomato_mosaic_virus", "Tomato Mosaic Virus", False),
            ("tomato_healthy", "Healthy Tomato Plant", True),
        ],
    }

    def __init__(self):
        self._name = "MockModelEngine (10-Crop Simulation)"

    @property
    def engine_name(self) -> str:
        return self._name

    def _normalize_crop_key(self, crop: str) -> str:
        clean = crop.strip().lower().replace(" ", "_")
        if clean in self.CROP_CLASS_MAP:
            return clean
        if clean == "bellpepper":
            return "bell_pepper"
        return clean

    def predict(
        self,
        image_array: np.ndarray,
        crop_hint: Optional[str] = None,
        plant_part_hint: Optional[str] = None,
        force_simulation: Optional[Dict[str, any]] = None,
        original_image: Optional[Any] = None,
    ) -> ModelPrediction:
        lesion_meas = None
        try:
            from ml.lesion_detector.detector import PlantLesionDetector
            lesion_input = original_image if original_image is not None else image_array
            lesion_meas = PlantLesionDetector.analyze(lesion_input)
        except Exception:
            pass

        if force_simulation:
            forced_class = force_simulation.get("class", "tomato_early_blight")
            forced_conf = float(force_simulation.get("confidence", 0.94))
            forced_crop = force_simulation.get("crop", "Tomato")
            is_healthy = "healthy" in forced_class
            detected_part = plant_part_hint or "Leaf"
            return ModelPrediction(
                predicted_class=forced_class,
                crop=forced_crop,
                confidence=forced_conf,
                is_healthy=is_healthy,
                plant_part=detected_part,
                display_name=forced_class.replace("_", " ").title(),
                top_k=[{"class": forced_class, "confidence": forced_conf}],
                lesion_measurement=lesion_meas,
            )

        if crop_hint:
            c_clean = self._normalize_crop_key(crop_hint)
            if c_clean not in self.CROP_CLASS_MAP:
                return ModelPrediction(
                    predicted_class="unsupported_crop",
                    crop=crop_hint.capitalize(),
                    confidence=0.15,
                    is_healthy=False,
                    plant_part=plant_part_hint or "Unknown",
                    display_name="Unsupported Crop",
                    top_k=[],
                    lesion_measurement=lesion_meas,
                )
            target_crop_key = c_clean
            auto_detected = False
        else:
            try:
                from ml.crop_classifier.classifier import PlantCropClassifier
                pred_c, c_conf, _, is_conf = PlantCropClassifier.classify(
                    original_image if original_image is not None else image_array
                )
                target_crop_key = self._normalize_crop_key(pred_c)
                auto_detected = True
            except Exception:
                target_crop_key = "tomato"
                auto_detected = False

        classes = self.CROP_CLASS_MAP.get(target_crop_key, self.CROP_CLASS_MAP["tomato"])
        img_mean = float(np.mean(image_array))
        img_std = float(np.std(image_array))
        hash_val = int((img_mean * 1000 + img_std * 500) * 100) % len(classes)

        chosen_class, display_name, is_healthy = classes[hash_val]
        confidence = round(0.85 + (hash_val % 10) * 0.012, 4)
        confidence = min(0.98, max(0.72, confidence))

        top_k = [
            {"class": chosen_class, "confidence": confidence},
            {"class": classes[(hash_val + 1) % len(classes)][0], "confidence": round((1.0 - confidence) * 0.7, 4)},
            {"class": classes[(hash_val + 2) % len(classes)][0], "confidence": round((1.0 - confidence) * 0.3, 4)},
        ]

        crop_display = target_crop_key.replace("_", " ").capitalize()
        if target_crop_key == "bell_pepper":
            crop_display = "Bell pepper"

        detected_part = plant_part_hint
        if not detected_part:
            try:
                from ml.plant_part_classifier.detector import PlantPartClassifier
                detected_part, _ = PlantPartClassifier.classify(image_array, crop_hint=crop_display)
            except Exception:
                detected_part = "Leaf"

        return ModelPrediction(
            predicted_class=chosen_class,
            crop=crop_display,
            confidence=confidence,
            is_healthy=is_healthy,
            plant_part=detected_part,
            display_name=display_name,
            top_k=top_k,
            crop_detected_automatically=auto_detected,
            lesion_measurement=lesion_meas,
        )


class TFLiteModelEngine(BaseModelEngine):
    """
    TensorFlow Lite model inference engine for mobile-grade lightweight deployment.
    Features automated multi-crop species identification, lesion analysis,
    persistent label mappings, thread-safe invocation, and dynamic crop routing.
    """

    def __init__(self, model_path: Path, labels: Optional[List[str]] = None):
        self.model_path = model_path
        self._lock = threading.Lock()
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self.labels: List[str] = labels or []
        self._crop_cache: Dict[str, Tuple[any, any, any, List[str]]] = {}
        self._load_model()

    def _load_labels(self, path: Path) -> List[str]:
        candidates = [
            path.with_name("classes.json"),
            path.parent / "classes.json",
            settings.WEIGHTS_DIR / "classes.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except Exception as e:
                    print(f"Warning: Failed reading {candidate}: {e}")
        return []

    def _load_model(self):
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            if not self.labels:
                self.labels = self._load_labels(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model at {self.model_path}: {e}")

    def _get_crop_interpreter(self, crop: str):
        clean_crop = crop.strip().lower().replace(" ", "_")
        if clean_crop in self._crop_cache:
            return self._crop_cache[clean_crop]

        crop_model_path = settings.WEIGHTS_DIR / clean_crop / f"{clean_crop}_model.tflite"
        if not crop_model_path.exists():
            return None

        try:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=str(crop_model_path))
            interp.allocate_tensors()
            in_det = interp.get_input_details()
            out_det = interp.get_output_details()
            labels = self._load_labels(crop_model_path)
            cached = (interp, in_det, out_det, labels)
            self._crop_cache[clean_crop] = cached
            return cached
        except Exception as e:
            print(f"Notice: Failed to load crop model for {clean_crop} ({e}), using default engine.")
            return None

    @property
    def engine_name(self) -> str:
        return f"TFLiteModelEngine ({self.model_path.name})"

    def predict(
        self,
        image_array: np.ndarray,
        crop_hint: Optional[str] = None,
        plant_part_hint: Optional[str] = None,
        force_simulation: Optional[Dict[str, any]] = None,
        original_image: Optional[Any] = None,
    ) -> ModelPrediction:
        # Determine plant part (Fruit vs Leaf) first so lesion analysis knows the tissue structure
        detected_part = plant_part_hint
        if not detected_part:
            try:
                from ml.plant_part_classifier.detector import PlantPartClassifier
                detected_part, _ = PlantPartClassifier.classify(image_array, crop_hint=crop_hint)
            except Exception:
                detected_part = "Leaf"

        # Simulation override for testing and reproduction
        if force_simulation:
            forced_class = force_simulation.get("class", "tomato_early_blight")
            forced_conf = float(force_simulation.get("confidence", 0.94))
            forced_crop = force_simulation.get("crop", "Tomato")
            is_healthy = "healthy" in forced_class
            try:
                from ml.lesion_detector.detector import PlantLesionDetector
                lesion_input = original_image if original_image is not None else image_array
                lesion_meas = PlantLesionDetector.analyze(
                    lesion_input, plant_part=detected_part, is_healthy=is_healthy
                )
            except Exception:
                lesion_meas = None
            return ModelPrediction(
                predicted_class=forced_class,
                crop=forced_crop,
                confidence=forced_conf,
                is_healthy=is_healthy,
                plant_part=detected_part,
                display_name=forced_class.replace("_", " ").title(),
                top_k=[{"class": forced_class, "confidence": forced_conf}],
                candidate_crops=[],
                lesion_measurement=lesion_meas,
            )

        # Image ambiguity / featureless checks
        check_arr = np.array(original_image) if original_image is not None else image_array
        if check_arr is not None and check_arr.size > 0:
            std_dev = float(np.std(check_arr))
            if std_dev < 12.0:  # Featureless, solid flat color or extreme blur
                try:
                    from ml.lesion_detector.detector import PlantLesionDetector
                    lesion_meas = PlantLesionDetector.analyze(
                        check_arr, plant_part=detected_part, is_healthy=False
                    )
                except Exception:
                    lesion_meas = None
                return ModelPrediction(
                    predicted_class="ambiguous_or_poor_quality",
                    crop="Unknown",
                    confidence=0.20,
                    crop_confidence=0.20,
                    crop_detected_automatically=True,
                    candidate_crops=[],
                    is_healthy=False,
                    plant_part="Unknown",
                    display_name="Ambiguous or Low Quality Image",
                    top_k=[],
                    lesion_measurement=lesion_meas,
                )

        crop_confidence = 1.0
        auto_detected_crop = False
        candidate_crops = []

        if crop_hint:
            clean_c = crop_hint.strip().lower().replace(" ", "_")
            supported_clean = {c.lower().replace(" ", "_"): c for c in settings.SUPPORTED_CROPS}
            if clean_c not in supported_clean and clean_c != "bellpepper":
                try:
                    from ml.lesion_detector.detector import PlantLesionDetector
                    lesion_input = original_image if original_image is not None else image_array
                    lesion_meas = PlantLesionDetector.analyze(
                        lesion_input, plant_part=detected_part, is_healthy=False
                    )
                except Exception:
                    lesion_meas = None
                return ModelPrediction(
                    predicted_class="unsupported_crop",
                    crop=crop_hint.capitalize(),
                    confidence=0.15,
                    crop_confidence=0.15,
                    crop_detected_automatically=False,
                    candidate_crops=[],
                    is_healthy=False,
                    plant_part=detected_part or "Unknown",
                    display_name="Unsupported Crop",
                    top_k=[],
                    lesion_measurement=lesion_meas,
                )
            resolved_crop = supported_clean.get(clean_c, crop_hint.capitalize())
        else:
            # Automated Crop Species Identification via dedicated crop classifier
            try:
                from ml.crop_classifier.classifier import PlantCropClassifier
                pred_c, c_conf, top_k, is_conf = PlantCropClassifier.classify(
                    original_image if original_image is not None else image_array,
                    min_confidence=0.45,
                )
                crop_confidence = c_conf
                auto_detected_crop = True
                candidate_crops = top_k

                if is_conf and pred_c and pred_c != "Unknown":
                    resolved_crop = pred_c
                else:
                    # Detection is uncertain! Do NOT silently choose an incorrect classifier.
                    try:
                        from ml.lesion_detector.detector import PlantLesionDetector
                        lesion_input = original_image if original_image is not None else image_array
                        lesion_meas = PlantLesionDetector.analyze(
                            lesion_input, plant_part=detected_part, is_healthy=False
                        )
                    except Exception:
                        lesion_meas = None
                    return ModelPrediction(
                        predicted_class="uncertain_crop_detection",
                        crop=pred_c if pred_c and pred_c != "Unknown" else "Uncertain",
                        confidence=round(c_conf, 4),
                        crop_confidence=round(c_conf, 4),
                        crop_detected_automatically=True,
                        candidate_crops=top_k,
                        is_healthy=False,
                        plant_part=detected_part or "Leaf",
                        display_name=f"Crop Uncertain: Possible {pred_c} ({round(c_conf*100, 1)}%)",
                        top_k=[{"class": c["crop"], "confidence": c["confidence"]} for c in top_k],
                        lesion_measurement=lesion_meas,
                    )
            except Exception as e:
                print(f"Notice: Automated crop classification error ({e}).")
                return ModelPrediction(
                    predicted_class="unsupported_crop",
                    crop="Unknown",
                    confidence=0.10,
                    crop_confidence=0.10,
                    crop_detected_automatically=True,
                    candidate_crops=[],
                    is_healthy=False,
                    plant_part=detected_part or "Leaf",
                    display_name="Crop Identification Failed",
                    top_k=[],
                    lesion_measurement=None,
                )

        # Load target crop interpreter
        crop_bundle = self._get_crop_interpreter(resolved_crop)
        if crop_bundle:
            interpreter, input_details, output_details, crop_labels = crop_bundle
            labels = crop_labels if crop_labels else self.labels
        else:
            if resolved_crop.lower() == "tomato" and self._interpreter:
                interpreter = self._interpreter
                input_details = self._input_details
                output_details = self._output_details
                labels = self.labels
            else:
                return ModelPrediction(
                    predicted_class="unsupported_crop",
                    crop=resolved_crop,
                    confidence=0.15,
                    crop_confidence=crop_confidence,
                    crop_detected_automatically=auto_detected_crop,
                    candidate_crops=candidate_crops,
                    is_healthy=False,
                    plant_part=detected_part or "Leaf",
                    display_name=f"Classifier Unavailable for {resolved_crop}",
                    top_k=[],
                    lesion_measurement=None,
                )

        with self._lock:
            interpreter.set_tensor(input_details[0]["index"], image_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

        probs = output_data[0]
        # Handle legacy single-unit sigmoid binary output gracefully
        if probs.ndim == 1 and len(probs) == 1 and len(labels) == 2:
            sig = float(probs[0])
            probs = np.array([1.0 - sig, sig], dtype=np.float32)

        max_idx = int(np.argmax(probs))
        confidence = round(float(probs[max_idx]), 4)

        if labels and max_idx < len(labels):
            predicted_class = labels[max_idx]
            display_name = predicted_class.replace("_", " ").title()
            is_healthy = "healthy" in predicted_class.lower()
        else:
            predicted_class = f"class_{max_idx}"
            display_name = f"Class {max_idx}"
            is_healthy = False

        # Run plant-part aware lesion detection
        lesion_meas = None
        try:
            from ml.lesion_detector.detector import PlantLesionDetector
            lesion_input = original_image if original_image is not None else image_array
            lesion_meas = PlantLesionDetector.analyze(
                lesion_input,
                plant_part=detected_part,
                is_healthy=is_healthy,
            )
        except Exception as e:
            print(f"Notice: Lesion detection skipped ({e})")

        # Check if foliage / fruit was present in the image
        if lesion_meas and lesion_meas.leaf_area_pixels == 0:
            return ModelPrediction(
                predicted_class="no_foliar_leaf_detected",
                crop=resolved_crop,
                confidence=0.10,
                crop_confidence=crop_confidence,
                crop_detected_automatically=auto_detected_crop,
                candidate_crops=candidate_crops,
                is_healthy=False,
                plant_part="None",
                display_name="No Plant Foliage or Fruit Detected",
                top_k=[],
                lesion_measurement=lesion_meas,
            )

        # Check for false healthy predictions:
        # If model claims Healthy, but vision clearly detects severe foliar necrosis:
        if is_healthy and lesion_meas and lesion_meas.lesion_detected:
            if lesion_meas.affected_area_percentage >= 15.0 and lesion_meas.lesion_count >= 5:
                confidence = min(confidence, 0.62)
                is_healthy = False
                display_name = f"{resolved_crop} Foliar Damage (Suspected Disease)"

        top_k_indices = np.argsort(probs)[::-1][:min(3, len(probs))]
        top_k = [
            {
                "class": labels[idx] if labels and idx < len(labels) else f"class_{idx}",
                "confidence": round(float(probs[idx]), 4),
            }
            for idx in top_k_indices
        ]

        return ModelPrediction(
            predicted_class=predicted_class,
            crop=resolved_crop,
            confidence=confidence,
            is_healthy=is_healthy,
            plant_part=detected_part,
            display_name=display_name,
            top_k=top_k,
            raw_probabilities=[round(float(p), 4) for p in probs],
            crop_confidence=crop_confidence,
            crop_detected_automatically=auto_detected_crop,
            candidate_crops=candidate_crops,
            lesion_measurement=lesion_meas,
        )


_engine_instance: Optional[BaseModelEngine] = None


def get_inference_engine() -> BaseModelEngine:
    global _engine_instance
    if _engine_instance is None:
        tflite_path = settings.WEIGHTS_DIR / "disease_classifier.tflite"
        if tflite_path.exists():
            try:
                _engine_instance = TFLiteModelEngine(tflite_path)
            except Exception as e:
                print(f"Notice: TFLite engine initialization skipped ({e}), falling back to MockModelEngine.")
                _engine_instance = MockModelEngine()
        else:
            _engine_instance = MockModelEngine()
    return _engine_instance
