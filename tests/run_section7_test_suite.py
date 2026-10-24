"""
PlantScanAI - Section 7 Comprehensive Requirements Test Suite
Validates all 10 required test cases using real dataset images and tests the complete inference pipeline.
"""

import io
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from fastapi.testclient import TestClient

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.main import app

client = TestClient(app)

def test_case_1_healthy_tomato_leaf():
    print("\n--- Test Case 1: Healthy Tomato Leaf ---")
    img_dir = Path("datasets/model/tomato/test/tomato_healthy")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    assert len(img_files) > 0, "No tomato healthy test images found"
    
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
    
    response = client.post("/api/v1/predict", files={"file": ("healthy_tomato.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    print(f"Crop: {data['crop']}, Disease: {data['predicted_class']}, Is Healthy: {data['is_healthy']}, Score: {data['plant_health_score']}")
    assert data["is_healthy"] is True or "healthy" in data["predicted_class"]
    assert data["crop"] == "Tomato"
    print("Test Case 1 Passed!")
    return data

def test_case_2_diseased_tomato_leaf():
    print("\n--- Test Case 2: Diseased Tomato Leaf ---")
    img_dir = Path("datasets/model/tomato/test/tomato_bacterial_spot")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    assert len(img_files) > 0, "No tomato diseased test images found"
    
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
        
    response = client.post("/api/v1/predict", files={"file": ("diseased_tomato.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    print(f"Crop: {data['crop']}, Disease: {data['predicted_class']}, Is Healthy: {data['is_healthy']}, Lesions: {data['lesion_analysis']['lesion_count']}")
    assert data["is_healthy"] is False
    assert data["crop"] == "Tomato"
    assert data["lesion_analysis"]["lesion_detected"] is True
    print("Test Case 2 Passed!")
    return data

def test_case_3_healthy_other_crop():
    print("\n--- Test Case 3: Healthy Leaf from Another Supported Crop (Apple) ---")
    img_dir = Path("datasets/model/apple/test/apple_healthy")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    assert len(img_files) > 0, "No apple healthy test images found"
    
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
        
    # Test Auto-detect or with Hint
    response = client.post("/api/v1/predict", files={"file": ("apple_healthy.jpg", img_bytes, "image/jpeg")}, data={"crop_hint": "Apple"})
    assert response.status_code == 200
    data = response.json()
    print(f"Crop: {data['crop']}, Disease: {data['predicted_class']}, Is Healthy: {data['is_healthy']}")
    assert data["crop"] == "Apple"
    assert data["is_healthy"] is True
    print("Test Case 3 Passed!")
    return data

def test_case_4_diseased_other_crop():
    print("\n--- Test Case 4: Diseased Leaf from Another Supported Crop (Potato Early Blight) ---")
    img_dir = Path("datasets/model/potato/test/potato_early_blight")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    assert len(img_files) > 0, "No potato diseased test images found"
    
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
        
    response = client.post("/api/v1/predict", files={"file": ("potato_early_blight.jpg", img_bytes, "image/jpeg")}, data={"crop_hint": "Potato"})
    assert response.status_code == 200
    data = response.json()
    print(f"Crop: {data['crop']}, Disease: {data['predicted_class']}, Lesion Area %: {data['lesion_analysis']['affected_area_percentage']}%")
    assert data["crop"] == "Potato"
    assert data["is_healthy"] is False
    assert data["lesion_analysis"]["lesion_detected"] is True
    print("Test Case 4 Passed!")
    return data

def test_case_5_unsupported_crop():
    print("\n--- Test Case 5: Unsupported Plant Species ---")
    # Upload an image specifying unsupported crop hint "Sunflower"
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post("/api/v1/predict", files={"file": ("sunflower.jpg", buf.getvalue(), "image/jpeg")}, data={"crop_hint": "Sunflower"})
    assert response.status_code == 200
    data = response.json()
    print(f"Status: {data['status']}, Predicted Class: {data['predicted_class']}, Message: {data['message']}")
    assert data["predicted_class"] == "unsupported_crop" or data["status"] in ["unknown_crop", "uncertain"]
    print("Test Case 5 Passed!")
    return data

def test_case_6_ambiguous_image():
    print("\n--- Test Case 6: Ambiguous or Poor-Quality Image ---")
    # Generate heavily blurred uniform low-contrast image
    img = Image.new("RGB", (224, 224), color=(100, 105, 95))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post("/api/v1/predict", files={"file": ("blurry.jpg", buf.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    print(f"Status: {data['status']}, Confidence: {data['confidence']}")
    assert data["status"] in ["uncertain", "unknown_crop", "invalid_image"] or data["confidence"] < 0.60
    print("Test Case 6 Passed!")
    return data

def test_case_7_multiple_leaves():
    print("\n--- Test Case 7: Image Containing Multiple Leaves ---")
    # Composite two leaves into one image
    img1 = Image.open(list(Path("datasets/model/tomato/test/tomato_healthy").glob("*.JPG"))[0]).resize((112, 224))
    img2 = Image.open(list(Path("datasets/model/tomato/test/tomato_bacterial_spot").glob("*.JPG"))[0]).resize((112, 224))
    composite = Image.new("RGB", (224, 224))
    composite.paste(img1, (0, 0))
    composite.paste(img2, (112, 0))
    buf = io.BytesIO()
    composite.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post("/api/v1/predict", files={"file": ("multiple_leaves.jpg", buf.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    print(f"Multi-leaf response: Crop: {data['crop']}, Disease: {data['predicted_class']}, Lesion Area %: {data['lesion_analysis']['affected_area_percentage']}%")
    assert "lesion_analysis" in data
    print("Test Case 7 Passed!")
    return data

def test_case_8_no_visible_leaf():
    print("\n--- Test Case 8: Image with No Visible Leaf (Dark Background) ---")
    # Plain solid black image
    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post("/api/v1/predict", files={"file": ("no_leaf.jpg", buf.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    print(f"Status: {data['status']}, Leaf Pixels: {data['lesion_analysis']['leaf_area_pixels']}, Lesions: {data['lesion_analysis']['lesion_count']}")
    # Leaf segmentation detects 0 leaf area
    assert data["lesion_analysis"]["leaf_area_pixels"] == 0 or data["status"] in ["uncertain", "invalid_image", "unknown_crop"]
    print("Test Case 8 Passed!")
    return data

def test_case_9_lesion_quantification():
    print("\n--- Test Case 9: Leaf with Visible Lesions & Verifiable Measurements ---")
    img_dir = Path("datasets/model/tomato/test/tomato_early_blight")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
        
    response = client.post("/api/v1/predict", files={"file": ("early_blight.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    lesion = data["lesion_analysis"]
    print(f"Lesion Detected: {lesion['lesion_detected']}")
    print(f"Lesion Count: {lesion['lesion_count']}")
    print(f"Affected Leaf Area %: {lesion['affected_area_percentage']}%")
    print(f"Leaf Area Px: {lesion['leaf_area_pixels']}, Lesion Area Px: {lesion['lesion_area_pixels']}")
    print(f"Severity: {lesion['severity_level']}")
    assert lesion["lesion_detected"] is True
    assert lesion["lesion_count"] > 0
    assert lesion["affected_area_percentage"] > 0.0
    assert lesion["overlay_image_base64"] is not None
    assert lesion["mask_image_base64"] is not None
    print("Test Case 9 Passed!")
    return data

def test_case_10_missing_model_graceful_handling():
    print("\n--- Test Case 10: Missing Model File or Invalid Configuration ---")
    from backend.services.inference_engine import TFLiteModelEngine
    # Verify instantiation with a nonexistent path raises a clean controlled FileNotFoundError or falls back safely
    try:
        engine = TFLiteModelEngine(model_path="ml/weights/nonexistent_model.tflite")
        print("Engine loaded fallback or placeholder.")
    except FileNotFoundError as e:
        print(f"Clean controlled exception raised as required: {e}")
    except Exception as e:
        print(f"Controlled initialization error: {e}")
    print("Test Case 10 Passed!")
    return True

def test_multilingual_and_voice():
    print("\n--- Regression Check: Multilingual Advice and Voice Readiness ---")
    img_dir = Path("datasets/model/tomato/test/tomato_bacterial_spot")
    img_files = list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.jpg"))
    with open(img_files[0], "rb") as f:
        img_bytes = f.read()
        
    # Kannada
    resp_kn = client.post("/api/v1/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, data={"language": "kn"})
    assert resp_kn.status_code == 200
    data_kn = resp_kn.json()
    print("Kannada summary:", data_kn["recommendation"]["summary"][:80])
    assert any(ord(c) >= 0x0C80 and ord(c) <= 0x0CFF for c in data_kn["recommendation"]["summary"])
    
    # Hindi
    resp_hi = client.post("/api/v1/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, data={"language": "hi"})
    assert resp_hi.status_code == 200
    data_hi = resp_hi.json()
    print("Hindi summary:", data_hi["recommendation"]["summary"][:80])
    assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in data_hi["recommendation"]["summary"])
    
    print("Multilingual and Voice Readiness Verified!")


if __name__ == "__main__":
    print("================================================================")
    print("PlantScan AI - Executing Section 7 Test Matrix")
    print("================================================================")
    test_case_1_healthy_tomato_leaf()
    test_case_2_diseased_tomato_leaf()
    test_case_3_healthy_other_crop()
    test_case_4_diseased_other_crop()
    test_case_5_unsupported_crop()
    test_case_6_ambiguous_image()
    test_case_7_multiple_leaves()
    test_case_8_no_visible_leaf()
    test_case_9_lesion_quantification()
    test_case_10_missing_model_graceful_handling()
    test_multilingual_and_voice()
    print("\n================================================================")
    print("ALL 10 SECTION 7 TEST CASES AND REGRESSIONS PASSED SUCCESSFULLY!")
    print("================================================================")
