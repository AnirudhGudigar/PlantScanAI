import io
from fastapi.testclient import TestClient


def test_predict_valid_image(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("leaf.jpg", valid_leaf_image_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "uncertain"]
    assert data["crop"] in [
        "Apple", "Banana", "Bell pepper", "Cotton", "Grape",
        "Maize", "Mango", "Potato", "Rice", "Tomato"
    ]
    assert "predicted_class" in data
    assert not data["predicted_class"].isdigit(), f"predicted_class should not be numeric string: {data['predicted_class']}"
    assert 0 <= data["plant_health_score"] <= 100
    assert data["recommendation"] is not None
    assert len(data["recommendation"]["disclaimer"]) > 0
    assert "lesion_analysis" in data


def test_predict_crop_routing_mango(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("mango_leaf.jpg", valid_leaf_image_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files, data={"crop_hint": "Mango"})
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Mango"
    assert "predicted_class" in data
    assert not data["predicted_class"].isdigit()


def test_predict_forced_simulation_healthy(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("tomato_healthy.jpg", valid_leaf_image_bytes, "image/jpeg")}
    headers = {
        "X-Simulate-Class": "tomato_healthy",
        "X-Simulate-Confidence": "0.96",
        "X-Simulate-Crop": "Tomato",
    }
    response = client.post("/api/v1/predict", files=files, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_class"] == "tomato_healthy"
    assert data["is_healthy"] is True
    assert data["confidence"] == 0.96
    assert data["plant_health_score"] >= 90
    assert len(data["recommendation"]["chemical"]) == 0
    assert len(data["recommendation"]["organic"]) >= 1


def test_predict_uncertain_low_confidence(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("ambiguous.jpg", valid_leaf_image_bytes, "image/jpeg")}
    headers = {
        "X-Simulate-Class": "tomato_early_blight",
        "X-Simulate-Confidence": "0.45",
        "X-Simulate-Crop": "Tomato",
    }
    response = client.post("/api/v1/predict", files=files, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "uncertain"
    assert data["confidence"] == 0.45
    assert len(data["recommendation"]["chemical"]) == 0
    assert data["recommendation"]["photo_guidance"] is not None


def test_predict_unsupported_crop(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("dragonfruit.jpg", valid_leaf_image_bytes, "image/jpeg")}
    data_form = {"crop_hint": "DragonFruit"}
    response = client.post("/api/v1/predict", files=files, data=data_form)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unsupported_crop"
    assert data["predicted_class"] == "unsupported_crop"


def test_predict_invalid_corrupt_image(client: TestClient, corrupt_image_bytes: bytes):
    files = {"file": ("corrupt.jpg", corrupt_image_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)
    
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "invalid_image"


def test_predict_json_base64_valid(client: TestClient, valid_base64_image: str):
    payload = {
        "image_base64": valid_base64_image,
        "crop_hint": "Tomato",
        "language": "en",
    }
    response = client.post("/api/v1/predict/json", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "uncertain"]
    assert data["crop"] == "Tomato"


def test_predict_kannada_localization(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("leaf.jpg", valid_leaf_image_bytes, "image/jpeg")}
    headers = {
        "X-Simulate-Class": "tomato_early_blight",
        "X-Simulate-Confidence": "0.95",
        "X-Simulate-Crop": "Tomato",
    }
    response = client.post("/api/v1/predict", files=files, data={"language": "kn"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "ಟೊಮೆಟೊ"
    assert "PlantScan AI" in data["recommendation"]["disclaimer"]
    assert "ರೋಗನಿರ್ಣಯ ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಂಡಿದೆ." in data["message"]


def test_predict_hindi_localization(client: TestClient, valid_leaf_image_bytes: bytes):
    files = {"file": ("leaf.jpg", valid_leaf_image_bytes, "image/jpeg")}
    headers = {
        "X-Simulate-Class": "tomato_early_blight",
        "X-Simulate-Confidence": "0.95",
        "X-Simulate-Crop": "Tomato",
    }
    response = client.post("/api/v1/predict", files=files, data={"language": "hi"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "टमाटर"
    assert "PlantScan AI" in data["recommendation"]["disclaimer"]
    assert "निदान सफलतापूर्वक पूरा हुआ।" in data["message"]


def test_predict_json_base64_kannada(client: TestClient, valid_base64_image: str):
    payload = {
        "image_base64": valid_base64_image,
        "crop_hint": "Tomato",
        "language": "kn",
    }
    response = client.post("/api/v1/predict/json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "ಟೊಮೆಟೊ"
    assert "PlantScan AI" in data["recommendation"]["disclaimer"]
