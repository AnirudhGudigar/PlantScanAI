from fastapi.testclient import TestClient
from treatment_database.repository import TreatmentRepository


ALL_10_CROPS = [
    "Apple",
    "Banana",
    "Bell pepper",
    "Cotton",
    "Grape",
    "Maize",
    "Mango",
    "Potato",
    "Rice",
    "Tomato",
]


def test_all_10_crops_present_in_repository():
    repo = TreatmentRepository()
    supported = repo.list_supported_crops()
    assert len(supported) == 10
    for crop in ALL_10_CROPS:
        assert crop in supported, f"Missing crop {crop} in TreatmentRepository"


def test_all_46_canonical_classes_exist():
    repo = TreatmentRepository()
    all_classes = repo.list_all_canonical_classes()
    assert len(all_classes) == 46
    for cls_name in all_classes:
        record = repo.get_by_class(cls_name)
        assert record is not None, f"Missing record for class {cls_name}"
        assert len(record.get("organic_measures", [])) >= 1, f"Missing organic measures for {cls_name}"
        assert len(record.get("prevention_measures", [])) >= 1, f"Missing prevention measures for {cls_name}"


def test_healthy_classes_no_chemical_pesticides():
    repo = TreatmentRepository()
    healthy_classes = [
        "apple_healthy", "banana_healthy", "bell_pepper_healthy", "cotton_healthy",
        "grape_healthy", "mango_healthy", "potato_healthy", "healthy_rice_leaf", "tomato_healthy"
    ]
    for h_cls in healthy_classes:
        record = repo.get_by_class(h_cls)
        if record:
            chemicals = record.get("chemical_measures", [])
            assert len(chemicals) == 0, f"Healthy class {h_cls} must not recommend synthetic pesticides"


def test_list_crops_endpoint(client: TestClient):
    response = client.get("/api/v1/treatments/crops")
    assert response.status_code == 200
    crops = response.json()
    assert isinstance(crops, list)
    assert len(crops) == 10
    assert "Tomato" in crops
    assert "Rice" in crops
    assert "Mango" in crops


def test_list_classes_endpoint(client: TestClient):
    response = client.get("/api/v1/treatments/classes")
    assert response.status_code == 200
    classes = response.json()
    assert isinstance(classes, list)
    assert len(classes) == 46
    assert "tomato_early_blight" in classes
    assert "anthracnose" in classes


def test_get_treatment_endpoint_valid(client: TestClient):
    response = client.get("/api/v1/treatments/Mango/anthracnose")
    assert response.status_code == 200
    data = response.json()
    assert "Mango Anthracnose" in data["summary"]
    assert len(data["organic"]) >= 1
    assert len(data["chemical"]) >= 1
    assert len(data["prevention"]) >= 1
    assert len(data["disclaimer"]) > 0


def test_get_treatment_endpoint_not_found(client: TestClient):
    response = client.get("/api/v1/treatments/Tomato/alien_disease_xyz")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "TREATMENT_NOT_FOUND"


def test_multilingual_repository_support():
    repo = TreatmentRepository()
    languages = repo.list_supported_languages()
    assert "en" in languages
    assert "kn" in languages
    assert "hi" in languages

    kn = repo.get_translation("kn")
    assert "disclaimers" in kn
    assert "general_disclaimer" in kn["disclaimers"]
    assert "crops" in kn
    assert kn["crops"]["Tomato"] == "ಟೊಮೆಟೊ"

    hi = repo.get_translation("hi")
    assert "disclaimers" in hi
    assert hi["crops"]["Tomato"] == "टमाटर"


def test_list_languages_endpoint(client: TestClient):
    response = client.get("/api/v1/treatments/languages")
    assert response.status_code == 200
    langs = response.json()
    assert isinstance(langs, list)
    assert "en" in langs
    assert "kn" in langs
    assert "hi" in langs


def test_get_treatment_localized(client: TestClient):
    # Kannada
    response_kn = client.get("/api/v1/treatments/Mango/anthracnose?lang=kn")
    assert response_kn.status_code == 200
    data_kn = response_kn.json()
    assert "ಮಾವು" in data_kn["summary"]
    assert "PlantScan AI" in data_kn["disclaimer"]

    # Hindi
    response_hi = client.get("/api/v1/treatments/Mango/anthracnose?lang=hi")
    assert response_hi.status_code == 200
    data_hi = response_hi.json()
    assert "आम" in data_hi["summary"]
    assert "PlantScan AI" in data_hi["disclaimer"]
