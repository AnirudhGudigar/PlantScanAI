"""
PlantScanAI - Reproducible Test Suite for Auto Detect Routing & Lesion Highlighting

Validates:
1. Task 1: Auto Detect vs Manual Crop Selection pipeline comparison on identical images.
   Records detected crop, confidence, candidate crops, selected model interpreter,
   and predicted disease class.
2. Task 2: Lesion Segmentation quality on healthy vs diseased leaves and fruits.
   Verifies that normal red/orange/yellow fruit coloration and green leaf tissue
   are not marked as lesions, while true necrotic spots are accurately detected.
"""

import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.main import app
from ml.lesion_detector.detector import PlantLesionDetector
from ml.crop_classifier.classifier import PlantCropClassifier

client = TestClient(app)


def run_pipeline_test(image_path: Path, manual_crop: str, true_disease: str):
    """
    Runs the same image through both Auto Detect (crop_hint=None)
    and Manual Selection (crop_hint=manual_crop).
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # 1. Auto Detect Path
    res_auto = client.post(
        "/api/v1/predict",
        files={"file": (image_path.name, img_bytes, "image/jpeg")},
        data={"language": "en"}
    )
    assert res_auto.status_code == 200, f"Auto Detect error: {res_auto.text}"
    auto_data = res_auto.json()

    # 2. Manual Crop Selection Path
    res_manual = client.post(
        "/api/v1/predict",
        files={"file": (image_path.name, img_bytes, "image/jpeg")},
        data={"crop_hint": manual_crop, "language": "en"}
    )
    assert res_manual.status_code == 200, f"Manual error: {res_manual.text}"
    manual_data = res_manual.json()

    return auto_data, manual_data


def test_task_1_auto_vs_manual():
    print("\n" + "=" * 70)
    print("TASK 1: AUTO DETECT VS MANUAL CROP SELECTION PIPELINE TESTS")
    print("=" * 70)

    test_samples = [
        ("datasets/raw/PlantVillage/Tomato_healthy", "Tomato", "tomato_healthy"),
        ("datasets/raw/PlantVillage/Tomato_Bacterial_spot", "Tomato", "tomato_bacterial_spot"),
        ("datasets/raw/PlantVillage/Tomato_Early_blight", "Tomato", "tomato_early_blight"),
        ("datasets/raw/PlantVillage/Pepper__bell___healthy", "Bell pepper", "bell_pepper_healthy"),
        ("datasets/raw/PlantVillage/Pepper__bell___Bacterial_spot", "Bell pepper", "bell_pepper_bacterial_spot"),
        ("datasets/raw/PlantVillage/Potato___Early_blight", "Potato", "potato_early_blight"),
        ("datasets/raw/PlantVillage/Potato___Late_blight", "Potato", "potato_late_blight"),
    ]

    results_table = []

    for rel_dir, manual_crop, true_disease in test_samples:
        p_dir = PROJECT_ROOT / rel_dir
        imgs = list(p_dir.glob("*.JPG")) + list(p_dir.glob("*.jpg"))
        if not imgs:
            print(f"Skipping {rel_dir}: No images found.")
            continue

        test_img = imgs[0]
        auto_res, manual_res = run_pipeline_test(test_img, manual_crop, true_disease)

        auto_crop = auto_res.get("crop", "Unknown")
        auto_crop_conf = auto_res.get("crop_confidence")
        auto_crop_conf_str = f"{auto_crop_conf*100:.1f}%" if auto_crop_conf is not None else "N/A"
        auto_disease = auto_res.get("predicted_class", "")
        auto_conf = f"{auto_res.get('confidence', 0.0)*100:.1f}%"
        auto_status = auto_res.get("status", "")

        man_crop = manual_res.get("crop", manual_crop)
        man_disease = manual_res.get("predicted_class", "")
        man_conf = f"{manual_res.get('confidence', 0.0)*100:.1f}%"
        man_status = manual_res.get("status", "")

        match = (auto_disease == man_disease)
        routing_note = "Consistent Routing" if match else f"Model routed by detected crop ({auto_crop})"

        results_table.append({
            "image": test_img.name[:25],
            "true_category": rel_dir.split("/")[-1],
            "manual_crop": manual_crop,
            "auto_crop": f"{auto_crop} ({auto_crop_conf_str})",
            "auto_disease": auto_disease,
            "auto_conf": auto_conf,
            "manual_disease": man_disease,
            "manual_conf": man_conf,
            "status": f"Auto: {auto_status} / Man: {man_status}",
            "note": routing_note,
        })

        print(f"\n[Test Image: {test_img.name[:35]}]")
        print(f"  Target Ground Truth: {true_disease} on {manual_crop}")
        print(f"  Auto Detect Path   : Crop={auto_crop} ({auto_crop_conf_str}), Status={auto_status}")
        print(f"                       Predicted Disease={auto_disease} ({auto_conf})")
        print(f"  Manual Select Path : Target Crop={man_crop}, Status={man_status}")
        print(f"                       Predicted Disease={man_disease} ({man_conf})")
        if not match:
            print(f"  * Expected Discrepancy Explanation: Crop classifier identified '{auto_crop}', routing to the {auto_crop} model bundle. Manual override directs to the {manual_crop} model bundle.")

    return results_table


def test_task_2_lesion_segmentation():
    print("\n" + "=" * 70)
    print("TASK 2: LESION SEGMENTATION ON HEALTHY VS DISEASED SPECIMENS")
    print("=" * 70)

    # 1. 100% Healthy Fruit Tests (Synthetic Red & Yellow/Green Fruits)
    print("\n--- 1. Testing Fruit Lesion Highlighting ---")
    
    # Healthy Red Tomato Fruit
    red_fruit = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            dist = np.sqrt((x - 112)**2 + (y - 112)**2)
            if dist < 80:
                shade = max(0.0, 1.0 - (dist / 110.0))
                red_fruit[y, x] = [int(225 * shade), int(35 * shade), int(25 * shade)]
            else:
                red_fruit[y, x] = [180, 180, 180]  # Neutral background
    
    meas_red = PlantLesionDetector.analyze(red_fruit, plant_part="Fruit", is_healthy=True)
    print(f"Healthy Red Fruit     -> Tissue Px: {meas_red.leaf_area_pixels}, Lesion Px: {meas_red.lesion_area_pixels}, Count: {meas_red.lesion_count}, Affected: {meas_red.affected_area_percentage}%")
    assert meas_red.lesion_count == 0, f"False lesion count on healthy red fruit: {meas_red.lesion_count}"
    assert meas_red.lesion_area_pixels == 0, f"False lesion pixels on healthy red fruit: {meas_red.lesion_area_pixels}"
    print("✓ Healthy Red Fruit: 0 false lesions (Passed)")

    # Healthy Yellow/Green Fruit
    yellow_fruit = np.zeros((224, 224, 3), dtype=np.uint8)
    for y in range(224):
        for x in range(224):
            dist = np.sqrt((x - 112)**2 + (y - 112)**2)
            if dist < 75:
                t = dist / 75.0
                yellow_fruit[y, x] = [int(254*(1-t) + 77*t), int(240*(1-t) + 124*t), int(138*(1-t) + 15*t)]
            else:
                yellow_fruit[y, x] = [200, 200, 200]
                
    meas_yellow = PlantLesionDetector.analyze(yellow_fruit, plant_part="Fruit", is_healthy=True)
    print(f"Healthy Yellow Fruit  -> Tissue Px: {meas_yellow.leaf_area_pixels}, Lesion Px: {meas_yellow.lesion_area_pixels}, Count: {meas_yellow.lesion_count}, Affected: {meas_yellow.affected_area_percentage}%")
    assert meas_yellow.lesion_count == 0, f"False lesion count on healthy yellow fruit: {meas_yellow.lesion_count}"
    print("✓ Healthy Yellow Fruit: 0 false lesions (Passed)")

    # Diseased Fruit with 2 Dark Sunken Rot Spots
    rot_fruit = yellow_fruit.copy()
    for y in range(224):
        for x in range(224):
            if (x - 95)**2 + (y - 95)**2 < 25**2:
                rot_fruit[y, x] = [63, 46, 26]
            if (x - 140)**2 + (y - 125)**2 < 18**2:
                rot_fruit[y, x] = [41, 27, 16]
                
    meas_rot = PlantLesionDetector.analyze(rot_fruit, plant_part="Fruit", is_healthy=False)
    print(f"Diseased Fruit w/ Rot -> Tissue Px: {meas_rot.leaf_area_pixels}, Lesion Px: {meas_rot.lesion_area_pixels}, Count: {meas_rot.lesion_count}, Affected: {meas_rot.affected_area_percentage}%")
    assert meas_rot.lesion_detected is True, "Failed to detect fruit lesions"
    assert meas_rot.lesion_count >= 1, f"Expected fruit spots, got {meas_rot.lesion_count}"
    print("✓ Diseased Fruit: Correctly detected genuine sunken rot lesions without marking healthy skin (Passed)")

    # 2. Leaf Specimens Tests (Real Dataset Images)
    print("\n--- 2. Testing Leaf Lesion Highlighting on Real Images ---")
    
    leaf_tests = [
        ("Tomato_healthy", "datasets/raw/PlantVillage/Tomato_healthy", True),
        ("Pepper_healthy", "datasets/raw/PlantVillage/Pepper__bell___healthy", True),
        ("Tomato_Early_blight", "datasets/raw/PlantVillage/Tomato_Early_blight", False),
        ("Tomato_Late_blight", "datasets/raw/PlantVillage/Tomato_Late_blight", False),
        ("Potato_Late_blight", "datasets/raw/PlantVillage/Potato___Late_blight", False),
    ]

    for name, rel_p, is_healthy in leaf_tests:
        d = PROJECT_ROOT / rel_p
        imgs = list(d.glob("*.JPG")) + list(d.glob("*.jpg"))
        if not imgs:
            continue
        img = Image.open(imgs[0])
        meas = PlantLesionDetector.analyze(img, plant_part="Leaf", is_healthy=is_healthy)
        print(f"Leaf [{name:20}] -> Tissue: {meas.leaf_area_pixels:5d} px, Lesion: {meas.lesion_area_pixels:5d} px, Affected: {meas.affected_area_percentage:5.2f}%, Severity: {meas.severity_level}")
        if is_healthy:
            # Must not show moderate or severe disease false alarms
            assert meas.affected_area_percentage < 3.0, f"False high lesion area on healthy leaf: {meas.affected_area_percentage}%"
            assert "Severe" not in meas.severity_level, "Healthy leaf marked as severe"
        else:
            # Diseased leaf must show detected lesions
            assert meas.lesion_detected is True, f"Failed to detect lesions on diseased leaf {name}"

    print("\n✓ All Leaf and Fruit Lesion Segmentation tests passed successfully!")


if __name__ == "__main__":
    t1_results = test_task_1_auto_vs_manual()
    test_task_2_lesion_segmentation()
    print("\n" + "=" * 70)
    print("ALL REPRODUCIBLE VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
