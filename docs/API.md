# PlantScan AI — Backend API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000` (Local) / `/api/v1` (Versioned Prefix)  
**OpenAPI / Interactive Swagger UI:** `http://localhost:8000/docs`  
**ReDoc Specification:** `http://localhost:8000/redoc`  
**OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## 1. Overview

The **PlantScan AI Backend** provides high-performance, modular endpoints for crop disease diagnosis, plant part classification, out-of-scope/uncertainty detection, and sustainable **Integrated Pest Management (IPM)** treatment recommendations.

### Key Architectural Principles
- **Engineering First**: Clean separation of API routing, service logic, inference engines, and knowledge bases.
- **Organic First**: Non-chemical cultural, biological, and preventive measures are prioritized before synthetic chemicals.
- **Responsible Diagnosis**: Ambiguous or out-of-scope images return explicit `uncertain` or `unsupported_crop` statuses rather than false confidence.
- **Scientific Sourcing**: All treatment advisories cite agricultural university extension manuals (e.g., ICAR, FAO, UC Davis IPM).

---

## 2. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service status and documentation links |
| `GET` | `/api/v1/health` | Service health, model status, and database metrics |
| `GET` | `/api/v1/info` | Supported crop list, confidence thresholds, and constraints |
| `POST` | `/api/v1/predict` | **Primary:** Diagnose plant disease from multipart image upload |
| `POST` | `/api/v1/predict/json` | Diagnose plant disease from Base64 encoded JSON string |
| `GET` | `/api/v1/treatments/crops` | List all supported crop species |
| `GET` | `/api/v1/treatments/classes` | List all canonical disease and healthy class keys |
| `GET` | `/api/v1/treatments/languages` | List supported localization language codes (`en`, `hi`, `kn`) |
| `GET` | `/api/v1/treatments/{crop}/{disease}` | Query IPM treatment protocol for a specific disease |

---

## 3. Detailed Endpoint Specifications

### 3.1 POST `/api/v1/predict` (Multipart Upload)

Upload a plant leaf/crop photograph for AI diagnosis.

#### Request Parameters (Multipart Form-Data)

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | `binary` | **Yes** | Image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`). Max: **10 MB**. Min resolution: **32x32 px**. |
| `crop_hint` | `string` | No | Target crop hint (e.g., `"Tomato"`, `"Rice"`, `"Potato"`, `"Banana"`, `"Cotton"`, `"Mango"`). |
| `plant_part_hint` | `string` | No | Anatomy hint (`"Leaf"`, `"Fruit"`, `"Stem"`, `"Bark"`). Default: `"Leaf"`. |
| `language` | `string` | No | Language code for response (`"en"`, `"kn"`, `"hi"`). Default: `"en"`. |

#### Response Schema (HTTP 200 OK)

```json
{
  "status": "success",
  "crop": "Tomato",
  "predicted_class": "tomato_early_blight",
  "display_name": "Tomato Early Blight",
  "confidence": 0.9412,
  "is_healthy": false,
  "plant_part": "Leaf",
  "plant_health_score": 41,
  "recommendation": {
    "summary": "Identified Tomato Early Blight with 94.1% confidence.",
    "organic": [
      {
        "name": "Copper Hydroxide / Bordeaux Mixture Spray",
        "method": "Apply copper-based organic fungicide (1-2 g per liter) thoroughly on both leaf surfaces.",
        "application": "Foliar spray during cool morning or evening hours",
        "frequency": "Every 7 to 10 days until disease progression halts",
        "active_ingredient": null,
        "dosage": null,
        "safety_precautions": null,
        "caution_level": "Low"
      },
      {
        "name": "Bio-fungicide (Trichoderma viride / Bacillus subtilis)",
        "method": "Spray Trichoderma viride or Bacillus subtilis formulation (5 g/ml per liter of water) to outcompete fungal spores.",
        "application": "Soil drenching and foliar spray",
        "frequency": "Every 10 to 14 days",
        "active_ingredient": null,
        "dosage": null,
        "safety_precautions": null,
        "caution_level": "Low"
      }
    ],
    "chemical": [
      {
        "name": "Mancozeb 75% WP",
        "method": null,
        "application": null,
        "frequency": null,
        "active_ingredient": "Mancozeb",
        "dosage": "2.0 - 2.5 g per liter of water",
        "safety_precautions": "Wear PPE (gloves, mask, goggles). Do not spray during bloom period. Pre-harvest interval (PHI): 7 days.",
        "caution_level": "Moderate"
      }
    ],
    "prevention": [
      "Practice 2-3 year crop rotation with non-solanaceous crops.",
      "Use drip irrigation instead of overhead watering to keep foliage dry.",
      "Apply organic straw or plastic mulch around the plant base to prevent soil-splash."
    ],
    "safety_notes": [
      "DANGER / CAUTION: Synthetic chemicals and pesticides must be handled strictly according to manufacturer label instructions and local regulatory approvals. Always wear appropriate Personal Protective Equipment (PPE).",
      "Organic and biological controls are recommended as first-line measures under Integrated Pest Management (IPM)."
    ],
    "photo_guidance": null,
    "disclaimer": "This advisory is generated by PlantScan AI for preliminary screening and educational purposes. Always consult your local agricultural extension service (KVK) or certified agronomist before major farm management decisions.",
    "sources": [
      "ICAR - Indian Institute of Horticultural Research (IIHR) Tomato Disease Guide",
      "University of Florida IFAS Extension: Early Blight of Tomato",
      "FAO Integrated Pest Management in Vegetable Production"
    ]
  },
  "model_version": "MobileNetV2-PlantScan-v1",
  "inference_time_ms": 24.5,
  "message": "Diagnosis completed successfully."
}
```

---

### 3.2 Uncertain / Low-Confidence Handling (`status: "uncertain"`)

When model diagnostic certainty is below the confidence threshold (`0.70`), the API safely rejects guesswork:

```json
{
  "status": "uncertain",
  "crop": "Tomato",
  "predicted_class": "tomato_early_blight",
  "display_name": "Uncertain Diagnosis",
  "confidence": 0.45,
  "is_healthy": false,
  "plant_part": "Leaf",
  "plant_health_score": 50,
  "recommendation": {
    "summary": "Diagnostic confidence (45.0%) is below the 70% threshold. Visual symptoms are ambiguous. Do not apply chemical pesticides without physical verification.",
    "organic": [
      {
        "name": "General Field Hygiene",
        "method": "Inspect adjacent plants for similar symptoms. Prune any visibly rotten or dead plant material using sterilized tools.",
        "application": "Morning hours",
        "frequency": "As needed during weekly monitoring",
        "caution_level": "Low"
      }
    ],
    "chemical": [],
    "prevention": [
      "Capture a clear, well-focused close-up photo of the affected plant part in natural daylight.",
      "Avoid applying broad-spectrum synthetic chemical pesticides without verified diagnostic confirmation.",
      "Take fresh physical plant samples to your nearest agricultural extension center or KVK."
    ],
    "safety_notes": [
      "When AI diagnostic confidence is below the threshold or visual features are ambiguous, avoid treating based on presumption."
    ],
    "photo_guidance": [
      "Ensure good natural daylight (avoid direct harsh midday sun or deep shadows).",
      "Focus sharply on the lesion or affected area.",
      "Include both the top surface and underside of the leaf if possible.",
      "Avoid taking photos with messy backgrounds, hands covering symptoms, or heavy motion blur."
    ],
    "disclaimer": "This advisory is generated by PlantScan AI for preliminary screening and educational purposes. Always consult your local agricultural extension service...",
    "sources": [
      "ICAR Extension Field Handbook",
      "FAO Integrated Pest Management Guidelines"
    ]
  },
  "model_version": "MobileNetV2-PlantScan-v1",
  "inference_time_ms": 18.2,
  "message": "Low diagnostic certainty. Follow the photo tips to retake a clear photo in natural light."
}
```

---

### 3.3 Error Responses

#### Corrupted / Non-Image File (HTTP 400 Bad Request)
```json
{
  "status": "invalid_image",
  "error_code": "INVALID_IMAGE",
  "message": "The uploaded file is not a recognized image or is severely corrupted.",
  "hint": "Ensure you are uploading a valid JPEG, PNG, BMP, or WebP image."
}
```

#### Unsupported File Type (HTTP 400 Bad Request)
```json
{
  "status": "invalid_image",
  "error_code": "INVALID_IMAGE",
  "message": "Unsupported file format '.txt'. Allowed formats: .bmp, .jpeg, .jpg, .png, .webp.",
  "hint": "Convert your image to JPEG or PNG before uploading."
}
```

---

## 4. Usage Examples

### 4.1 cURL — Multipart Image Upload
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Accept: application/json" \
     -F "file=@/path/to/tomato_leaf.jpg" \
     -F "crop_hint=Tomato"
```

### 4.2 Python (`requests`)
```python
import requests

url = "http://localhost:8000/api/v1/predict"
with open("sample_leaf.jpg", "rb") as f:
    files = {"file": ("sample_leaf.jpg", f, "image/jpeg")}
    data = {"crop_hint": "Tomato", "language": "en"}
    response = requests.post(url, files=files, data=data)

result = response.json()
print(f"Status: {result['status']}")
print(f"Disease: {result['display_name']} (Confidence: {result['confidence']*100:.1f}%)")
print(f"Plant Health Score: {result['plant_health_score']}/100")

# Display Organic Remedies First
print("\n--- Organic Remedies ---")
for remedy in result["recommendation"]["organic"]:
    print(f"• {remedy['name']}: {remedy['method']}")
```

---

## 5. Running the Backend Server

```bash
# From the repository root (PlantScanAI/):
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6. Running Automated Tests

```bash
python -m pytest backend/tests -v
```
