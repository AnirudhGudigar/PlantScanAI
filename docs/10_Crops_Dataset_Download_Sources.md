# PlantScan AI — 10-Crop Master Dataset Download Sources & References

This document provides official, verified links and source repositories for downloading the datasets corresponding to the **10 crops** supported by PlantScan AI.

---

## 🌾 Master Dataset Summary Table (10 Crops)

| # | Crop Name | Primary Classes Included | Primary Sources & Repositories | Direct Access / Download Links |
|---|---|---|---|---|
| 1 | **Tomato** | 10 Classes (Early Blight, Late Blight, Bacterial Spot, Septoria, Leaf Mold, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy) | PlantVillage, PlantDoc, New Plant Diseases | [Kaggle PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) <br> [PlantDoc GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) |
| 2 | **Banana** | 4 Classes (Cordana, Pestalotiopsis, Sigatoka, Healthy) | Mendeley Data Banana Leaf Collection | [Mendeley Banana Dataset 1](https://data.mendeley.com/datasets/wfzpdmc5vx/1) <br> [Mendeley Banana Dataset 2](https://data.mendeley.com/datasets/f6p4b53m79/1) |
| 3 | **Cotton** | 4 Classes (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy) | Mendeley Data Cotton Leaf & Kaggle | [Mendeley Cotton Dataset](https://data.mendeley.com/datasets/d765z4z36v/1) <br> [Kaggle Cotton Disease](https://www.kaggle.com/datasets/janmejaybhatt/cotton-disease-dataset) |
| 4 | **Mango** | 8 Classes (Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould, Healthy) | Mendeley Data Mango Leaf BD | [Mendeley Mango Dataset 1](https://data.mendeley.com/datasets/j3bn63t4sp/1) <br> [Mendeley Mango Dataset 2](https://data.mendeley.com/datasets/3795b87v6j/1) |
| 5 | **Rice** | 6 Classes (Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, Healthy) | Rice Disease Benchmark & Paddy Doctor | [Kaggle Rice Diseases](https://www.kaggle.com/datasets/minhhuynguyen/rice-leaf-disease-images) <br> [Paddy Doctor Competition](https://www.kaggle.com/c/paddy-disease-classification) |
| 6 | **Potato** | 3 Classes (Early Blight, Late Blight, Healthy) | PlantVillage & PlantDoc | [Kaggle New Plant Diseases](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) <br> [PlantDoc Repository](https://github.com/pratikkayal/PlantDoc-Dataset) |
| 7 | **Bell Pepper** | 3 Classes (Bacterial Spot, Cercospora Leaf Spot, Healthy) | PlantVillage & PlantDoc | [Kaggle PlantVillage](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) <br> [PlantDoc GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) |
| 8 | **Maize (Corn)** | 3 Classes (Gray Leaf Spot, Northern Leaf Blight, Common Rust) | PlantVillage & PlantDoc | [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) <br> [PlantDoc Repository](https://github.com/pratikkayal/PlantDoc-Dataset) |
| 9 | **Apple** | 3 Classes (Apple Scab, Cedar Apple Rust, Healthy) | Plant Pathology FGVC & PlantDoc | [Kaggle Plant Pathology 2021](https://www.kaggle.com/c/plant-pathology-2021-fgvc8) <br> [Kaggle Apple Symptoms](https://www.kaggle.com/datasets/arshmahmed/apple-leaf-disease-symptoms-dataset) |
| 10 | **Grape** | 2 Classes (Grape Black Rot, Healthy) | PlantVillage & PlantDoc | [Kaggle Plant Diseases](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) <br> [PlantDoc Repository](https://github.com/pratikkayal/PlantDoc-Dataset) |

---

## 📂 Local Storage Structure in Repository

Once downloaded, place raw crop datasets under `datasets/raw/`:

```
PlantScanAI/
└── datasets/
    ├── raw/
    │   ├── Banana/
    │   ├── Cotton/
    │   ├── Mango/
    │   ├── PlantDoc/
    │   ├── PlantVillage/
    │   └── Rice/
    ├── processed/
    │   └── cleaned_raw/
    └── model/
        ├── apple/
        ├── banana/
        ├── bell_pepper/
        ├── cotton/
        ├── grape/
        ├── maize/
        ├── mango/
        ├── potato/
        ├── rice/
        └── tomato/
```

---

## ⚙️ Automated Preprocessing & Model Building Commands

After downloading raw dataset files into `datasets/raw/`:

```bash
# 1. Clean dataset & exclude audit duplicates (creates datasets/processed/cleaned_raw)
.\.venv\Scripts\python.exe preprocessing/build_clean_dataset.py

# 2. Build 80/10/10 reproducible model splits for ALL 10 crops
.\.venv\Scripts\python.exe preprocessing/build_crop_model_dataset.py --crop all --overwrite

# 3. Train & evaluate MobileNetV2 models across crops
.\.venv\Scripts\python.exe experiments/train_crop_mobilenetv2.py --crop all --epochs 10
```
