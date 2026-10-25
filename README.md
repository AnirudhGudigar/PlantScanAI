# 🌿 PlantScan AI

> **Multi-Crop Disease Diagnosis & Integrated Pest Management (IPM) Advisory System using Deep Learning & Mobile-Cloud Architecture**

---

## 🚀 Quick Execution (1-Click Batch Scripts)

On Windows, you can double-click or run these batch scripts directly in your terminal:

- 🧪 **Run Automated Unit Tests**: `.\run_tests.bat`
- 🌐 **Start FastAPI Server**: `.\run_backend.bat` (`http://localhost:8000/docs`)
- 🔍 **Run Disease Predictor CLI**: `.\run_predictor.bat`

---

## 📚 Complete Project Documentation

For detailed technical design and step-by-step execution guides, please see:

- 📖 **[Project Technical Overview](PROJECT_OVERVIEW.md)**: Full architecture breakdown, 10 crops, 46 canonical classes, dataset deduplication, and ML pipeline details.
- 🛠️ **[Step-by-Step Execution Guide](HOW_TO_RUN.md)**: Complete guide to running unit tests, backend API, model inference CLI, and mobile client.

---

## ⚡ Technical Highlights

- **10 Agricultural Crops**: Apple, Banana, Bell Pepper, Cotton, Grape, Maize, Mango, Potato, Rice, Tomato.
- **46 Canonical Classes**: Fine-grained disease classifications + healthy leaf detection.
- **Integrated Pest Management (IPM)**: Organic, biological, and chemical treatment guidelines with PPE safety rules.
- **Dual Inference Engine**: High-accuracy server-side Keras models + lightweight edge TensorFlow Lite (`.tflite`) models.
- **Out-of-Scope Rejection**: Confidence thresholding ($<70\%$) for low-certainty inputs.
- **Production REST API**: FastAPI asynchronous backend with automated Pytest test suite (16/16 tests passing).