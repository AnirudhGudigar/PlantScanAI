# National Education Society (R.)
## J. N. N. College of Engineering, Shivamogga
*(Approved by AICTE, New Delhi, Certified by UGC 2f & 12B, Accredited by NAAC – 'A',*  
*All 7 UG Programs have been Accredited by National Board of Accreditation (NBA) for period 1.7.2022 to 30.6.2025,*  
*Recognized by Govt. of Karnataka and Affiliated to VTU, Belagavi)*

### **Department of Computer Science and Engineering**

---

# **PROJECT PHASE – III STATUS REPORT**

**Date:** October 22, 2026  

---

### **1. Basic Details**

| Field | Information |
| :--- | :--- |
| **Batch Number:** | Batch-12 (CSE-Phase III) |
| **Project Title:** | **PlantScan AI: Multi-Crop Disease Diagnosis and Multilingual Integrated Pest Management (IPM) Advisory System using Deep Learning & Mobile-Cloud Architecture** |
| **Student Name(s) with USN(s):** | 1. Sagar Nag (USN: `4JN22CS___`) <br> 2. [Team Member 2 Name] (USN: `4JN22CS___`) <br> 3. [Team Member 3 Name] (USN: `4JN22CS___`) |
| **Project Guide:** | Prof. [Guide Name], Dept. of Computer Science & Engineering |

---

### **2. Project Status & Progress Breakdown**

| Status | Details / Remarks | Progress (%) |
| :--- | :--- | :---: |
| **Completed Modules (Phases I, II & III)** | **Module 1: Dataset Architecture & Data Governance**<br>• Curated 35,000+ images covering 10 major commercial crops (*Tomato, Banana, Cotton, Mango, Rice, Potato, Bell Pepper, Maize, Apple, Grape*) across 46 canonical classes.<br>• Documented dataset taxonomy, split ratios (70/15/15), and preprocessing pipelines in [`docs/Dataset.md`](file:///d:/Sagarnag/major%20project/pgpt/PlantScanAI/docs/Dataset.md).<br><br>**Module 2: Standardized MobileNetV2 Deep Learning Pipeline**<br>• Standardized classification heads to `Dense(num_classes, activation="softmax")` across all crops, resolving binary sigmoid 1D argmax edge failure.<br>• Automated generation of class manifest artifacts (`classes.json`) alongside `.tflite` model conversion.<br>• Successfully retrained Grape model (accuracy boosted from **42.8% to 78.57%**, F1: 0.785) and Apple model (**64.29%**, Val loss: 0.521).<br>• Fully documented model architecture and evaluation in [`docs/Training.md`](file:///d:/Sagarnag/major%20project/pgpt/PlantScanAI/docs/Training.md).<br><br>**Module 3: Production FastAPI Backend & Concurrency Hardening**<br>• Re-architected model inference with `threading.Lock()` to ensure TFLite interpreter thread safety.<br>• Offloaded synchronous CPU image processing and TFLite execution to threadpools (`run_in_threadpool`) preventing async event loop blockages.<br>• Hardened security: gated test simulation headers (`ENABLE_SIMULATION_HEADERS`) and secured CORS policy against wildcard credential vulnerabilities.<br>• Created comprehensive deployment and containerization documentation in [`docs/Deployment.md`](file:///d:/Sagarnag/major%20project/pgpt/PlantScanAI/docs/Deployment.md).<br><br>**Module 4: Multilingual IPM Localization (Kannada, Hindi, English)**<br>• Built complete translation dictionaries (`treatment_database/translations/` for `kn`, `hi`, `en`).<br>• Implemented dynamic localized advisory assembly for crop names, safety notices, diagnostic summaries, photo guidance, and general disclaimers.<br>• Added `/api/v1/treatments/languages` endpoint and supported `language` query/form parameters across all prediction routes.<br><br>**Module 5: Automated Test Suite & Code Quality**<br>• 23/23 unit and integration tests passing in **~7.0 seconds** (down from initial 79s unoptimized run).<br>• Cleaned empty legacy boilerplate directories (`backend/app`). | **100%** |
| **Final Integration & Delivery** | **Module 6: Mobile Client Packaging & Final Defense Preparation**<br>• Bundling retrained `.tflite` weights and `classes.json` manifests into mobile offline asset bundle.<br>• Finalizing field trial demonstration videos and project defense presentation slides. | **85%** |
| **Overall Remarks** | **The entire core AI architecture, standardized model training pipeline, thread-safe production backend, multilingual IPM repository, and complete technical documentation are 100% completed and verified.** The test suite is fully green with 23 passing tests. The project is fully on schedule for final defense. | **95%** |

---

### **3. Key Engineering Metrics & Milestones Achieved**

- **Automated Tests Passing**: 23 / 23 (100% pass rate in 7.06s)
- **Supported Crops**: 10 (*Tomato, Banana, Cotton, Mango, Rice, Potato, Bell Pepper, Maize, Apple, Grape*)
- **Canonical Classes Mapped**: 46 / 46 (100% treatment database coverage)
- **Supported Localized Languages**: English (`en`), Kannada (`kn`), Hindi (`hi`)
- **Model Size per Crop**: ~8.7 MB – 8.9 MB (optimized MobileNetV2 TFLite)
- **Inference Latency**: <50 ms per image prediction on CPU

---

### **4. Signatures**

<br><br>

| **Student Signature** | **Project Guide Signature** | **Project Coordinator Signature** |
| :---: | :---: | :---: |
| <br><br>___________________________<br>**(Student Name)** | <br><br>___________________________<br>**(Project Guide)** | <br><br>___________________________<br>**(Project Coordinator)** |

---
