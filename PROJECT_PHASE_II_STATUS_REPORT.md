# National Education Society (R.)
## J. N. N. College of Engineering, Shivamogga
*(Approved by AICTE, New Delhi, Certified by UGC 2f & 12B, Accredited by NAAC – 'A',*  
*All 7 UG Programs have been Accredited by National Board of Accreditation (NBA) for period 1.7.2022 to 30.6.2025,*  
*Recognized by Govt. of Karnataka and Affiliated to VTU, Belagavi)*

### **Department of Computer Science and Engineering**

---

# **PROJECT PHASE – II STATUS REPORT**

**Date:** September 2, 2026  

---

### **1. Basic Details**

| Field | Information |
| :--- | :--- |
| **Batch Number:** | Batch-12 (CSE-Phase II) |
| **Project Title:** | **PlantScan AI: Multi-Crop Disease Diagnosis and Integrated Pest Management (IPM) Advisory System using Deep Learning & Mobile-Cloud Architecture** |
| **Student Name(s) with USN(s):** | 1. Sagar Nag (USN: `4JN22CS___`) <br> 2. [Team Member 2 Name] (USN: `4JN22CS___`) <br> 3. [Team Member 3 Name] (USN: `4JN22CS___`) |
| **Project Guide:** | Prof. [Guide Name], Dept. of Computer Science & Engineering |

---

### **2. Project Status & Progress Breakdown**

| Status | Details / Remarks | Progress (%) |
| :--- | :--- | :---: |
| **Completed Modules** | **Module 1: Dataset Acquisition, Audit & Clean Pipeline**<br>• Processed 35,264 raw images across 6 source datasets (*PlantVillage, PlantDoc, Rice, Banana, Cotton, Mango*).<br>• Performed near-duplicate audit (excluded 258 duplicates) and implemented SHA-256 hashing protection.<br>• Built reproducible 80/10/10 train/validation/test dataset splits for 10 crops (34,314 images across 46 canonical classes).<br><br>**Module 2: Deep Learning Model Training & Optimization**<br>• Developed generalized multi-crop MobileNetV2 transfer learning pipeline (`experiments/train_crop_mobilenetv2.py`).<br>• Implemented balanced class weighting (`sklearn.utils.class_weight`) to tackle dataset imbalances.<br>• Quantized models into FP16/INT8 TensorFlow Lite (`.tflite`) format for edge deployment.<br><br>**Module 3: Integrated Pest Management (IPM) Knowledge Base**<br>• Created comprehensive JSON knowledge repository (`treatment_database/`) for all 10 crops (*Tomato, Mango, Rice, Bell Pepper, Banana, Potato, Cotton, Maize, Apple, Grape*).<br>• Prioritized non-chemical organic & biological remedies, followed by chemical interventions (with PHI and PPE safety warnings), cultural prevention, and university extension citations (ICAR, FAO, UC Davis).<br><br>**Module 4: Production RESTful Backend API**<br>• Developed high-performance FastAPI service (`backend/main.py`) with Swagger UI (`/docs`).<br>• Integrated `ImageProcessor` for EXIF orientation correction, dimension checks, and input sanitization.<br>• Designed dual inference engine architecture (`TFLiteModelEngine` + `MockModelEngine`) with confidence threshold rejection (<70% uncertain status).<br>• Passed complete automated unit test suite (16/16 pytest cases passing). | **100%** |
| **Remaining Modules** | **Module 5: Mobile Application Integration & Cross-Platform UI**<br>• Integrating React Native camera module, image picker, and real-time offline TFLite inference execution on edge devices.<br>• Connecting API client for cloud advisory sync and dynamic Plant Health Score display.<br><br>**Module 6: Field Testing, Agronomist Validation & Cloud Deployment**<br>• Conducting field validation trials with live plant leaf samples under varying daylight/shadow conditions.<br>• Containerizing backend with Docker and deploying to cloud infrastructure for low-latency inference. | **70%** |
| **Overall Remarks** | **The core AI model pipeline, multi-crop dataset architecture (10 crops, 46 classes), IPM treatment repository, and production REST API backend are fully completed and verified.** The mobile application edge integration and field validation trials are in final testing. The project is strictly on schedule for final Phase-II presentation. | **90%** |

---

### **3. Signatures**

<br><br>

| **Student Signature** | **Project Guide Signature** | **Project Coordinator Signature** |
| :---: | :---: | :---: |
| <br><br>___________________________<br>**(Student Name)** | <br><br>___________________________<br>**(Project Guide)** | <br><br>___________________________<br>**(Project Coordinator)** |

---
