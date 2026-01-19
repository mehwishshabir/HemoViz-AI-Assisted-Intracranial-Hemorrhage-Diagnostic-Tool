# HemoViz-AI-Assisted-Intracranial-Hemorrhage-Diagnostic-Tool
**MS Thesis Project**

---
## 📌 Project Overview
HemoViz is an end-to-end AI-guided pipeline designed for the automated management of **Spontaneous and Traumatic Intracranial Hemorrhage**. The system provides a clinical decision-support interface that classifies CT scans and performs detailed segmentations to assist in rapid neurosurgical intervention.

This repository showcases the **HemoViz Web Interface** and the **System Architecture**. To protect unpublished research findings and proprietary model logic, the core weights and raw datasets are kept private.

---

## 📺 Project Demonstration
Instead of browsing raw code, you can view the full functionality of the pipeline here:

* **[Watch the HemoViz Demo Video](assets/demo_video.mp4)**
* **[View System Workflow](assets/workflow_pipeline.png)**

---

## 🛠 Technical Workflow
The HemoViz protocol follows a modular architecture:

1.  **Preprocessing & Input Handling:** Automated DICOM to NIfTI conversion and intensity normalization.
2.  **3D CNN Classification:** A binary classifier that screens for hemorrhagic vs. normal scans with a confidence threshold > 0.5.
3.  **3D Segmentation (V-Net):** High-fidelity segmentation of bleeding regions to determine volume and location.
4.  **Categorization Engine:** Classifies hemorrhages as **Intra-Axial**, **Extra-Axial**, or **Both** to guide clinical protocols.

---

## 📂 Repository Structure
```text
HemoViz-AI-Hemorrhage-Management/
├── assets/                  # Demo videos and workflow diagrams
├── interface/               # UI/Web application code (unified_app.py)
├── utils/                   # DICOM processing and helper functions
├── .gitignore               # Configured to protect models and data
├── LICENSE                  # CC BY-NC-ND 4.0
└── README.md                # Project documentation
