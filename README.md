# HemoViz: AI-Assisted Intracranial Hemorrhage Diagnostic Tool
**MS Thesis Project | Bioinformatics & Medical Deep Learning**

---

## 📌 Project Overview
HemoViz is an end-to-end AI-guided pipeline designed for the automated management of **Spontaneous and Traumatic Intracranial Hemorrhage**. The system provides a clinical decision-support interface that classifies CT scans and performs detailed segmentations to assist in rapid neurosurgical intervention.

This repository showcases the **HemoViz Web Interface** and the **System Architecture**. To protect unpublished research findings and proprietary model logic, the core weights and raw datasets are kept private.

---

## 📊 Project Presentation & Demonstration
Since this project involves unpublished MS Thesis research, the full methodology and application walkthrough are consolidated into a professional presentation.

* 📄 **[View HemoViz Project Presentation (PDF)](HemoViz_Presentation.pdf)**
  * *Slide 1: Technical Workflow and Architectural Pipeline*
  * *Slide 2: System Demonstration (Includes Link to Google Drive Video)*

> **Note:** Open the PDF to access the interactive video link. The demonstration shows the transition from 3D CT upload to final clinical protocol generation.

---

## 🛠 Technical Workflow
The HemoViz protocol follows a modular architecture:

1. **Preprocessing & Input Handling:** Automated DICOM to NIfTI conversion and intensity normalization.
2. **3D CNN Classification:** A binary classifier that screens for hemorrhagic vs. normal scans with a confidence threshold > 0.5.
3. **3D Segmentation (V-Net):** High-fidelity segmentation of bleeding regions to determine volume and location.
4. **Categorization Engine:** Classifies hemorrhages as **Intra-Axial**, **Extra-Axial**, or **Both** to guide clinical protocols.

---

## 📂 Repository Structure
```text
HemoViz-AI-Assisted-Intracranial-Hemorrhage-Diagnostic-Tool/
├── assets/                  # CSS styling for interface and workflow images
├── interface/               # UI/Web application code (unified_app.py)
├── utils/                   # DICOM processing and helper functions
├── .gitignore               # Configured to protect models and data
├── LICENSE                  # CC BY-NC-ND 4.0
├── README.md                # Project documentation
└── HemoViz_Presentation.pdf # Presentation with embedded demo link
