# CardioVision3D

<p align="center">
  <strong>AI-Powered Cardiac Risk Prediction with Interactive 3D Visualization</strong>
</p>

<p align="center">
  An educational machine learning web application for estimating coronary artery disease (CAD) risk and vessel-specific stenosis probabilities using clinical data, integrated with an interactive 3D heart model.
</p>

<p align="center">
  <a href="https://preyapatel07-cardiovision3d-streamlit-app-mrgxe5.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
  <a href="https://github.com/preyapatel2007/CardioVision3D">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-Machine%20Learning-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Visualization-Three.js-black?logo=threedotjs" alt="Three.js">
</p>

---

## Overview

**CardioVision3D** combines machine learning and interactive 3D visualization to make cardiac risk estimates easier to explore and understand.

The application uses clinical, demographic, physiological, electrocardiographic (ECG), laboratory, and echocardiographic features to estimate:

* **Coronary Artery Disease (CAD):** Overall CAD probability.
* **LAD:** Left Anterior Descending artery stenosis probability.
* **LCX:** Left Circumflex artery stenosis probability.
* **RCA:** Right Coronary Artery stenosis probability.

The web interface presents these model outputs alongside an interactive 3D heart model that users can rotate and zoom.

> **Important:** CardioVision3D is an educational and research prototype. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical advice.

## Live Demo

**[Launch CardioVision3D](https://preyapatel07-cardiovision3d-streamlit-app-mrgxe5.streamlit.app/)**

The public demo runs through Streamlit Community Cloud.

## Key Features

### 1. Machine Learning-Based Predictions

* Estimates overall CAD probability.
* Estimates stenosis probabilities for LAD, LCX, and RCA.
* Uses saved machine learning models and their preprocessing pipelines.
* Maintains the trained 55-feature input schema.

### 2. Interactive 3D Heart Visualization

* Renders a 3D heart model using Three.js and WebGL.
* Supports rotation and zoom controls.
* Provides an anatomical visual reference alongside prediction results.

The current 3D mesh is a general anatomical visualization, not a patient-specific anatomical reconstruction or a precise map of an individual patient's stenosis.

### 3. Clinical Input Interface

* Organizes inputs into patient demographics, medical history, clinical findings, ECG, laboratory results, and echocardiography.
* Allows users to leave unavailable measurements blank.
* Uses the trained model's preprocessing pipeline to process input features.

Missing information may reduce prediction reliability.

### 4. Model Evaluation

The project includes saved evaluation metrics to support model assessment, including accuracy, precision, recall, F1-score, and ROC-AUC.

### 5. Two Application Interfaces

The repository contains both:

* A **Streamlit application** for public deployment and interactive exploration.
* An original **Flask-based web application** with HTML, CSS, JavaScript, and backend API functionality.

---

## Technology Stack

| Category               | Technologies              |
| ---------------------- | ------------------------- |
| Programming Language   | Python, JavaScript        |
| Machine Learning       | scikit-learn              |
| Data Processing        | NumPy, pandas             |
| Model Persistence      | Joblib                    |
| Streamlit Interface    | Streamlit                 |
| Original Web Interface | HTML, CSS, JavaScript     |
| 3D Visualization       | Three.js, WebGL, GLB/glTF |
| Backend                | Flask                     |
| Version Control        | Git, GitHub               |
| Deployment             | Streamlit Community Cloud |

## Machine Learning Methodology

The prediction pipeline is based on the extension of the Z-Alizadeh Sani dataset.

The workflow includes:

1. Loading and inspecting clinical data.
2. Separating predictive input features from target labels.
3. Preprocessing numeric and categorical features.
4. Training and evaluating classification models.
5. Saving trained model artifacts and the feature schema.
6. Loading the trained models in the web application.
7. Generating probability estimates from user-provided information.

### Prediction Targets

| Target | Description                       |
| ------ | --------------------------------- |
| CAD    | Overall coronary artery disease   |
| LAD    | Left Anterior Descending stenosis |
| LCX    | Left Circumflex stenosis          |
| RCA    | Right Coronary Artery stenosis    |

The target and angiography-result columns are excluded from predictive inputs to prevent target leakage.

## Model Evaluation

The following are the recorded holdout evaluation results from the project's model-training experiment.

| Target | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------ | -------: | --------: | -----: | -------: | ------: |
| CAD    |   86.89% |    88.89% | 93.02% |   90.91% |  86.56% |
| LAD    |   70.49% |    71.05% | 79.41% |   75.00% |  78.05% |
| LCX    |   70.49% |    72.22% | 50.00% |   59.09% |  71.32% |
| RCA    |   67.21% |    50.00% | 60.00% |   54.55% |  72.01% |

**Evaluation notes**

* The reported experiment used 242 training records and 61 test records.
* The test results are specific to the selected dataset split.
* Performance varies by prediction target, particularly for LCX and RCA.
* These metrics do not establish clinical safety, diagnostic accuracy in real-world populations, or generalizability to new hospitals or patient groups.

## Dataset

The project uses the **[Extension of the Z-Alizadeh Sani Dataset](https://archive.ics.uci.edu/dataset/411/extention%2Bof%2Bz%2B)** from the UCI Machine Learning Repository.

The original dataset contains 303 patient records and clinical attributes across demographic, symptom and examination, ECG, laboratory, and echocardiographic categories.

The model uses 55 input features and predicts CAD and vessel-specific target labels.

### Data Handling

The raw dataset is intentionally excluded from this GitHub repository. Download it from the official UCI source and place it in the location expected by `ml/train_model.py` if you need to retrain the models.

Do not upload identifiable patient information or confidential medical records.

## Repository Structure

```text
CardioVision3D/
│
├── streamlit_app.py
├── server.py
├── index.html
├── app.js
├── style.css
├── requirements.txt
├── README.md
│
├── ml/
│   └── train_model.py
│
├── models/
│   ├── cardiovision_model.joblib
│   ├── feature_columns.json
│   ├── metrics.json
│   └── README.txt
│
├── anatomy/
│   └── heart.glb
│
├── data/                  # Local dataset; excluded from Git
└── reports/               # Generated reports; excluded from Git
```

## Run Locally

### Prerequisites

* Python installed on your computer.
* Git installed.
* Internet access for installing dependencies and loading external 3D viewer libraries.

### 1. Clone the Repository

```bash
git clone https://github.com/preyapatel2007/CardioVision3D.git
cd CardioVision3D
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Use a Python version compatible with the dependency versions specified in `requirements.txt`.

### 4. Run the Streamlit Application

```bash
python -m streamlit run streamlit_app.py
```

Open the local address displayed by Streamlit, usually:

`http://localhost:8501`

### 5. Run the Original Flask Application

To launch the original Flask interface locally:

```bash
python server.py
```

Open the local address printed by the Flask server, typically:

`http://127.0.0.1:5000`

## Model Artifacts

The `models/` directory contains the trained model, input feature schema, and saved evaluation metrics.

| File                        | Purpose                                   |
| --------------------------- | ----------------------------------------- |
| `cardiovision_model.joblib` | Serialized trained model artifact         |
| `feature_columns.json`      | Expected input feature names and ordering |
| `metrics.json`              | Saved evaluation metrics                  |
| `README.txt`                | Model directory notes                     |

The raw dataset and generated patient reports are excluded through `.gitignore`.

## Limitations and Future Improvements

Potential areas for further development include:

* Adding model explainability through SHAP or LIME.
* Improving the mapping between individual coronary arteries and the 3D anatomical model.
* Conducting further testing with independent datasets.
* Evaluating robustness when clinical fields are missing.
* Improving accessibility, responsive design, and automated testing.
* Developing secure, persistent storage for reports where appropriate.

These are potential enhancements, not claims that all of these features are currently implemented.

## Responsible Use

CardioVision3D is intended for educational demonstration, academic projects, and machine learning experimentation.

* Predictions are model-generated estimates, not medical diagnoses.
* A displayed probability is not a guarantee that a condition is present or absent.
* Missing inputs can affect prediction reliability.
* The 3D heart model is for visualization and does not identify a patient's exact lesion location.
* Clinical decisions must be made by qualified healthcare professionals using appropriate medical evaluation.

## Dataset Attribution

Alizadehsani, R., Roshanzamir, M., & Sani, Z. (2013). *Extension of Z-Alizadeh Sani Dataset*. UCI Machine Learning Repository.

Dataset page: https://archive.ics.uci.edu/dataset/411/extention%2Bof%2Bz%2B

DOI: https://doi.org/10.24432/C5461K

## Author

**Preya Patel**

GitHub: [@preyapatel2007](https://github.com/preyapatel2007)

Project repository: [CardioVision3D](https://github.com/preyapatel2007/CardioVision3D)

---

<p align="center">
  Built for learning, experimentation, and responsible exploration of machine learning in cardiac risk assessment.
</p>
