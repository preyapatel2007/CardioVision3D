# CardioVision3D — Ready Starter Project

A local web prototype for the Multimodal AI Hackathon 2026, Track A: Cardiovascular Risk Visualization & Prediction.

## Included
- Dark dashboard inspired by the supplied reference image.
- Interactive procedural 3D heart illustration (rotate, zoom, pan, layers, labels).
- Anatomy structure visibility toggles, search, theme toggle, responsive layout.
- Patient input modal and prediction API endpoint.
- Baseline training script for CAD, LAD, LCX, and RCA target classifiers.
- Clinical safety notice and no-model-connected state.

## Requirements
- Python 3.10 or newer.
- Internet connection for the Three.js ES module loaded from jsDelivr the first time the page opens.
- VS Code is recommended.

## Run on Windows
1. Extract this ZIP to a simple folder, for example `D:\\CardioVision3D`.
2. Open that folder in VS Code.
3. Open VS Code Terminal: **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\\.venv\\Scripts\\Activate.ps1
   ```
   If PowerShell blocks activation, use:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\\.venv\\Scripts\\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
6. Start the app:
   ```powershell
   python server.py
   ```
7. Open `http://127.0.0.1:5000` in your browser.

## Train the ML models (after obtaining the dataset)
1. Download the **Extension of Z-Alizadeh Sani Dataset** from the official UCI repository or its authorized mirror. Check its terms and cite the dataset in your report.
2. Keep the dataset file locally; do not upload patient data to a public repository.
3. Inspect its column headers. The script expects targets named `CAD`, `LAD`, `LCX`, `RCA` (case-insensitive). If your file uses different names, edit `TARGETS` in `ml/train_model.py`.
4. Ensure that the exact target columns and `Cath` are excluded from input features. The starter script drops those columns from X to avoid target leakage.
5. Run:
   ```powershell
   python ml/train_model.py "D:\\path\\to\\your_dataset.csv"
   ```
6. The script writes `models/cardiovision_model.joblib`, `models/metrics.json`, and `models/feature_columns.json`.
7. Restart `python server.py`. The app will try to use the saved models.

## Important integration note
The clinical input form is a UI starter, not yet a schema-matched form for every version of the dataset. The dataset may encode categories differently and use different feature names. Before treating the prediction endpoint as functional, update the form and `app.js` payload to match the dataset's exact feature schema. The endpoint intentionally shows a no-prediction state when no trained model is present.

## Current 3D limitation
The heart is a lightweight procedural illustration built from Three.js meshes so the starter project works without a large external anatomy asset. It is not a clinical-grade mesh or patient-specific reconstruction. For the final submission, replace it with a properly licensed anatomical `.glb`/`.gltf` asset, then map LAD/LCX/RCA highlights to the actual mesh regions.

## Clinical safety
For educational / decision-support research only. Not a medical device and not a substitute for professional assessment, ECG, angiography, or formal diagnostic imaging. Never use this prototype to make real clinical decisions.

## Before submission
- Verify target mapping and feature schema.
- Use leakage-safe validation and report all metrics transparently.
- Add SHAP or another explanation method and show feature contributions.
- Replace the stylized mesh with a licensed anatomical model.
- Add dataset and 3D asset attribution.
- Record the required demo video and write the project report.
