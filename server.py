from flask import Flask, request, jsonify, send_from_directory, make_response

from pathlib import Path
from datetime import datetime
import csv
import io
import json
import uuid

import joblib
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "cardiovision_model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

REPORTS_DIR = BASE_DIR / "reports"
REPORT_INDEX = REPORTS_DIR / "reports.json"

ANATOMY_DIR = BASE_DIR / "anatomy"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ANATOMY_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__, static_folder=None)

model = None


# =========================================================
# REPORT HELPERS
# =========================================================

def load_reports():
    if not REPORT_INDEX.exists():
        return []

    try:
        with open(REPORT_INDEX, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_reports(reports):
    with open(REPORT_INDEX, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)


def create_report(patient_data, predictions):
    report_id = (
        "CV3D-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )

    created_at = datetime.now().astimezone().isoformat()

    report = {
        "report_id": report_id,
        "created_at": created_at,
        "patient_data": patient_data,
        "predictions": predictions
    }

    reports = load_reports()
    reports.append(report)
    save_reports(reports)

    return report


def find_report(report_id):
    reports = load_reports()

    for report in reports:
        if str(report.get("report_id", "")) == str(report_id):
            return report

    return None


# =========================================================
# MODEL LOADING
# =========================================================

def load_model():
    global model

    if not MODEL_PATH.exists():
        model = None
        return None

    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model

    except Exception as exc:
        print("Model loading error:", repr(exc))
        model = None
        return None


def load_feature_columns():
    if not FEATURE_COLUMNS_PATH.exists():
        return []

    try:
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception as exc:
        print("Feature column loading error:", repr(exc))
        return []


def load_metrics():
    if not METRICS_PATH.exists():
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


# =========================================================
# MODEL ESTIMATORS
# =========================================================

def get_estimators(loaded_model):
    targets = ["CAD", "LAD", "LCX", "RCA"]

    if isinstance(loaded_model, dict):
        result = {}

        for target in targets:
            if target in loaded_model:
                result[target] = loaded_model[target]

        if result:
            return result

        for key, value in loaded_model.items():
            key_upper = str(key).upper()

            if key_upper in targets:
                result[key_upper] = value

        if result:
            return result

    return {"CAD": loaded_model}


# =========================================================
# PREPROCESSOR COLUMNS
# =========================================================

def get_preprocessor_columns(estimator):
    numeric_columns = []
    categorical_columns = []

    preprocessor = None

    if hasattr(estimator, "named_steps"):
        preprocessor = estimator.named_steps.get("preprocessor")

    if preprocessor is None and hasattr(estimator, "transformers_"):
        if estimator.__class__.__name__ == "ColumnTransformer":
            preprocessor = estimator

    if preprocessor is None:
        return numeric_columns, categorical_columns

    try:
        for transformer_name, transformer, columns in preprocessor.transformers_:

            if transformer_name == "remainder":
                continue

            if columns is None:
                continue

            columns = list(columns)
            name_lower = str(transformer_name).lower()

            if "num" in name_lower or "numeric" in name_lower:
                numeric_columns.extend(columns)

            elif "cat" in name_lower or "categor" in name_lower:
                categorical_columns.extend(columns)

    except Exception as exc:
        print("Preprocessor inspection error:", repr(exc))

    return numeric_columns, categorical_columns


# =========================================================
# PREPARE INPUT
# =========================================================

def prepare_input_row(loaded_model, payload, feature_columns):
    if not feature_columns:
        raise ValueError("Feature columns are missing.")

    row = {}

    for column in feature_columns:
        row[column] = payload.get(column, None)

    df = pd.DataFrame([row], columns=feature_columns)

    estimators = get_estimators(loaded_model)
    first_estimator = next(iter(estimators.values()), None)

    numeric_columns = []
    categorical_columns = []

    if first_estimator is not None:
        (
            numeric_columns,
            categorical_columns
        ) = get_preprocessor_columns(first_estimator)

    # Numeric fields
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Categorical fields
    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .fillna("")
                .astype(str)
            )

    # Fallback for anything not detected
    detected = set(
        numeric_columns + categorical_columns
    )

    for column in feature_columns:

        if column in detected:
            continue

        value = df.at[0, column]

        if value is None or value == "":
            continue

        numeric_value = pd.to_numeric(
            value,
            errors="coerce"
        )

        if pd.isna(numeric_value):
            df.at[0, column] = str(value)
        else:
            df.at[0, column] = numeric_value

    return df.reindex(columns=feature_columns)


# =========================================================
# POSITIVE CLASS PROBABILITY
# =========================================================

def get_positive_probability(estimator, row):

    if hasattr(estimator, "predict_proba"):

        probabilities = estimator.predict_proba(row)

        if len(probabilities.shape) == 2:

            classes = getattr(
                estimator,
                "classes_",
                None
            )

            if classes is not None and len(classes) > 0:

                class_list = list(classes)

                if 1 in class_list:
                    positive_index = class_list.index(1)
                else:
                    positive_index = len(class_list) - 1

            else:
                positive_index = probabilities.shape[1] - 1

            return float(
                probabilities[0, positive_index]
            )

    prediction = estimator.predict(row)

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    try:
        return float(prediction)

    except Exception:
        return 0.0


# =========================================================
# RUN PREDICTION
# =========================================================

def run_prediction(payload):

    global model

    if model is None:
        load_model()

    if model is None:
        raise RuntimeError(
            "Trained model is not available."
        )

    feature_columns = load_feature_columns()

    if not feature_columns:
        raise RuntimeError(
            "Feature columns are not available."
        )

    row = prepare_input_row(
        model,
        payload,
        feature_columns
    )

    estimators = get_estimators(model)

    predictions = {}

    for target in ["CAD", "LAD", "LCX", "RCA"]:

        if target not in estimators:
            continue

        probability = get_positive_probability(
            estimators[target],
            row
        )

        probability = max(
            0.0,
            min(1.0, probability)
        )

        predictions[target] = probability

    if not predictions:
        raise RuntimeError(
            "No prediction models were found."
        )

    return predictions


# =========================================================
# FRONTEND ROUTES
# =========================================================

@app.route("/")
def index():
    return send_from_directory(
        str(BASE_DIR),
        "index.html"
    )


@app.route("/app.js")
def app_js():
    return send_from_directory(
        str(BASE_DIR),
        "app.js"
    )


@app.route("/style.css")
def style_css():
    return send_from_directory(
        str(BASE_DIR),
        "style.css"
    )


# =========================================================
# HEART GLB ROUTE
# =========================================================

@app.route("/anatomy/<path:filename>")
def anatomy_files(filename):
    return send_from_directory(
        str(ANATOMY_DIR),
        filename
    )


# =========================================================
# STATUS
# =========================================================

@app.route("/api/status", methods=["GET"])
def api_status():

    feature_columns = load_feature_columns()
    metrics = load_metrics()

    return jsonify({
        "status": "ok",
        "model_available": model is not None,
        "model_path": str(MODEL_PATH),
        "feature_file_available": FEATURE_COLUMNS_PATH.exists(),
        "feature_count": len(feature_columns),
        "metrics_available": bool(metrics),
        "heart_available": (
            ANATOMY_DIR / "heart.glb"
        ).exists(),
        "reports_directory": str(REPORTS_DIR)
    })


@app.route("/api/health", methods=["GET"])
def api_health():

    return jsonify({
        "status": "ok",
        "service": "CardioVision3D"
    })


# =========================================================
# PREDICTION
# =========================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:
        payload = request.get_json(
            silent=True
        ) or {}

        if not isinstance(payload, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid JSON payload."
            }), 400

        predictions = run_prediction(payload)

        report = create_report(
            patient_data=payload,
            predictions=predictions
        )

        print(
            "Prediction successful:",
            predictions
        )

        return jsonify({
            "status": "success",
            "message": "Prediction generated successfully.",
            "model": "CardioVision3D trained model",
            "predictions": predictions,
            "report_id": report["report_id"],
            "created_at": report["created_at"]
        })

    except Exception as exc:

        print(
            "Prediction error:",
            repr(exc)
        )

        return jsonify({
            "status": "error",
            "message": str(exc),
            "predictions": None
        }), 500


@app.route("/predict", methods=["POST"])
def predict_legacy():
    return api_predict()


# =========================================================
# REPORT LIST
# =========================================================

@app.route("/api/reports", methods=["GET"])
def api_reports():

    reports = load_reports()

    reports.sort(
        key=lambda item: item.get(
            "created_at",
            ""
        ),
        reverse=True
    )

    return jsonify(reports)


# =========================================================
# SINGLE REPORT
# =========================================================

@app.route(
    "/api/reports/<report_id>",
    methods=["GET"]
)
def api_single_report(report_id):

    report = find_report(
        report_id
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Report not found."
        }), 404

    return jsonify(report)


# =========================================================
# CSV DOWNLOAD
# =========================================================

@app.route(
    "/api/reports/<report_id>/csv",
    methods=["GET"]
)
def download_csv(report_id):

    report = find_report(
        report_id
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Report not found."
        }), 404

    output = io.StringIO(
        newline=""
    )

    writer = csv.writer(output)

    writer.writerow([
        "CardioVision3D Patient Report"
    ])

    writer.writerow([])

    writer.writerow([
        "Report ID",
        report.get("report_id", "")
    ])

    writer.writerow([
        "Created At",
        report.get("created_at", "")
    ])

    writer.writerow([])

    writer.writerow([
        "PATIENT INPUT",
        "VALUE"
    ])

    patient_data = report.get(
        "patient_data",
        {}
    )

    for key, value in patient_data.items():
        writer.writerow([
            key,
            value
        ])

    writer.writerow([])

    writer.writerow([
        "PREDICTION",
        "PROBABILITY"
    ])

    predictions = report.get(
        "predictions",
        {}
    )

    for key, value in predictions.items():
        writer.writerow([
            key,
            f"{float(value) * 100:.2f}%"
        ])

    writer.writerow([])

    writer.writerow([
        "Disclaimer",
        "Educational/research output only. This is not a medical diagnosis."
    ])

    response = make_response(
        output.getvalue()
    )

    response.headers[
        "Content-Type"
    ] = "text/csv; charset=utf-8"

    response.headers[
        "Content-Disposition"
    ] = (
        f'attachment; '
        f'filename="{report_id}.csv"'
    )

    return response


# =========================================================
# PDF DOWNLOAD
# =========================================================

@app.route(
    "/api/reports/<report_id>/pdf",
    methods=["GET"]
)
def download_pdf(report_id):

    if not REPORTLAB_AVAILABLE:
        return jsonify({
            "status": "error",
            "message": "ReportLab is not installed."
        }), 500

    report = find_report(
        report_id
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Report not found."
        }), 404

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CardioTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "CardioSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.HexColor(
            "#5f6f82"
        ),
        spaceAfter=16
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "CardioNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )

    story = []

    story.append(
        Paragraph(
            "CardioVision 3D",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Cardiac Risk Prediction Report",
            subtitle_style
        )
    )

    info_data = [
        [
            "Report ID",
            report.get("report_id", "")
        ],
        [
            "Created At",
            report.get("created_at", "")
        ]
    ]

    info_table = Table(
        info_data,
        colWidths=[
            38 * mm,
            125 * mm
        ]
    )

    info_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#edf3fa")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd6e2")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        info_table
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Prediction Results",
            heading_style
        )
    )

    prediction_rows = [
        ["Output", "Probability"]
    ]

    predictions = report.get(
        "predictions",
        {}
    )

    for key, value in predictions.items():
        prediction_rows.append([
            key,
            f"{float(value) * 100:.2f}%"
        ])

    prediction_table = Table(
        prediction_rows,
        colWidths=[
            80 * mm,
            70 * mm
        ]
    )

    prediction_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172b43")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cbd6e2")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        prediction_table
    )

    story.append(
        Paragraph(
            "Patient Input Data",
            heading_style
        )
    )

    patient_rows = [
        ["Feature", "Value"]
    ]

    for key, value in report.get(
        "patient_data",
        {}
    ).items():

        patient_rows.append([
            str(key),
            str(value)
        ])

    patient_table = Table(
        patient_rows,
        colWidths=[
            85 * mm,
            65 * mm
        ],
        repeatRows=1
    )

    patient_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172b43")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#d2dbe5")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            )
        ])
    )

    story.append(
        patient_table
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "<b>Clinical Safety Disclaimer:</b> "
            "This report contains research-oriented and "
            "educational model output generated from the "
            "provided clinical feature values. It is not a "
            "medical diagnosis and should not replace formal "
            "clinical evaluation, diagnostic imaging, or "
            "professional medical advice.",
            normal_style
        )
    )

    document.build(story)

    response = make_response(
        buffer.getvalue()
    )

    response.headers[
        "Content-Type"
    ] = "application/pdf"

    response.headers[
        "Content-Disposition"
    ] = (
        f'attachment; '
        f'filename="{report_id}.pdf"'
    )

    return response


# =========================================================
# STARTUP INFORMATION
# =========================================================

def print_startup_status():

    feature_columns = load_feature_columns()
    reports = load_reports()

    print()
    print("==============================================")
    print("           CardioVision3D Backend")
    print("==============================================")
    print("Project:", BASE_DIR)
    print("Model exists:", MODEL_PATH.exists())
    print("Feature file exists:", FEATURE_COLUMNS_PATH.exists())
    print("Features:", len(feature_columns))
    print("Reports folder:", REPORTS_DIR)
    print("Saved reports:", len(reports))
    print(
        "Heart GLB exists:",
        (ANATOMY_DIR / "heart.glb").exists()
    )
    print(
        "ReportLab available:",
        REPORTLAB_AVAILABLE
    )
    print("==============================================")
    print()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    load_model()
    print_startup_status()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )