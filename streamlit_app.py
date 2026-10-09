from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cardiovision_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.json"
METRICS_PATH = ROOT / "models" / "metrics.json"

HEART_URL = (
    "https://raw.githubusercontent.com/"
    "preyapatel2007/CardioVision3D/main/anatomy/heart.glb"
)

st.set_page_config(
    page_title="CardioVision3D",
    page_icon="❤️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 50% 0%, #10233b 0%, transparent 45%), #060d17;
    color: #eaf2ff;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #ecf4ff !important;
}
.hero {
    padding: 16px 0 26px;
}
.eyebrow {
    color: #78aaff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin-bottom: 10px;
}
.subtitle {
    color: #8ba0bb;
    font-size: 14px;
    line-height: 1.7;
    max-width: 800px;
}
div[data-testid="stMetric"] {
    background: #0b192a;
    border: 1px solid #223952;
    border-radius: 14px;
    padding: 16px;
}
.stButton button, .stFormSubmitButton button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}
.footnote {
    color: #8b9db4;
    font-size: 12px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FORM SECTIONS
# ============================================================

NUMERIC_COLUMNS_DEFAULT = {
    "Age", "Weight", "Length", "BMI", "DM", "HTN",
    "Current Smoker", "EX-Smoker", "FH", "BP", "PR",
    "Edema", "Typical Chest Pain", "Function Class",
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT",
    "EF-TTE", "Region RWMA"
}

GROUPS = {
    "Patient Demographics": [
        "Age", "Weight", "Length", "Sex", "BMI"
    ],
    "Medical History": [
        "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
        "Obesity", "CRF", "CVA", "Airway disease",
        "Thyroid Disease", "CHF", "DLP"
    ],
    "Vitals and Clinical Findings": [
        "BP", "PR", "Edema", "Weak Peripheral Pulse",
        "Lung rales", "Systolic Murmur", "Diastolic Murmur",
        "Typical Chest Pain", "Dyspnea", "Function Class",
        "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang"
    ],
    "ECG Findings": [
        "Q Wave", "St Elevation", "St Depression",
        "Tinversion", "LVH", "Poor R Progression", "BBB"
    ],
    "Laboratory Results": [
        "FBS", "CR", "TG", "LDL", "HDL", "BUN",
        "ESR", "HB", "K", "Na", "WBC", "Lymph",
        "Neut", "PLT"
    ],
    "Echocardiography": [
        "EF-TTE", "Region RWMA", "VHD"
    ],
}


# ============================================================
# LOAD THE EXISTING TRAINED MODEL
# ============================================================

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with FEATURES_PATH.open("r", encoding="utf-8") as file:
        features = json.load(file)

    if not isinstance(features, list):
        raise ValueError(
            "feature_columns.json must contain a list of feature names."
        )

    if not all(isinstance(item, str) for item in features):
        raise ValueError("All feature names must be strings.")

    if len(features) != 55:
        raise ValueError(
            f"Expected 55 input features, found {len(features)}."
        )

    metrics = {}

    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open(
                "r", encoding="utf-8"
            ) as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    return model, features, metrics


def as_estimator_map(model):
    if not isinstance(model, dict):
        raise TypeError(
            "The saved model must contain a dictionary of trained "
            "CAD, LAD, LCX and RCA estimators."
        )

    result = {
        str(key).strip().upper(): value
        for key, value in model.items()
    }

    if "CAD" not in result and "CATH" in result:
        result["CAD"] = result["CATH"]

    missing = [
        key for key in ("CAD", "LAD", "LCX", "RCA")
        if key not in result
    ]

    if missing:
        raise ValueError(
            "Missing model estimator(s): " + ", ".join(missing)
        )

    return result


def find_column_transformer(estimator):
    for step in getattr(estimator, "named_steps", {}).values():
        if hasattr(step, "transformers_"):
            return step

        for nested in getattr(step, "named_steps", {}).values():
            if hasattr(nested, "transformers_"):
                return nested

    return None


def find_encoder(obj):
    if hasattr(obj, "categories_"):
        return obj

    for step in getattr(obj, "named_steps", {}).values():
        encoder = find_encoder(step)
        if encoder is not None:
            return encoder

    return None


def get_types_and_options(model, features):
    estimators = as_estimator_map(model)
    numeric = set()
    categorical = set()
    options = {}

    for estimator in estimators.values():
        transformer = find_column_transformer(estimator)

        if transformer is None:
            continue

        for name, fitted_transformer, columns in transformer.transformers_:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(
                columns, (list, tuple, np.ndarray, pd.Index)
            ):
                continue

            columns = [str(column) for column in columns]
            label = str(name).lower()

            if label in {"num", "numeric", "numerical"}:
                numeric.update(columns)

            elif label in {"cat", "categorical"}:
                categorical.update(columns)
                encoder = find_encoder(fitted_transformer)

                if encoder is not None:
                    for column, values in zip(
                        columns, encoder.categories_
                    ):
                        cleaned = []

                        for value in values:
                            if isinstance(value, np.generic):
                                value = value.item()

                            try:
                                if pd.isna(value):
                                    continue
                            except (TypeError, ValueError):
                                pass

                            if value not in cleaned:
                                cleaned.append(value)

                        options[column] = cleaned

        if numeric or categorical:
            break

    if not numeric and not categorical:
        numeric = NUMERIC_COLUMNS_DEFAULT.intersection(features)
        categorical = set(features) - numeric

    unknown = set(features) - numeric - categorical

    if unknown:
        raise ValueError(
            "Could not determine feature types: "
            + ", ".join(sorted(unknown))
        )

    return numeric, categorical, options


# ============================================================
# PROBABILITY HELPERS
# ============================================================

def positive_probability(estimator, row, target):
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"The {target} model has no predict_proba method."
        )

    probabilities = np.asarray(estimator.predict_proba(row))[0]
    classes = getattr(estimator, "classes_", None)

    if classes is None or len(classes) != len(probabilities):
        raise ValueError(
            f"Could not read class labels for {target}."
        )

    labels = [str(value).strip().lower() for value in classes]

    if target.upper() == "CAD":
        positives = {
            "cad", "1", "yes", "true", "positive", "disease"
        }
    else:
        positives = {
            "stenotic", "stenosis", "1", "yes",
            "true", "positive", "disease"
        }

    index = next(
        (i for i, label in enumerate(labels) if label in positives),
        None,
    )

    if index is None:
        negatives = {
            "normal", "0", "no", "false", "negative",
            "non-cad", "nonstenotic"
        }

        negative_index = next(
            (i for i, label in enumerate(labels) if label in negatives),
            None,
        )

        if negative_index is not None and len(labels) == 2:
            index = 1 - negative_index
        elif len(labels) == 2:
            index = 1
        else:
            raise ValueError(
                f"Could not identify positive class for {target}: {classes}"
            )

    probability = float(probabilities[index])

    if not math.isfinite(probability):
        raise ValueError(f"Invalid probability for {target}.")

    return max(0.0, min(1.0, probability))


def display_label(value):
    if value is None:
        return "Not provided"

    labels = {
        "Y": "Yes",
        "N": "No",
        "Fmale": "Female",
    }

    return labels.get(str(value), str(value))


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">CARDIOVASCULAR INTELLIGENCE</div>
    <h1 style="font-size:38px;margin-bottom:10px;">
        ❤️ CardioVision3D
    </h1>
    <div class="subtitle">
        AI-assisted cardiac risk prediction with an interactive
        3D heart model. Enter the clinical information you have;
        unavailable measurements can remain blank.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:
    model, feature_columns, saved_metrics = load_artifacts()

    numeric_cols, categorical_cols, category_options = (
        get_types_and_options(model, feature_columns)
    )

    estimator_map = as_estimator_map(model)

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Check that models/cardiovision_model.joblib and "
        "models/feature_columns.json exist and that the saved model "
        "contains fitted CAD, LAD, LCX and RCA estimators."
    )
    st.stop()


# ============================================================
# PATIENT FORM
# ============================================================

st.markdown("## Patient Clinical Profile")

st.caption(
    "All 55 trained features are retained. Fill only the information "
    "you have. Blank values are treated as missing values, not zero or No."
)

inputs = {}

with st.form("cardiovision_patient_form", clear_on_submit=False):
    shown = set()

    for group, group_features in GROUPS.items():
        fields = [
            name for name in group_features
            if name in feature_columns
        ]

        if not fields:
            continue

        with st.expander(
            group,
            expanded=(group == "Patient Demographics")
        ):
            columns = st.columns(3)

            for index, name in enumerate(fields):
                shown.add(name)

                with columns[index % 3]:
                    if name in numeric_cols:
                        inputs[name] = st.text_input(
                            name,
                            value="",
                            placeholder="Optional — leave blank",
                            key=f"field_{name}",
                        )
                    else:
                        values = category_options.get(name, [])

                        inputs[name] = st.selectbox(
                            name,
                            options=[None] + values,
                            index=0,
                            format_func=display_label,
                            key=f"field_{name}",
                        )

    remaining = [
        name for name in feature_columns if name not in shown
    ]

    if remaining:
        with st.expander("Other Model Features"):
            columns = st.columns(3)

            for index, name in enumerate(remaining):
                with columns[index % 3]:
                    if name in numeric_cols:
                        inputs[name] = st.text_input(
                            name,
                            value="",
                            placeholder="Optional — leave blank",
                            key=f"field_{name}",
                        )
                    else:
                        values = category_options.get(name, [])

                        inputs[name] = st.selectbox(
                            name,
                            options=[None] + values,
                            index=0,
                            format_func=display_label,
                            key=f"field_{name}",
                        )

    st.info(
        f"{len(feature_columns)} features available. "
        "You do not need to fill every field."
    )

    submitted = st.form_submit_button(
        "✦ Analyze Cardiac Risk",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# RUN ML PREDICTION
# ============================================================

if submitted:
    try:
        row = {}

        for name in feature_columns:
            value = inputs.get(name)

            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                row[name] = np.nan

            elif name in numeric_cols:
                try:
                    number = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name} must be a number or left blank."
                    ) from error

                if not math.isfinite(number):
                    raise ValueError(
                        f"{name} must be a valid number."
                    )

                row[name] = number

            else:
                row[name] = value

        if all(pd.isna(value) for value in row.values()):
            raise ValueError(
                "Enter at least one available clinical value."
            )

        input_frame = pd.DataFrame(
            [row],
            columns=feature_columns,
        )

        results = {
            target: positive_probability(
                estimator_map[target],
                input_frame,
                target,
            )
            for target in ("CAD", "LAD", "LCX", "RCA")
        }

        st.session_state["cardiovision_results"] = results

    except Exception as error:
        st.session_state.pop("cardiovision_results", None)
        st.error(f"Prediction failed: {error}")


# ============================================================
# DISPLAY RESULTS
# ============================================================

st.divider()
st.markdown("## Cardiac Risk Results")

results = st.session_state.get("cardiovision_results")

if results:
    st.success(
        "Prediction completed. These values are model estimates, "
        "not a medical diagnosis."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Overall CAD", f"{results['CAD']:.1%}")
    col2.metric("LAD stenosis", f"{results['LAD']:.1%}")
    col3.metric("LCX stenosis", f"{results['LCX']:.1%}")
    col4.metric("RCA stenosis", f"{results['RCA']:.1%}")

    st.markdown("### Probability overview")

    for key, label in [
        ("CAD", "Overall CAD"),
        ("LAD", "LAD — Left Anterior Descending"),
        ("LCX", "LCX — Left Circumflex"),
        ("RCA", "RCA — Right Coronary Artery"),
    ]:
        st.write(f"**{label}:** {results[key]:.1%}")
        st.progress(results[key])

else:
    st.info(
        "Your prediction results will appear here after you submit the form."
    )


# ============================================================
# 3D HEART VIEWER
# ============================================================

st.markdown("## Interactive 3D Heart")

st.caption(
    "Drag to rotate and scroll to zoom. The 3D model is an anatomical "
    "visualization; predictions do not identify an exact lesion on this mesh."
)

heart_html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">

<style>
html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: radial-gradient(ellipse at center, #102943, #07111e 75%);
    font-family: Arial, sans-serif;
}
#viewer {
    position: absolute;
    inset: 0;
}
#loading {
    position: absolute;
    inset: 0;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #a9bfd8;
    font-size: 13px;
    text-align: center;
    padding: 20px;
    box-sizing: border-box;
}
</style>

<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"
    }
}
</script>
</head>

<body>
<div id="viewer"></div>
<div id="loading">Loading 3D heart model...</div>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const MODEL_URL = "__HEART_URL__";
const viewer = document.getElementById("viewer");
const loading = document.getElementById("loading");

try {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07111e);

    const camera = new THREE.PerspectiveCamera(
        42,
        Math.max(viewer.clientWidth, 1) /
            Math.max(viewer.clientHeight, 1),
        0.01,
        1000
    );

    camera.position.set(0, 0, 3);

    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false
    });

    renderer.setPixelRatio(
        Math.min(window.devicePixelRatio || 1, 1.5)
    );

    renderer.setSize(viewer.clientWidth, viewer.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    viewer.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 2.0));

    const keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
    keyLight.position.set(3, 4, 5);
    scene.add(keyLight);

    const blueLight = new THREE.DirectionalLight(0x6699ff, 1.1);
    blueLight.position.set(-4, 1, 3);
    scene.add(blueLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.55;
    controls.enablePan = false;
    controls.minDistance = 0.7;
    controls.maxDistance = 8;

    const loader = new GLTFLoader();
    loader.setCrossOrigin("anonymous");

    loader.load(
        MODEL_URL,

        function (gltf) {
            const heart = gltf.scene;
            scene.add(heart);

            const box = new THREE.Box3().setFromObject(heart);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            heart.position.sub(center);

            const maxSize = Math.max(size.x, size.y, size.z);

            if (maxSize > 0) {
                heart.scale.setScalar(2.1 / maxSize);
            }

            camera.position.set(0, 0, 3);
            controls.target.set(0, 0, 0);
            controls.update();

            loading.style.display = "none";
        },

        undefined,

        function (error) {
            console.error("Could not load heart.glb:", error);

            loading.textContent =
                "Could not load the 3D heart model. "
                + "Check browser console and network access.";
        }
    );

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    animate();

    window.addEventListener("resize", function () {
        const width = Math.max(viewer.clientWidth, 1);
        const height = Math.max(viewer.clientHeight, 1);

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });

} catch (error) {
    console.error("3D viewer initialization failed:", error);

    loading.textContent =
        "The 3D viewer could not start. Check browser WebGL support.";
}
</script>
</body>
</html>'''

components.html(
    heart_html.replace("__HEART_URL__", HEART_URL),
    height=560,
    scrolling=False,
)


# ============================================================
# FOOTER AND MODEL DETAILS
# ============================================================

st.divider()

st.markdown("""
<div class="footnote">
    <b>Research use only.</b> CardioVision3D provides machine-learning
    estimates based on available clinical features. It is not a medical
    diagnosis and does not replace professional medical advice. Predictions
    may be less reliable when many measurements are missing.
</div>
""", unsafe_allow_html=True)

with st.expander("Model and dataset files"):
    st.write(f"Input features: {len(feature_columns)}")
    st.write(f"Model: `{MODEL_PATH.name}`")
    st.write(f"Feature schema: `{FEATURES_PATH.name}`")

    if saved_metrics:
        st.write("Saved evaluation metrics:")
        st.json(saved_metrics)