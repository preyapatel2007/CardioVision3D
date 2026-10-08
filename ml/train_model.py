from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "extention+of+z+alizadeh+sani+dataset"
    / "extention of Z-Alizadeh sani dataset.xlsx"
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "cardiovision_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURES_PATH = MODELS_DIR / "feature_columns.json"


# ============================================================
# TARGET COLUMNS
# ============================================================

# Dataset uses "Cath" for the overall CAD classification.
SOURCE_TARGET_COLUMNS = [
    "LAD",
    "LCX",
    "RCA",
    "Cath"
]


# ============================================================
# TARGET CONVERSION
# ============================================================

def convert_target(value, target_name):
    """
    Convert dataset labels into binary values.

    CAD target:
        CAD    -> 1
        Normal -> 0

    Vessel targets:
        Stenotic -> 1
        Normal   -> 0
    """

    value = str(value).strip().lower()

    if target_name == "Cath":

        if value == "cad":
            return 1

        if value == "normal":
            return 0

    else:

        if value == "stenotic":
            return 1

        if value == "normal":
            return 0

    raise ValueError(
        f"Unexpected value '{value}' found in target '{target_name}'."
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred, y_probability):

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        )
    }

    # ROC-AUC requires both classes to be present
    try:

        if len(set(y_true)) == 2:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_probability
                )
            )

        else:

            metrics["roc_auc"] = None

    except Exception:

        metrics["roc_auc"] = None

    return metrics


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 60)
print("          CARDIOVISION3D MODEL TRAINING")
print("=" * 60)
print()

print("Dataset:")
print(DATASET_PATH)
print()

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


df = pd.read_excel(
    DATASET_PATH,
    sheet_name="Sheet 1 - Table 1"
)

print(
    f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns"
)

print()


# ============================================================
# VERIFY TARGET COLUMNS
# ============================================================

missing_targets = [
    column
    for column in SOURCE_TARGET_COLUMNS
    if column not in df.columns
]

if missing_targets:

    raise ValueError(
        "Missing required target columns: "
        + ", ".join(missing_targets)
    )


print("Target columns found:")
print(SOURCE_TARGET_COLUMNS)
print()


# ============================================================
# CONVERT TARGETS
# ============================================================

target_data = pd.DataFrame(index=df.index)

# Cath -> CAD
target_data["CAD"] = df["Cath"].apply(
    lambda value: convert_target(
        value,
        "Cath"
    )
)

# Vessel targets
target_data["LAD"] = df["LAD"].apply(
    lambda value: convert_target(
        value,
        "LAD"
    )
)

target_data["LCX"] = df["LCX"].apply(
    lambda value: convert_target(
        value,
        "LCX"
    )
)

target_data["RCA"] = df["RCA"].apply(
    lambda value: convert_target(
        value,
        "RCA"
    )
)


# ============================================================
# INPUT FEATURES
# ============================================================

# IMPORTANT:
# Remove all target/leakage columns.
features_to_remove = [
    "LAD",
    "LCX",
    "RCA",
    "Cath"
]

X = df.drop(
    columns=features_to_remove
)

Y = target_data[
    [
        "CAD",
        "LAD",
        "LCX",
        "RCA"
    ]
]


print(
    f"Input features: {X.shape[1]}"
)

print(
    "Removed leakage columns:",
    features_to_remove
)

print()


# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


print(
    f"Numeric features: {len(numeric_features)}"
)

print(
    f"Categorical features: {len(categorical_features)}"
)

print()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y["CAD"]
)


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows:  {len(X_test)}"
)

print()


# ============================================================
# TRAIN FOUR MODELS
# ============================================================

targets = [
    "CAD",
    "LAD",
    "LCX",
    "RCA"
]

models = {}
metrics = {}


for target in targets:

    print("-" * 60)
    print(f"Training model: {target}")
    print("-" * 60)

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        Y_train[target]
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    model_metrics = calculate_metrics(
        Y_test[target],
        predictions,
        probabilities
    )

    models[target] = pipeline
    metrics[target] = model_metrics

    print(
        f"Accuracy : {model_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {model_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {model_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {model_metrics['f1_score']:.4f}"
    )

    if model_metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC  : {model_metrics['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC  : unavailable"
        )

    print()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    models,
    MODEL_PATH
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=2
    )


# ============================================================
# SAVE FEATURE COLUMNS
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        X.columns.tolist(),
        file,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 60)
print("                TRAINING COMPLETE")
print("=" * 60)
print()

print(
    "Model saved:"
)
print(MODEL_PATH)
print()

print(
    "Metrics saved:"
)
print(METRICS_PATH)
print()

print(
    "Feature columns saved:"
)
print(FEATURES_PATH)
print()

print("Targets trained:")
print("CAD")
print("LAD")
print("LCX")
print("RCA")
print()

print("=" * 60)