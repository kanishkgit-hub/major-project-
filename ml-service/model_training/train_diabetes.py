
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = Path("data/processed/diabetes/diabetes_train.csv")
TEST_PATH = Path("data/processed/diabetes/diabetes_test.csv")

OUTPUT_DIR = Path("reports/diabetes/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("artifacts/diabetes")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_012"

CLASS_NAMES = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes"
}

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load training and testing datasets
# --------------------------------------------------

print("Loading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].astype(int)

FEATURES = list(X_train.columns)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

# Binary and ordinal features are retained as numeric values.
# Scaling is applied for Logistic Regression.
# Tree-based models do not require scaling.

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, FEATURES)
])

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ]),

    "LightGBM": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
}

# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------

results = []
trained_models = {}

print("\nStarting model training...\n")

for name, model in models.items():

    print("=" * 60)
    print("Training:", name)

    if name == "XGBoost":
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )
        model.fit(X_train, y_train, classifier__sample_weight=weights)

    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions,
        average="weighted",
        zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            CLASS_NAMES[0],
            CLASS_NAMES[1],
            CLASS_NAMES[2]
        ],
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            CLASS_NAMES[0],
            CLASS_NAMES[1],
            CLASS_NAMES[2]
        ],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Balanced Accuracy": balanced_acc,
        "Prediabetes Precision": report["Prediabetes"]["precision"],
        "Prediabetes Recall": report["Prediabetes"]["recall"],
        "Prediabetes F1": report["Prediabetes"]["f1-score"]
    })

    trained_models[name] = model

# --------------------------------------------------
# 6. Save model comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. Save all trained models
# --------------------------------------------------

for name, model in trained_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

# --------------------------------------------------
# 8. Save feature and class metadata
# --------------------------------------------------

metadata = {
    "target": TARGET,
    "classes": CLASS_NAMES,
    "features": FEATURES,
    "number_of_features": len(FEATURES),
    "models": list(trained_models.keys()),
    "selection_metric": "Macro F1"
}

with open(MODEL_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)

print("\nTraining completed successfully!")
print("Comparison saved inside:", OUTPUT_DIR)
print("Models saved inside:", MODEL_DIR)
