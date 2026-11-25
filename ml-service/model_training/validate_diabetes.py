
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = Path("data/processed/diabetes/diabetes_train.csv")

OUTPUT_DIR = Path("reports/diabetes/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_012"
RANDOM_STATE = 42
N_SPLITS = 5

CLASS_NAMES = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes"
}


# --------------------------------------------------
# 2. Load training dataset ONLY
# --------------------------------------------------

print("Loading training dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

print("Training dataset shape:", X.shape)

# Create groups using identical feature combinations.
groups = pd.MultiIndex.from_frame(X).factorize()[0]

print("Unique feature groups:", len(np.unique(groups)))


# --------------------------------------------------
# 3. Define model creation function
# --------------------------------------------------

def create_model(name):

    if name == "Logistic Regression":

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

    elif name == "Random Forest":

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

    elif name == "XGBoost":

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])

    elif name == "LightGBM":

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])


# --------------------------------------------------
# 4. Define cross-validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM"
]

results = []

print("\nStarting 5-fold group-aware cross-validation...")


# --------------------------------------------------
# 5. Train and validate each model
# --------------------------------------------------

for model_name in models:

    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    fold_number = 1

    for train_idx, val_idx in cv.split(X, y, groups):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Verify group separation.
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        assert len(train_groups.intersection(val_groups)) == 0

        model = create_model(model_name)

        # Apply balanced sample weights to XGBoost.
        if model_name == "XGBoost":

            weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            model.fit(
                X_train,
                y_train,
                classifier__sample_weight=weights
            )

        else:
            model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_val,
            predictions,
            average="weighted",
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(
            y_val,
            predictions
        )

        prediabetes_precision = precision_score(
            y_val,
            predictions,
            labels=[1],
            average="macro",
            zero_division=0
        )

        prediabetes_recall = recall_score(
            y_val,
            predictions,
            labels=[1],
            average="macro",
            zero_division=0
        )

        prediabetes_f1 = f1_score(
            y_val,
            predictions,
            labels=[1],
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Fold": fold_number,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Balanced Accuracy": balanced_acc,
            "Prediabetes Precision": prediabetes_precision,
            "Prediabetes Recall": prediabetes_recall,
            "Prediabetes F1": prediabetes_f1
        })

        print(
            f"Fold {fold_number}: "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Accuracy={balanced_acc:.4f}, "
            f"Prediabetes Recall={prediabetes_recall:.4f}"
        )

        fold_number += 1


# --------------------------------------------------
# 6. Save individual fold results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "fold_results.csv",
    index=False
)


# --------------------------------------------------
# 7. Calculate mean and standard deviation
# --------------------------------------------------

metrics = [
    "Macro F1",
    "Weighted F1",
    "Balanced Accuracy",
    "Prediabetes Precision",
    "Prediabetes Recall",
    "Prediabetes F1"
]

summary = results_df.groupby("Model")[metrics].agg(
    ["mean", "std"]
)

summary.to_csv(
    OUTPUT_DIR / "validation_summary.csv"
)

print("\n" + "=" * 75)
print("FINAL CROSS-VALIDATION SUMMARY")
print("=" * 75)

for model_name, group in results_df.groupby("Model"):

    print("\n", model_name)

    for metric in metrics:

        mean_value = group[metric].mean()
        std_value = group[metric].std()

        print(
            f"{metric}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )


# --------------------------------------------------
# 8. Final output
# --------------------------------------------------

print("\nCross-validation completed successfully!")

print("Fold results:", OUTPUT_DIR / "fold_results.csv")
print("Summary:", OUTPUT_DIR / "validation_summary.csv")
