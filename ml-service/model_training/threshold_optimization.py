
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

TRAIN_PATH = Path(
    "data/processed/diabetes/diabetes_train.csv"
)

OUTPUT_DIR = Path(
    "reports/diabetes/threshold"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_012"

RANDOM_STATE = 42
N_SPLITS = 5

# Thresholds to investigate.
THRESHOLDS = np.round(
    np.arange(0.05, 0.51, 0.01),
    2
)


# ==================================================
# 2. LOAD TRAINING DATA ONLY
# ==================================================

print("\nLoading training dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

groups = pd.MultiIndex.from_frame(X).factorize()[0]

print("Dataset shape:", X.shape)
print("Unique feature groups:", len(np.unique(groups)))

print("\nClass distribution:")
print(y.value_counts().sort_index())


# ==================================================
# 3. CREATE MODEL
# ==================================================

def create_model():

    return Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )

    ])


# ==================================================
# 4. GENERATE OUT-OF-FOLD PROBABILITIES
# ==================================================

print("\nStarting 5-fold group-aware cross-validation...")

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Store probabilities in original row order.
oof_probabilities = np.zeros(
    (len(X), 3)
)

oof_predictions = np.zeros(
    len(X),
    dtype=int
)

fold_number = 1

for train_idx, val_idx in cv.split(X, y, groups):

    print(f"\nTraining fold {fold_number}...")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]

    # Check for group leakage.
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    assert len(train_groups.intersection(val_groups)) == 0, (
        "Group leakage detected!"
    )

    model = create_model()

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)

    # Ensure probability columns correspond to classes 0, 1, 2.
    for class_index, class_label in enumerate(model.classes_):

        oof_probabilities[
            val_idx, int(class_label)
        ] = probabilities[:, class_index]

    predictions = model.predict(X_val)

    oof_predictions[val_idx] = predictions

    print(
        f"Fold {fold_number} completed | "
        f"Validation records: {len(val_idx)}"
    )

    fold_number += 1


# ==================================================
# 5. SAVE OOF PROBABILITIES
# ==================================================

oof_df = pd.DataFrame({

    "Actual": y.to_numpy(),

    "Probability_No_Diabetes": oof_probabilities[:, 0],

    "Probability_Prediabetes": oof_probabilities[:, 1],

    "Probability_Diabetes": oof_probabilities[:, 2],

    "Default_Prediction": oof_predictions

})

oof_df.to_csv(
    OUTPUT_DIR / "oof_probabilities.csv",
    index=False
)

print("\nOut-of-fold probabilities generated successfully.")


# ==================================================
# 6. BASELINE PERFORMANCE
# ==================================================

baseline_macro_f1 = f1_score(
    y,
    oof_predictions,
    average="macro",
    zero_division=0
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y,
    oof_predictions
)

baseline_precision = precision_score(
    y,
    oof_predictions,
    labels=[1],
    average="macro",
    zero_division=0
)

baseline_recall = recall_score(
    y,
    oof_predictions,
    labels=[1],
    average="macro",
    zero_division=0
)

baseline_prediabetes_f1 = f1_score(
    y,
    oof_predictions,
    labels=[1],
    average="macro",
    zero_division=0
)

print("\n" + "=" * 75)
print("DEFAULT ARGMAX BASELINE")
print("=" * 75)

print(f"Macro F1: {baseline_macro_f1:.4f}")
print(f"Balanced Accuracy: {baseline_balanced_accuracy:.4f}")
print(f"Prediabetes Precision: {baseline_precision:.4f}")
print(f"Prediabetes Recall: {baseline_recall:.4f}")
print(f"Prediabetes F1: {baseline_prediabetes_f1:.4f}")


# ==================================================
# 7. THRESHOLD OPTIMIZATION
# ==================================================

results = []

print("\nTesting prediabetes thresholds...")

p_no_diabetes = oof_probabilities[:, 0]
p_prediabetes = oof_probabilities[:, 1]
p_diabetes = oof_probabilities[:, 2]

for threshold in THRESHOLDS:

    # Predict prediabetes when its probability
    # crosses the selected threshold.
    predictions = np.where(
        p_prediabetes >= threshold,
        1,
        np.where(
            p_no_diabetes >= p_diabetes,
            0,
            2
        )
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        labels=[1],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        labels=[1],
        average="macro",
        zero_division=0
    )

    prediabetes_f1 = f1_score(
        y,
        predictions,
        labels=[1],
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Macro F1": macro_f1,

        "Weighted F1": weighted_f1,

        "Balanced Accuracy": balanced_acc,

        "Prediabetes Precision": precision,

        "Prediabetes Recall": recall,

        "Prediabetes F1": prediabetes_f1

    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Prediabetes Precision: {precision:.4f} | "
        f"Prediabetes Recall: {recall:.4f} | "
        f"Prediabetes F1: {prediabetes_f1:.4f}"
    )


# ==================================================
# 8. SAVE THRESHOLD RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)


# ==================================================
# 9. FIND BEST THRESHOLDS
# ==================================================

best_macro = results_df.loc[
    results_df["Macro F1"].idxmax()
]

best_prediabetes_f1 = results_df.loc[
    results_df["Prediabetes F1"].idxmax()
]

best_balanced = results_df.loc[
    results_df["Balanced Accuracy"].idxmax()
]

print("\n" + "=" * 75)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 75)

print("\nBEST THRESHOLD FOR MACRO F1")

print(best_macro.to_string())

print("\nBEST THRESHOLD FOR PREDIABETES F1")

print(best_prediabetes_f1.to_string())

print("\nBEST THRESHOLD FOR BALANCED ACCURACY")

print(best_balanced.to_string())


# ==================================================
# 10. BEST MACRO F1 CLASSIFICATION REPORT
# ==================================================

selected_threshold = float(best_macro["Threshold"])

final_predictions = np.where(
    p_prediabetes >= selected_threshold,
    1,
    np.where(
        p_no_diabetes >= p_diabetes,
        0,
        2
    )
)

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT AT BEST MACRO F1 THRESHOLD")
print("=" * 75)

print(
    classification_report(
        y,
        final_predictions,
        labels=[0, 1, 2],
        target_names=[
            "No Diabetes",
            "Prediabetes",
            "Diabetes"
        ],
        zero_division=0
    )
)


# ==================================================
# 11. COMPLETION
# ==================================================

print("\n" + "=" * 75)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 75)

print("Best Macro F1 threshold:", selected_threshold)

print(
    "Threshold results saved:",
    OUTPUT_DIR / "threshold_comparison.csv"
)

print(
    "OOF probabilities saved:",
    OUTPUT_DIR / "oof_probabilities.csv"
)
