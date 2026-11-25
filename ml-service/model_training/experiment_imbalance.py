
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE
)
from imblearn.combine import SMOTETomek


# ==================================================
# 1. CONFIGURATION
# ==================================================

TRAIN_PATH = Path(
    "data/processed/diabetes/diabetes_train.csv"
)

OUTPUT_DIR = Path(
    "reports/diabetes/imbalance"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_012"

RANDOM_STATE = 42
N_SPLITS = 5

# Increase prediabetes to 20% of majority class.
PREDIABETES_RATIO = 0.20


# ==================================================
# 2. LOAD TRAINING DATA ONLY
# ==================================================

print("\nLoading training dataset...")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# Create groups from identical feature combinations.
groups = pd.MultiIndex.from_frame(X).factorize()[0]

print("Dataset shape:", X.shape)
print("Number of feature groups:", len(np.unique(groups)))

print("\nOriginal class distribution:")
print(y.value_counts().sort_index())


# ==================================================
# 3. GROUP-AWARE CROSS VALIDATION
# ==================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []

EXPERIMENTS = [
    "Class Weighted Baseline",
    "Random Oversampling",
    "SMOTE",
    "SMOTE Tomek"
]


# ==================================================
# 4. CREATE SAMPLER
# ==================================================

def create_sampler(experiment_name, sampling_strategy):

    if experiment_name == "Random Oversampling":

        return RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        )

    elif experiment_name == "SMOTE":

        return SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=5,
            random_state=RANDOM_STATE
        )

    elif experiment_name == "SMOTE Tomek":

        return SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        )

    return None


# ==================================================
# 5. RUN EXPERIMENTS
# ==================================================

print("\nStarting imbalance experiments...")

for experiment_name in EXPERIMENTS:

    print("\n" + "=" * 70)
    print("EXPERIMENT:", experiment_name)
    print("=" * 70)

    fold_number = 1

    for train_idx, val_idx in cv.split(X, y, groups):

        print(f"\nPreparing fold {fold_number}...")

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # ------------------------------------------
        # Verify no group leakage
        # ------------------------------------------

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])

        overlap = train_groups.intersection(val_groups)

        assert len(overlap) == 0, (
            "Data leakage detected between training "
            "and validation groups!"
        )

        # ------------------------------------------
        # Calculate sampling target
        # ------------------------------------------

        majority_count = int((y_train == 0).sum())

        target_prediabetes = int(
            majority_count * PREDIABETES_RATIO
        )

        current_prediabetes = int(
            (y_train == 1).sum()
        )

        sampling_strategy = {
            1: max(target_prediabetes, current_prediabetes)
        }

        print(
            "Training distribution before sampling:",
            y_train.value_counts().sort_index().to_dict()
        )

        # ------------------------------------------
        # Build preprocessing pipeline
        # ------------------------------------------

        steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]

        # ------------------------------------------
        # Add sampler where required
        # ------------------------------------------

        sampler = create_sampler(
            experiment_name,
            sampling_strategy
        )

        if sampler is not None:

            steps.append(
                ("sampler", sampler)
            )

        # ------------------------------------------
        # Logistic Regression configuration
        # ------------------------------------------

        if experiment_name == "Class Weighted Baseline":

            classifier = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )

        else:

            classifier = LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=RANDOM_STATE
            )

        steps.append(
            ("classifier", classifier)
        )

        model = Pipeline(steps)

        # ------------------------------------------
        # Train model
        # ------------------------------------------

        print("Training model...")

        model.fit(
            X_train,
            y_train
        )

        # ------------------------------------------
        # Validate model
        # ------------------------------------------

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

        precision = precision_score(
            y_val,
            predictions,
            labels=[1],
            average="macro",
            zero_division=0
        )

        recall = recall_score(
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

        # ------------------------------------------
        # Store results
        # ------------------------------------------

        results.append({

            "Experiment": experiment_name,

            "Fold": fold_number,

            "Macro F1": macro_f1,

            "Weighted F1": weighted_f1,

            "Balanced Accuracy": balanced_acc,

            "Prediabetes Precision": precision,

            "Prediabetes Recall": recall,

            "Prediabetes F1": prediabetes_f1

        })

        print(
            f"Fold {fold_number} RESULTS:\n"
            f"Macro F1: {macro_f1:.4f}\n"
            f"Weighted F1: {weighted_f1:.4f}\n"
            f"Balanced Accuracy: {balanced_acc:.4f}\n"
            f"Prediabetes Precision: {precision:.4f}\n"
            f"Prediabetes Recall: {recall:.4f}\n"
            f"Prediabetes F1: {prediabetes_f1:.4f}"
        )

        fold_number += 1


# ==================================================
# 6. SAVE FOLD RESULTS
# ==================================================

results_df = pd.DataFrame(results)

fold_path = (
    OUTPUT_DIR / "imbalance_fold_results.csv"
)

results_df.to_csv(
    fold_path,
    index=False
)


# ==================================================
# 7. CALCULATE MEAN AND STANDARD DEVIATION
# ==================================================

metrics = [
    "Macro F1",
    "Weighted F1",
    "Balanced Accuracy",
    "Prediabetes Precision",
    "Prediabetes Recall",
    "Prediabetes F1"
]

summary = results_df.groupby("Experiment")[metrics].agg(
    ["mean", "std"]
)

summary_path = (
    OUTPUT_DIR / "imbalance_summary.csv"
)

summary.to_csv(summary_path)


# ==================================================
# 8. DISPLAY FINAL RESULTS
# ==================================================

print("\n" + "=" * 80)
print("FINAL IMBALANCE EXPERIMENT SUMMARY")
print("=" * 80)

for experiment_name, group in results_df.groupby("Experiment"):

    print("\n" + experiment_name)

    for metric in metrics:

        mean_value = group[metric].mean()

        std_value = group[metric].std()

        print(
            f"{metric}: "
            f"{mean_value:.4f} ± {std_value:.4f}"
        )


# ==================================================
# 9. COMPLETION
# ==================================================

print("\n" + "=" * 70)
print("IMBALANCE EXPERIMENTS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFold results saved:", fold_path)

print("Summary saved:", summary_path)
