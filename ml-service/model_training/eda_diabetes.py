
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dataset
DATA_PATH = Path(
    "data/diabetes-health-indicators-dataset/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

# Output directory
OUTPUT_DIR = Path("reports/diabetes/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

target = "Diabetes_012"

class_names = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes"
}

sns.set_theme(style="whitegrid")

# 1. Target distribution
plt.figure(figsize=(8, 5))

counts = df[target].value_counts().sort_index()

sns.barplot(
    x=[class_names[int(i)] for i in counts.index],
    y=counts.values
)

plt.title("Diabetes Target Class Distribution")
plt.xlabel("Diabetes Class")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=300)
plt.close()


# 2. BMI distribution by class
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x=target,
    y="BMI",
    showfliers=False
)

plt.xticks([0, 1, 2], ["No Diabetes", "Prediabetes", "Diabetes"])
plt.title("BMI Distribution Across Diabetes Classes")
plt.xlabel("Diabetes Class")
plt.ylabel("BMI")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bmi_by_class.png", dpi=300)
plt.close()


# 3. Age distribution by class
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x=target,
    y="Age",
    showfliers=False
)

plt.xticks([0, 1, 2], ["No Diabetes", "Prediabetes", "Diabetes"])
plt.title("Age Distribution Across Diabetes Classes")
plt.xlabel("Diabetes Class")
plt.ylabel("Age Category")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_by_class.png", dpi=300)
plt.close()


# 4. Correlation heatmap
plt.figure(figsize=(16, 12))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7}
)

plt.title("Correlation Heatmap - Diabetes Dataset")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)
plt.close()


# 5. Class-wise feature summary
print("\n===== CLASS-WISE FEATURE SUMMARY =====")

print(
    df.groupby(target)[
        ["BMI", "Age", "GenHlth", "MentHlth", "PhysHlth"]
    ].mean().round(2)
)

print("\n===== EDA COMPLETED =====")
print("Charts saved inside:", OUTPUT_DIR)
