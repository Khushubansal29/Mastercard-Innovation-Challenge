import pandas as pd
import numpy as np

# LOAD DATASET

df = pd.read_csv("data/fraud_dataset_v3.csv")


print("\n" + "=" * 50)
print("DATASET AUDIT")
print("=" * 50)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# TARGET DISTRIBUTION

print("\n" + "=" * 50)
print("TARGET DISTRIBUTION")
print("=" * 50)

print(df["is_fraud"].value_counts())

print("\nTarget percentages:")
print(
    (df["is_fraud"].value_counts(normalize=True) * 100)
    .round(2)
)


# ATTACK DISTRIBUTION

print("\n" + "=" * 50)
print("ATTACK DISTRIBUTION")
print("=" * 50)

if "attack_type" in df.columns:
    print(df["attack_type"].value_counts())


# DUPLICATE CHECK

print("\n" + "=" * 50)
print("DUPLICATE CHECK")
print("=" * 50)

duplicate_rows = df.duplicated().sum()

print(f"Exact duplicate rows: {duplicate_rows}")

if duplicate_rows == 0:
    print("No exact duplicate rows found")
else:
    print("Duplicate rows found")


# ID / TARGET COLUMNS

print("\n" + "=" * 50)
print("IMPORTANT COLUMNS")
print("=" * 50)

for column in ["customer_id", "attack_type", "is_fraud"]:
    if column in df.columns:
        print(
            f"{column}: "
            f"unique={df[column].nunique()}, "
            f"dtype={df[column].dtype}"
        )


# NUMERIC FEATURES

numeric_features = df.select_dtypes(
    include=np.number
).columns.tolist()

if "is_fraud" in numeric_features:
    numeric_features.remove("is_fraud")


print("\n" + "=" * 50)
print("FRAUD vs LEGITIMATE FEATURE MEANS")
print("=" * 50)

comparison = df.groupby("is_fraud")[numeric_features].mean().T

comparison.columns = [
    "Legitimate",
    "Fraud"
]

comparison["Difference %"] = (
    (
        comparison["Fraud"] -
        comparison["Legitimate"]
    )
    /
    comparison["Legitimate"].replace(0, np.nan)
    * 100
)

print(
    comparison.sort_values(
        "Difference %",
        key=lambda x: x.abs(),
        ascending=False
    ).round(3)
)


# CORRELATION WITH TARGET

print("\n" + "=" * 50)
print("FEATURE CORRELATION WITH FRAUD")
print("=" * 50)

correlations = (
    df[numeric_features + ["is_fraud"]]
    .corr()["is_fraud"]
    .drop("is_fraud")
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print(correlations.round(4))


# SUSPICIOUS FEATURES

print("\n" + "=" * 50)
print("POTENTIAL LEAKAGE WARNING")
print("=" * 50)

for column in df.columns:

    if column in ["is_fraud", "attack_type"]:
        continue

    if df[column].nunique() <= 2:

        fraud_rate = (
            df.groupby(column)["is_fraud"]
            .mean()
        )

        print(f"\n{column}")
        print(fraud_rate)


# MISSING VALUES

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()

print(
    missing[missing > 0]
    if missing.sum() > 0
    else "No missing values"
)


print("\n" + "=" * 50)
print("AUDIT COMPLETE")
print("=" * 50)