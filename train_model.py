import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)


# -----------------------------------
# 1. Load our dataset
# -----------------------------------

df = pd.read_csv("data/fraud_dataset_v3.csv")

print("Dataset loaded!")
print("Total transactions:", len(df))

# -----------------------------------
# Derived velocity feature
# -----------------------------------

df["velocity_ratio"] = (
    df["transactions_last_1h"] /
    (df["transactions_last_24h"] / 24 + 0.1)
)

# -----------------------------------
# 2. Select features and target
# -----------------------------------

features = [
    "amount",
    "hour",
    "is_new_device",
    "is_new_location",
    "distance_from_home",
    "avg_transaction_amount",
    "transactions_last_1h",
    "transactions_last_24h",
    "velocity_ratio",
    "account_age_days",
    "identity_risk_score",
    "verification_failed",
    "social_engineering_risk",
    "behaviour_deviation_score"
]

X = df[features]
y = df["is_fraud"]


# -----------------------------------
# 3. Split into training and testing
# -----------------------------------

# -----------------------------------
# Customer-level train/test split
# -----------------------------------

# Get unique customers
unique_customers = df["customer_id"].unique()

# Shuffle customers
np.random.seed(42)
np.random.shuffle(unique_customers)

# 80% customers for training
split_index = int(len(unique_customers) * 0.80)

train_customers = unique_customers[:split_index]
test_customers = unique_customers[split_index:]

# Create masks
train_mask = df["customer_id"].isin(train_customers)
test_mask = df["customer_id"].isin(test_customers)

# Split the data
X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("\nCustomer-level split:")
print("Training customers:", len(train_customers))
print("Testing customers:", len(test_customers))

print("\nTraining transactions:", len(X_train))
print("Testing transactions:", len(X_test))

print("\nTraining transactions:", len(X_train))
print("Testing transactions:", len(X_test))


# -----------------------------------
# 4. Create the ML model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------
# 5. Train the model
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training complete!")


# -----------------------------------
# 6. Make predictions
# -----------------------------------

y_pred = model.predict(X_test)


# -----------------------------------
# 7. Evaluate the model
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n-------------------------------")
print("MODEL RESULTS")
print("-------------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# 8. Feature importance
# -----------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# -----------------------------------
# 9. Performance by attack type
# -----------------------------------

test_results = df.loc[X_test.index].copy()

test_results["prediction"] = y_pred

fraud_results = test_results[
    test_results["is_fraud"] == 1
]

print("\n===================================")
print("PERFORMANCE BY ATTACK TYPE")
print("===================================")

for attack in fraud_results["attack_type"].unique():

    attack_data = fraud_results[
        fraud_results["attack_type"] == attack
    ]

    detected = (
        attack_data["prediction"] == 1
    ).sum()

    total = len(attack_data)

    recall = detected / total if total > 0 else 0

    print(
        f"{attack:25s} "
        f"{detected}/{total} detected "
        f"({recall * 100:.2f}%)"
    )

# -----------------------------------
# 10. Investigate missed fraud
# -----------------------------------

missed_fraud = test_results[
    (test_results["is_fraud"] == 1) &
    (test_results["prediction"] == 0)
]

print("\n===================================")
print("MISSED FRAUD TRANSACTIONS")
print("===================================")

print(
    missed_fraud[
        [
            "customer_id",
            "attack_type",
            "amount",
            "transactions_last_1h",
            "transactions_last_24h",
            "behaviour_deviation_score",
            "identity_risk_score",
            "distance_from_home",
            "is_new_device",
            "is_new_location",
            "social_engineering_risk"
        ]
    ].to_string(index=False)
)

# -----------------------------------
# 11. Check prediction confidence
# -----------------------------------

if len(missed_fraud) > 0:

    missed_indices = missed_fraud.index

    probabilities = model.predict_proba(
        X_test.loc[missed_indices]
    )[:, 1]

    print("\n===================================")
    print("MISSED FRAUD CONFIDENCE")
    print("===================================")

    for index, probability in zip(
        missed_indices,
        probabilities
    ):
        print(
            f"Customer {df.loc[index, 'customer_id']}: "
            f"fraud probability = {probability:.4f}"
        )