import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -----------------------------------
# 1. Load our dataset
# -----------------------------------

df = pd.read_csv("data/fraud_dataset.csv")

print("Dataset loaded!")
print("Total transactions:", len(df))


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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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