import joblib
import pandas as pd


# Load trained model
model = joblib.load("model.pkl")


# Features expected by the model
FEATURES = [
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


def detect_transaction(transaction):
    """
    Predict whether a transaction is fraudulent.
    """

    df = pd.DataFrame(
        [transaction]
    )

    # Calculate derived feature
    df["velocity_ratio"] = (
        df["transactions_last_1h"] /
        (
            df["transactions_last_24h"] / 24
            + 0.1
        )
    )

    # Keep only model features
    X = df[FEATURES]

    # Prediction
    prediction = model.predict(X)[0]

    # Fraud probability
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }

if __name__ == "__main__":

    test_transaction = {
        "amount": 2500,
        "hour": 21,
        "is_new_device": 1,
        "is_new_location": 1,
        "distance_from_home": 300,
        "avg_transaction_amount": 1500,
        "transactions_last_1h": 6,
        "transactions_last_24h": 15,
        "account_age_days": 200,
        "identity_risk_score": 0.7,
        "verification_failed": 0,
        "social_engineering_risk": 0.3,
        "behaviour_deviation_score": 0.6
    }

    result = detect_transaction(
        test_transaction
    )

    print("\nPrediction:")
    print(result)