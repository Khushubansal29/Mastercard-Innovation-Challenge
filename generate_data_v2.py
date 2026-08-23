import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# -----------------------------------
# Create normal transactions
# -----------------------------------

N = 5000

# Customer-level characteristics
customer_avg_amount = np.random.uniform(500, 4000, N)

# Generate transaction amounts around each customer's normal spending
amount = np.maximum(
    50,
    np.random.normal(
        customer_avg_amount,
        customer_avg_amount * 0.5
    )
)

# Keep amounts within a reasonable range
amount = np.clip(amount, 50, 50000)

data = {
    "amount": amount,

    # Most transactions happen during normal hours
    "hour": np.random.choice(
        range(24),
        N
    ),

    # Most customers use familiar devices
    "is_new_device": np.random.choice(
        [0, 1],
        N,
        p=[0.9, 0.1]
    ),

    # Most transactions happen near normal locations
    "is_new_location": np.random.choice(
        [0, 1],
        N,
        p=[0.9, 0.1]
    ),

    "distance_from_home": np.random.exponential(
        scale=20,
        size=N
    ),

    "avg_transaction_amount": customer_avg_amount,

    "transactions_last_1h": np.random.poisson(
        1.5,
        N
    ),

    "transactions_last_24h": np.random.poisson(
        8,
        N
    ),

    "account_age_days": np.random.randint(
        30,
        3000,
        N
    ),

    "identity_risk_score": np.random.uniform(
        0,
        0.25,
        N
    ),

    "verification_failed": np.random.choice(
        [0, 1],
        N,
        p=[0.98, 0.02]
    ),

    "social_engineering_risk": np.random.uniform(
        0,
        0.25,
        N
    ),

    "behaviour_deviation_score": np.random.uniform(
        0,
        0.3,
        N
    ),

    "attack_type": "normal",

    "is_fraud": 0
}

normal_df = pd.DataFrame(data)

print("Normal transactions created:", len(normal_df))

print("\nSample normal transactions:")
print(normal_df.head())

# -----------------------------------
# Attack 1: Account Takeover
# -----------------------------------

N_ATO = 1000

# Start with normal transactions
ato_df = normal_df.sample(
    N_ATO,
    random_state=43
).copy()

# Modify selected transactions to simulate account takeover

# Some transactions use a new device
ato_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.35, 0.65]
)

# Some transactions happen from a new location
ato_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.40, 0.60]
)

# Increase distance, but keep some transactions relatively close
ato_df["distance_from_home"] = np.clip(
    ato_df["distance_from_home"]
    + np.random.uniform(0, 500, N_ATO),
    0,
    1500
)

# Some transactions are larger than the customer's usual spending
amount_multiplier = np.random.uniform(
    1.0,
    4.0,
    N_ATO
)

ato_df["amount"] = np.clip(
    ato_df["amount"] * amount_multiplier,
    50,
    50000
)

# Increase behavioural deviation
ato_df["behaviour_deviation_score"] = np.clip(
    ato_df["behaviour_deviation_score"]
    + np.random.uniform(0.2, 0.6, N_ATO),
    0,
    1
)

# Slightly increase identity risk
ato_df["identity_risk_score"] = np.clip(
    ato_df["identity_risk_score"]
    + np.random.uniform(0.05, 0.4, N_ATO),
    0,
    1
)

# Mark as fraud
ato_df["attack_type"] = "account_takeover"
ato_df["is_fraud"] = 1

print("\nAccount Takeover transactions created:", len(ato_df))

print("\nSample ATO transactions:")
print(ato_df.head())