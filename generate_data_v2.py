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

# -----------------------------------
# Attack 2: AI Social Engineering / APP Fraud
# -----------------------------------

N_APP = 1000

# Start from normal transactions
app_df = normal_df.sample(
    N_APP,
    random_state=44
).copy()

# Most transactions still use a familiar device
app_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.80, 0.20]
)

# Most transactions still happen from a familiar location
app_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.80, 0.20]
)

# Payment amount is somewhat elevated,
# but can still look normal
app_df["amount"] = np.clip(
    app_df["amount"] * np.random.uniform(1.0, 3.0, N_APP),
    50,
    50000
)

# Social engineering is the main signal
app_df["social_engineering_risk"] = np.clip(
    app_df["social_engineering_risk"]
    + np.random.uniform(0.25, 0.65, N_APP),
    0,
    1
)

# Slight behavioural change
app_df["behaviour_deviation_score"] = np.clip(
    app_df["behaviour_deviation_score"]
    + np.random.uniform(0.1, 0.5, N_APP),
    0,
    1
)

# Identity itself may look perfectly normal
app_df["identity_risk_score"] = np.clip(
    app_df["identity_risk_score"]
    + np.random.uniform(0, 0.2, N_APP),
    0,
    1
)

# Mark as fraud
app_df["attack_type"] = "social_engineering"
app_df["is_fraud"] = 1

print(
    "\nSocial Engineering transactions created:",
    len(app_df)
)

print("\nSample Social Engineering transactions:")
print(app_df.head())

# -----------------------------------
# Attack 3: Synthetic Identity Fraud
# -----------------------------------

N_SYNTHETIC = 1000

# Start from normal transactions
synthetic_df = normal_df.sample(
    N_SYNTHETIC,
    random_state=45
).copy()

# Synthetic identities are often associated
# with relatively newer accounts
synthetic_df["account_age_days"] = np.random.randint(
    10,
    600,
    N_SYNTHETIC
)

# Increase identity risk, but keep overlap
synthetic_df["identity_risk_score"] = np.clip(
    synthetic_df["identity_risk_score"]
    + np.random.uniform(0.15, 0.65, N_SYNTHETIC),
    0,
    1
)

# Some identities fail verification,
# but many still pass
synthetic_df["verification_failed"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)

# Slightly unusual behaviour
synthetic_df["behaviour_deviation_score"] = np.clip(
    synthetic_df["behaviour_deviation_score"]
    + np.random.uniform(0.05, 0.4, N_SYNTHETIC),
    0,
    1
)

# Some use a new device/location
synthetic_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.65, 0.35]
)

synthetic_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)

# Keep payment amount relatively realistic
synthetic_df["amount"] = np.clip(
    synthetic_df["amount"]
    * np.random.uniform(0.8, 2.5, N_SYNTHETIC),
    50,
    30000
)

# Mark as fraud
synthetic_df["attack_type"] = "synthetic_identity"
synthetic_df["is_fraud"] = 1

print(
    "\nSynthetic Identity transactions created:",
    len(synthetic_df)
)

print("\nSample Synthetic Identity transactions:")
print(synthetic_df.head())

# -----------------------------------
# Attack 4: Transaction Velocity / Card Testing
# -----------------------------------

N_VELOCITY = 1000

# Start from normal transactions
velocity_df = normal_df.sample(
    N_VELOCITY,
    random_state=46
).copy()

# Increase transaction frequency
velocity_df["transactions_last_1h"] = np.random.randint(
    4,
    11,
    N_VELOCITY
)

velocity_df["transactions_last_24h"] = np.random.randint(
    10,
    35,
    N_VELOCITY
)

# Keep individual payments relatively small
velocity_df["amount"] = np.clip(
    velocity_df["amount"]
    * np.random.uniform(0.4, 1.5, N_VELOCITY),
    50,
    5000
)

# Slight behavioural deviation
velocity_df["behaviour_deviation_score"] = np.clip(
    velocity_df["behaviour_deviation_score"]
    + np.random.uniform(0.1, 0.5, N_VELOCITY),
    0,
    1
)

# Occasionally a new device/location is involved
velocity_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_VELOCITY,
    p=[0.75, 0.25]
)

velocity_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_VELOCITY,
    p=[0.75, 0.25]
)

# Mark as fraud
velocity_df["attack_type"] = "velocity_attack"
velocity_df["is_fraud"] = 1

print(
    "\nVelocity attack transactions created:",
    len(velocity_df)
)

print("\nSample Velocity transactions:")
print(velocity_df.head())

# -----------------------------------
# Attack 5: Behaviour Anomaly
# -----------------------------------

N_BEHAVIOUR = 1000

# Start from normal transactions
behaviour_df = normal_df.sample(
    N_BEHAVIOUR,
    random_state=47
).copy()

# Moderately change the transaction amount
behaviour_df["amount"] = np.clip(
    behaviour_df["amount"]
    * np.random.uniform(1.2, 3.0, N_BEHAVIOUR),
    50,
    30000
)

# Increase behavioural deviation
behaviour_df["behaviour_deviation_score"] = np.clip(
    behaviour_df["behaviour_deviation_score"]
    + np.random.uniform(0.25, 0.65, N_BEHAVIOUR),
    0,
    1
)

# Some transactions happen at unusual times
behaviour_df["hour"] = np.random.choice(
    range(24),
    N_BEHAVIOUR
)

# Some use a new device/location
behaviour_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.70, 0.30]
)

behaviour_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.65, 0.35]
)

# Moderate increase in distance
behaviour_df["distance_from_home"] = np.clip(
    behaviour_df["distance_from_home"]
    + np.random.uniform(0, 300, N_BEHAVIOUR),
    0,
    1000
)

# Mark as fraud
behaviour_df["attack_type"] = "behaviour_anomaly"
behaviour_df["is_fraud"] = 1

print(
    "\nBehaviour anomaly transactions created:",
    len(behaviour_df)
)

print("\nSample Behaviour Anomaly transactions:")
print(behaviour_df.head())

# -----------------------------------
# Attack 6: Deepfake-Assisted Identity Fraud
# -----------------------------------

N_DEEPFAKE = 1000

# Start from normal transactions
deepfake_df = normal_df.sample(
    N_DEEPFAKE,
    random_state=48
).copy()

# Relatively newer accounts are somewhat more common
deepfake_df["account_age_days"] = np.random.randint(
    10,
    800,
    N_DEEPFAKE
)

# Moderate-to-high identity risk
deepfake_df["identity_risk_score"] = np.clip(
    deepfake_df["identity_risk_score"]
    + np.random.uniform(0.2, 0.65, N_DEEPFAKE),
    0,
    1
)

# Most verification attempts can still pass
deepfake_df["verification_failed"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.75, 0.25]
)

# Some use a new device
deepfake_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.55, 0.45]
)

# Some use a new location
deepfake_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.60, 0.40]
)

# Moderate behavioural deviation
deepfake_df["behaviour_deviation_score"] = np.clip(
    deepfake_df["behaviour_deviation_score"]
    + np.random.uniform(0.15, 0.55, N_DEEPFAKE),
    0,
    1
)

# Keep payment amount realistic
deepfake_df["amount"] = np.clip(
    deepfake_df["amount"]
    * np.random.uniform(0.8, 2.5, N_DEEPFAKE),
    50,
    30000
)

# Mark as fraud
deepfake_df["attack_type"] = "deepfake_identity"
deepfake_df["is_fraud"] = 1

print(
    "\nDeepfake Identity transactions created:",
    len(deepfake_df)
)

print("\nSample Deepfake Identity transactions:")
print(deepfake_df.head())

# -----------------------------------
# Combine all transactions
# -----------------------------------

all_data_v2 = pd.concat([
    normal_df,
    ato_df,
    app_df,
    synthetic_df,
    velocity_df,
    behaviour_df,
    deepfake_df
], ignore_index=True)

# Shuffle the dataset
all_data_v2 = all_data_v2.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save V2 dataset
all_data_v2.to_csv(
    "data/fraud_dataset_v2.csv",
    index=False
)

print("\n===================================")
print("V2 DATASET")
print("===================================")

print("Total transactions:", len(all_data_v2))
print(
    "Legitimate:",
    (all_data_v2["is_fraud"] == 0).sum()
)
print(
    "Fraudulent:",
    (all_data_v2["is_fraud"] == 1).sum()
)

print("\nAttack distribution:")
print(all_data_v2["attack_type"].value_counts())

print("\nV2 dataset saved to:")
print("data/fraud_dataset_v2.csv")