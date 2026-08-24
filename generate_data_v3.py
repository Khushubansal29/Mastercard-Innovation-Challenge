import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------------
# Basic settings
# -----------------------------------

N_CUSTOMERS = 1000
TRANSACTIONS_PER_CUSTOMER = 8

# Each customer gets a unique ID
customer_ids = np.arange(1, N_CUSTOMERS + 1)

# Each customer has their own normal spending level
customer_avg_amount = np.random.uniform(
    500,
    4000,
    N_CUSTOMERS
)

# Each customer has their own normal transaction frequency
customer_avg_txn_per_day = np.random.uniform(
    2,
    15,
    N_CUSTOMERS
)

# Each customer has their own account age
customer_account_age = np.random.randint(
    30,
    3000,
    N_CUSTOMERS
)


# -----------------------------------
# Generate normal transactions
# -----------------------------------

rows = []

for i in range(N_CUSTOMERS):

    customer_id = customer_ids[i]
    avg_amount = customer_avg_amount[i]
    account_age = customer_account_age[i]

    for _ in range(TRANSACTIONS_PER_CUSTOMER):

        amount = np.random.normal(
            avg_amount,
            avg_amount * 0.4
        )

        amount = np.clip(
            amount,
            50,
            30000
        )

        normal_daily_rate = customer_avg_txn_per_day[i]

        normal_hourly_rate = normal_daily_rate / 24

        row = {
            "customer_id": customer_id,

            "amount": amount,

            "hour": np.random.randint(
                7,
                23
            ),

            "is_new_device": np.random.choice(
                [0, 1],
                p=[0.95, 0.05]
            ),

            "is_new_location": np.random.choice(
                [0, 1],
                p=[0.95, 0.05]
            ),

            "distance_from_home": np.random.exponential(
                scale=15
            ),

            "avg_transaction_amount": avg_amount,

            "transactions_last_1h": np.random.poisson(
                normal_hourly_rate
            ),

            "transactions_last_24h": np.random.poisson(
                normal_daily_rate
            ),

            "account_age_days": account_age,

            "identity_risk_score": np.random.uniform(
                0,
                0.2
            ),

            "verification_failed": np.random.choice(
                [0, 1],
                p=[0.98, 0.02]
            ),

            "social_engineering_risk": np.random.uniform(
                0,
                0.2
            ),

            "behaviour_deviation_score": np.random.uniform(
                0,
                0.25
            ),

            "attack_type": "normal",

            "is_fraud": 0
        }

        rows.append(row)


normal_df = pd.DataFrame(rows)

print(
    "Normal transactions created:",
    len(normal_df)
)

print(
    "Customers created:",
    normal_df["customer_id"].nunique()
)

print("\nSample:")
print(normal_df.head())

# -----------------------------------
# Attack 1: Account Takeover
# -----------------------------------

N_ATO = 100

# Pick normal transactions as the starting point
ato_df = normal_df.sample(
    N_ATO,
    random_state=43
).copy()

# ATO may involve a new device
ato_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.30, 0.70]
)

# ATO may involve a new location
ato_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.35, 0.65]
)

# Change the transaction amount,
# but keep some overlap with normal spending
ato_df["amount"] = np.clip(
    ato_df["amount"] *
    np.random.uniform(1.0, 4.0, N_ATO),
    50,
    50000
)

# Increase distance, but not always dramatically
ato_df["distance_from_home"] = np.clip(
    ato_df["distance_from_home"] +
    np.random.uniform(0, 400, N_ATO),
    0,
    1500
)

# Increase behavioural deviation
ato_df["behaviour_deviation_score"] = np.clip(
    ato_df["behaviour_deviation_score"] +
    np.random.uniform(0.15, 0.6, N_ATO),
    0,
    1
)

# Slightly increase identity risk
ato_df["identity_risk_score"] = np.clip(
    ato_df["identity_risk_score"] +
    np.random.uniform(0.05, 0.4, N_ATO),
    0,
    1
)

# Mark as fraud
ato_df["attack_type"] = "account_takeover"
ato_df["is_fraud"] = 1

print(
    "\nATO transactions created:",
    len(ato_df)
)

# -----------------------------------
# Attack 2: AI Social Engineering / APP Fraud
# -----------------------------------

N_APP = 100

# Start from normal customer transactions
app_df = normal_df.sample(
    N_APP,
    random_state=44
).copy()

# Most attacks still use the customer's normal device
app_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.85, 0.15]
)

# Most attacks happen from a familiar location
app_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.85, 0.15]
)

# Increase payment amount moderately
app_df["amount"] = np.clip(
    app_df["amount"] *
    np.random.uniform(1.0, 3.0, N_APP),
    50,
    30000
)

# Increase social engineering risk
app_df["social_engineering_risk"] = np.clip(
    app_df["social_engineering_risk"] +
    np.random.uniform(0.25, 0.65, N_APP),
    0,
    1
)

# Moderate behavioural deviation
app_df["behaviour_deviation_score"] = np.clip(
    app_df["behaviour_deviation_score"] +
    np.random.uniform(0.1, 0.45, N_APP),
    0,
    1
)

# Identity itself may remain normal
app_df["identity_risk_score"] = np.clip(
    app_df["identity_risk_score"] +
    np.random.uniform(0, 0.15, N_APP),
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

# -----------------------------------
# Attack 3: Synthetic Identity Fraud
# -----------------------------------

N_SYNTHETIC = 100

# Start from normal customer transactions
synthetic_df = normal_df.sample(
    N_SYNTHETIC,
    random_state=45
).copy()

# Synthetic identities are more likely to have
# relatively newer accounts
synthetic_df["account_age_days"] = np.random.randint(
    10,
    700,
    N_SYNTHETIC
)

# Moderately increase identity risk
synthetic_df["identity_risk_score"] = np.clip(
    synthetic_df["identity_risk_score"] +
    np.random.uniform(0.15, 0.65, N_SYNTHETIC),
    0,
    1
)

# Some verification attempts fail,
# but many still pass
synthetic_df["verification_failed"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)

# Slight behavioural deviation
synthetic_df["behaviour_deviation_score"] = np.clip(
    synthetic_df["behaviour_deviation_score"] +
    np.random.uniform(0.05, 0.4, N_SYNTHETIC),
    0,
    1
)

# Some use new devices
synthetic_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)

# Some appear from new locations
synthetic_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.75, 0.25]
)

# Keep transaction amount realistic
synthetic_df["amount"] = np.clip(
    synthetic_df["amount"] *
    np.random.uniform(0.8, 2.5, N_SYNTHETIC),
    50,
    25000
)

# Mark as fraud
synthetic_df["attack_type"] = "synthetic_identity"
synthetic_df["is_fraud"] = 1

print(
    "\nSynthetic Identity transactions created:",
    len(synthetic_df)
)

# -----------------------------------
# Attack 4: Transaction Velocity / Card Testing
# -----------------------------------

N_VELOCITY = 100

# Start from normal customer transactions
velocity_df = normal_df.sample(
    N_VELOCITY,
    random_state=46
).copy()

# Make transaction frequency unusually high
velocity_df["transactions_last_1h"] = np.maximum(
    velocity_df["transactions_last_1h"] + np.random.randint(
        3,
        8,
        N_VELOCITY
    ),
    4
)

velocity_df["transactions_last_24h"] = np.maximum(
    velocity_df["transactions_last_24h"] +
    np.random.randint(5, 20, N_VELOCITY),
    velocity_df["transactions_last_1h"]
)

# Keep individual payments relatively small
velocity_df["amount"] = np.clip(
    velocity_df["amount"] *
    np.random.uniform(0.5, 1.5, N_VELOCITY),
    50,
    5000
)

# Moderate behavioural change
velocity_df["behaviour_deviation_score"] = np.clip(
    velocity_df["behaviour_deviation_score"] +
    np.random.uniform(0.1, 0.45, N_VELOCITY),
    0,
    1
)

# Most transactions still use familiar devices
velocity_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_VELOCITY,
    p=[0.75, 0.25]
)

# Most still happen from familiar locations
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

# -----------------------------------
# Attack 5: Behaviour Anomaly
# -----------------------------------

N_BEHAVIOUR = 100

# Start from normal customer transactions
behaviour_df = normal_df.sample(
    N_BEHAVIOUR,
    random_state=47
).copy()

# Moderately increase transaction amount
behaviour_df["amount"] = np.clip(
    behaviour_df["amount"] *
    np.random.uniform(1.2, 3.0, N_BEHAVIOUR),
    50,
    30000
)

# Increase behavioural deviation
behaviour_df["behaviour_deviation_score"] = np.clip(
    behaviour_df["behaviour_deviation_score"] +
    np.random.uniform(0.25, 0.65, N_BEHAVIOUR),
    0,
    1
)

# Change transaction time
behaviour_df["hour"] = np.random.randint(
    0,
    24,
    N_BEHAVIOUR
)

# Some attacks involve a new device
behaviour_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.70, 0.30]
)

# Some attacks involve a new location
behaviour_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.65, 0.35]
)

# Moderately increase distance
behaviour_df["distance_from_home"] = np.clip(
    behaviour_df["distance_from_home"] +
    np.random.uniform(0, 300, N_BEHAVIOUR),
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

# -----------------------------------
# Attack 6: Deepfake-Assisted Identity Fraud
# -----------------------------------

N_DEEPFAKE = 100

# Start from normal customer transactions
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

# Increase identity risk, but keep overlap
deepfake_df["identity_risk_score"] = np.clip(
    deepfake_df["identity_risk_score"] +
    np.random.uniform(0.2, 0.65, N_DEEPFAKE),
    0,
    1
)

# Verification can still pass
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
    deepfake_df["behaviour_deviation_score"] +
    np.random.uniform(0.15, 0.55, N_DEEPFAKE),
    0,
    1
)

# Keep transaction amount realistic
deepfake_df["amount"] = np.clip(
    deepfake_df["amount"] *
    np.random.uniform(0.8, 2.5, N_DEEPFAKE),
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

# -----------------------------------
# Combine all transactions
# -----------------------------------

all_data_v3 = pd.concat([
    normal_df,
    ato_df,
    app_df,
    synthetic_df,
    velocity_df,
    behaviour_df,
    deepfake_df
], ignore_index=True)

# Shuffle transactions
all_data_v3 = all_data_v3.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save V3 dataset
all_data_v3.to_csv(
    "data/fraud_dataset_v3.csv",
    index=False
)

print("\n===================================")
print("V3 DATASET")
print("===================================")

print(
    "Total transactions:",
    len(all_data_v3)
)

print(
    "Legitimate:",
    (all_data_v3["is_fraud"] == 0).sum()
)

print(
    "Fraudulent:",
    (all_data_v3["is_fraud"] == 1).sum()
)

print("\nAttack distribution:")
print(
    all_data_v3["attack_type"].value_counts()
)

print("\nV3 dataset saved to:")
print("data/fraud_dataset_v3.csv")

