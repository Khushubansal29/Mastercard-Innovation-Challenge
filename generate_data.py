import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of normal transactions
N = 5000

data = {
    "amount": np.random.uniform(100, 5000, N),
    "hour": np.random.randint(0, 24, N),
    "is_new_device": np.random.choice([0, 1], N, p=[0.9, 0.1]),
    "is_new_location": np.random.choice([0, 1], N, p=[0.9, 0.1]),
    "distance_from_home": np.random.uniform(0, 50, N),
    "avg_transaction_amount": np.random.uniform(500, 4000, N),
    "transactions_last_1h": np.random.randint(0, 4, N),
    "transactions_last_24h": np.random.randint(1, 15, N),
    "account_age_days": np.random.randint(30, 3000, N),
    "identity_risk_score": np.random.uniform(0, 0.3, N),
    "verification_failed": np.random.choice([0, 1], N, p=[0.98, 0.02]),
    "social_engineering_risk": np.random.uniform(0, 0.3, N),
    "behaviour_deviation_score": np.random.uniform(0, 0.3, N),
    "attack_type": "normal",
    "is_fraud": 0
}

normal_df = pd.DataFrame(data)

print("Normal transactions created:", len(normal_df))
print(normal_df.head())

# Attack 1: Account Takeover

N_ATO = 200

ato_data = {
    "amount": np.random.uniform(1000, 60000, N_ATO),

    # Some attacks happen at unusual times,
    # but not all of them
    "hour": np.random.randint(0, 24, N_ATO),

    # Only some attacks use a new device
    "is_new_device": np.random.choice(
        [0, 1], N_ATO, p=[0.35, 0.65]
    ),

    # Only some attacks come from a new location
    "is_new_location": np.random.choice(
        [0, 1], N_ATO, p=[0.35, 0.65]
    ),

    "distance_from_home": np.random.uniform(
        0, 1200, N_ATO
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 4000, N_ATO
    ),

    "transactions_last_1h": np.random.randint(
        0, 6, N_ATO
    ),

    "transactions_last_24h": np.random.randint(
        1, 20, N_ATO
    ),

    "account_age_days": np.random.randint(
        100, 3000, N_ATO
    ),

    # Still somewhat elevated, but not always extreme
    "identity_risk_score": np.random.uniform(
        0.2, 0.85, N_ATO
    ),

    "verification_failed": np.random.choice(
        [0, 1], N_ATO, p=[0.8, 0.2]
    ),

    "social_engineering_risk": np.random.uniform(
        0, 0.5, N_ATO
    ),

    # Main behavioural signal
    "behaviour_deviation_score": np.random.uniform(
        0.4, 1.0, N_ATO
    ),

    "attack_type": "account_takeover",
    "is_fraud": 1
}

ato_df = pd.DataFrame(ato_data)

print(
    "\nAccount Takeover transactions created:",
    len(ato_df)
)

print(ato_df.head())

# Attack 2: AI Social Engineering / APP Fraud

N_APP = 200

app_data = {
    "amount": np.random.uniform(1000, 50000, N_APP),

    # Can happen at any time
    "hour": np.random.randint(0, 24, N_APP),

    # Often uses the victim's normal device
    "is_new_device": np.random.choice(
        [0, 1], N_APP, p=[0.75, 0.25]
    ),

    # Often from a normal location
    "is_new_location": np.random.choice(
        [0, 1], N_APP, p=[0.75, 0.25]
    ),

    "distance_from_home": np.random.uniform(
        0, 300, N_APP
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 4000, N_APP
    ),

    "transactions_last_1h": np.random.randint(
        0, 5, N_APP
    ),

    "transactions_last_24h": np.random.randint(
        1, 18, N_APP
    ),

    "account_age_days": np.random.randint(
        100, 3000, N_APP
    ),

    # Usually not an identity problem
    "identity_risk_score": np.random.uniform(
        0.05, 0.45, N_APP
    ),

    "verification_failed": np.random.choice(
        [0, 1], N_APP, p=[0.95, 0.05]
    ),

    # Main signal, but deliberately subtle
    "social_engineering_risk": np.random.uniform(
        0.35, 0.85, N_APP
    ),

    "behaviour_deviation_score": np.random.uniform(
        0.2, 0.8, N_APP
    ),

    "attack_type": "social_engineering",
    "is_fraud": 1
}

app_df = pd.DataFrame(app_data)

print(
    "\nSocial Engineering transactions created:",
    len(app_df)
)

print(app_df.head())

# Attack 3: Synthetic Identity Fraud

N_SYNTHETIC = 200

synthetic_data = {
    "amount": np.random.uniform(500, 20000, N_SYNTHETIC),

    "hour": np.random.randint(0, 24, N_SYNTHETIC),

    # Most can look normal
    "is_new_device": np.random.choice(
        [0, 1], N_SYNTHETIC, p=[0.65, 0.35]
    ),

    "is_new_location": np.random.choice(
        [0, 1], N_SYNTHETIC, p=[0.65, 0.35]
    ),

    "distance_from_home": np.random.uniform(
        0, 500, N_SYNTHETIC
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 4000, N_SYNTHETIC
    ),

    "transactions_last_1h": np.random.randint(
        0, 5, N_SYNTHETIC
    ),

    "transactions_last_24h": np.random.randint(
        1, 18, N_SYNTHETIC
    ),

    # Newer accounts are more common
    "account_age_days": np.random.randint(
        10, 500, N_SYNTHETIC
    ),

    # Moderate rather than extreme identity risk
    "identity_risk_score": np.random.uniform(
        0.35, 0.85, N_SYNTHETIC
    ),

    # Many still pass verification
    "verification_failed": np.random.choice(
        [0, 1], N_SYNTHETIC, p=[0.65, 0.35]
    ),

    "social_engineering_risk": np.random.uniform(
        0.05, 0.4, N_SYNTHETIC
    ),

    "behaviour_deviation_score": np.random.uniform(
        0.2, 0.8, N_SYNTHETIC
    ),

    "attack_type": "synthetic_identity",
    "is_fraud": 1
}

synthetic_df = pd.DataFrame(synthetic_data)

print(
    "\nSynthetic Identity transactions created:",
    len(synthetic_df)
)

print(synthetic_df.head())

# Attack 4: Transaction Velocity / Card Testing

N_VELOCITY = 200

velocity_data = {
    "amount": np.random.uniform(50, 2000, N_VELOCITY),

    "hour": np.random.randint(0, 24, N_VELOCITY),

    "is_new_device": np.random.choice(
        [0, 1], N_VELOCITY, p=[0.65, 0.35]
    ),

    "is_new_location": np.random.choice(
        [0, 1], N_VELOCITY, p=[0.65, 0.35]
    ),

    "distance_from_home": np.random.uniform(
        0, 500, N_VELOCITY
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 3000, N_VELOCITY
    ),

    # Main signal — elevated, but not extreme
    "transactions_last_1h": np.random.randint(
        4, 13, N_VELOCITY
    ),

    "transactions_last_24h": np.random.randint(
        10, 40, N_VELOCITY
    ),

    "account_age_days": np.random.randint(
        100, 3000, N_VELOCITY
    ),

    "identity_risk_score": np.random.uniform(
        0.05, 0.5, N_VELOCITY
    ),

    "verification_failed": np.random.choice(
        [0, 1], N_VELOCITY, p=[0.9, 0.1]
    ),

    "social_engineering_risk": np.random.uniform(
        0, 0.4, N_VELOCITY
    ),

    "behaviour_deviation_score": np.random.uniform(
        0.3, 0.8, N_VELOCITY
    ),

    "attack_type": "velocity_attack",
    "is_fraud": 1
}

velocity_df = pd.DataFrame(velocity_data)

print(
    "\nVelocity attack transactions created:",
    len(velocity_df)
)

print(velocity_df.head())

# Attack 5: Behaviour Anomaly

N_BEHAVIOUR = 200

behaviour_data = {
    "amount": np.random.uniform(1000, 15000, N_BEHAVIOUR),

    "hour": np.random.randint(0, 24, N_BEHAVIOUR),

    "is_new_device": np.random.choice(
        [0, 1], N_BEHAVIOUR, p=[0.65, 0.35]
    ),

    "is_new_location": np.random.choice(
        [0, 1], N_BEHAVIOUR, p=[0.6, 0.4]
    ),

    "distance_from_home": np.random.uniform(
        0, 500, N_BEHAVIOUR
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 4000, N_BEHAVIOUR
    ),

    "transactions_last_1h": np.random.randint(
        0, 6, N_BEHAVIOUR
    ),

    "transactions_last_24h": np.random.randint(
        1, 20, N_BEHAVIOUR
    ),

    "account_age_days": np.random.randint(
        100, 3000, N_BEHAVIOUR
    ),

    "identity_risk_score": np.random.uniform(
        0.05, 0.6, N_BEHAVIOUR
    ),

    "verification_failed": np.random.choice(
        [0, 1], N_BEHAVIOUR, p=[0.9, 0.1]
    ),

    "social_engineering_risk": np.random.uniform(
        0, 0.5, N_BEHAVIOUR
    ),

    # Main signal — moderate deviation
    "behaviour_deviation_score": np.random.uniform(
        0.4, 0.85, N_BEHAVIOUR
    ),

    "attack_type": "behaviour_anomaly",
    "is_fraud": 1
}

behaviour_df = pd.DataFrame(behaviour_data)

print(
    "\nBehaviour anomaly transactions created:",
    len(behaviour_df)
)

print(behaviour_df.head())

# Attack 6: Deepfake-Assisted Identity Fraud

N_DEEPFAKE = 200

deepfake_data = {
    "amount": np.random.uniform(1000, 25000, N_DEEPFAKE),

    "hour": np.random.randint(0, 24, N_DEEPFAKE),

    # Many attacks still use a familiar device
    "is_new_device": np.random.choice(
        [0, 1], N_DEEPFAKE, p=[0.55, 0.45]
    ),

    # Location can also appear normal
    "is_new_location": np.random.choice(
        [0, 1], N_DEEPFAKE, p=[0.55, 0.45]
    ),

    "distance_from_home": np.random.uniform(
        0, 700, N_DEEPFAKE
    ),

    "avg_transaction_amount": np.random.uniform(
        500, 4000, N_DEEPFAKE
    ),

    "transactions_last_1h": np.random.randint(
        0, 6, N_DEEPFAKE
    ),

    "transactions_last_24h": np.random.randint(
        1, 20, N_DEEPFAKE
    ),

    # Relatively new identities are more common
    "account_age_days": np.random.randint(
        10, 600, N_DEEPFAKE
    ),

    # Moderate-to-high identity risk
    "identity_risk_score": np.random.uniform(
        0.4, 0.9, N_DEEPFAKE
    ),

    # Importantly, verification often succeeds
    "verification_failed": np.random.choice(
        [0, 1], N_DEEPFAKE, p=[0.7, 0.3]
    ),

    "social_engineering_risk": np.random.uniform(
        0.05, 0.5, N_DEEPFAKE
    ),

    # Moderate behavioural difference
    "behaviour_deviation_score": np.random.uniform(
        0.3, 0.85, N_DEEPFAKE
    ),

    "attack_type": "deepfake_identity",
    "is_fraud": 1
}

deepfake_df = pd.DataFrame(deepfake_data)

print(
    "\nDeepfake identity transactions created:",
    len(deepfake_df)
)

print(deepfake_df.head())

# Combine all transactions

all_data = pd.concat([
    normal_df,
    ato_df,
    app_df,
    synthetic_df,
    velocity_df,
    behaviour_df,
    deepfake_df
], ignore_index=True)

# Shuffle the dataset
all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
all_data.to_csv("data/fraud_dataset.csv", index=False)

print("\n-------------------------------")
print("FINAL DATASET")
print("-------------------------------")

print("Total transactions:", len(all_data))
print("Legitimate:", (all_data["is_fraud"] == 0).sum())
print("Fraudulent:", (all_data["is_fraud"] == 1).sum())

print("\nAttack distribution:")
print(all_data["attack_type"].value_counts())

print("\nDataset saved to: data/fraud_dataset.csv")