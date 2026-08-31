import numpy as np
import pandas as pd

# AI FRAUD ATTACK LAB — DATASET V4

np.random.seed(42)

# Dataset sizes

N_NORMAL = 8000

N_ATO = 100
N_APP = 100
N_SYNTHETIC = 100
N_VELOCITY = 100
N_BEHAVIOUR = 100
N_DEEPFAKE = 100

TOTAL_FRAUD = (
    N_ATO
    + N_APP
    + N_SYNTHETIC
    + N_VELOCITY
    + N_BEHAVIOUR
    + N_DEEPFAKE
)

print("=" * 55)
print("GENERATING FRAUD DATASET V4")
print("=" * 55)

print("\nNormal transactions:", N_NORMAL)
print("Fraud transactions:", TOTAL_FRAUD)
print("Total transactions:", N_NORMAL + TOTAL_FRAUD)


# 1. NORMAL TRANSACTIONS

normal_rows = []

for i in range(N_NORMAL):

    account_age = np.random.randint(30, 2500)

    avg_amount = np.random.lognormal(
        mean=np.log(2000),
        sigma=0.45
    )

    amount = np.random.normal(
        avg_amount,
        avg_amount * 0.30
    )

    amount = max(100, amount)

    transactions_24h = np.random.poisson(8)

    transactions_1h = np.random.poisson(
        max(transactions_24h / 24, 0.5)
    )

    row = {
        "customer_id": (i % 1000) + 1,

        "amount": amount,

        "hour": np.random.randint(0, 24),

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

        "transactions_last_1h": transactions_1h,

        "transactions_last_24h": max(
            transactions_24h,
            transactions_1h
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

    normal_rows.append(row)


normal_df = pd.DataFrame(normal_rows)


# 2. CREATE BASE FRAUD TRANSACTIONS

def create_fraud_base(n, attack_type):

    rows = []

    for i in range(n):

        account_age = np.random.randint(
            30,
            2500
        )

        avg_amount = np.random.lognormal(
            mean=np.log(2000),
            sigma=0.45
        )

        amount = np.random.normal(
            avg_amount,
            avg_amount * 0.40
        )

        amount = max(
            100,
            amount
        )

        transactions_24h = np.random.poisson(8)

        transactions_1h = np.random.poisson(
            max(transactions_24h / 24, 0.5)
        )

        row = {
            "customer_id": np.random.randint(
                1,
                1001
            ),

            "amount": amount,

            "hour": np.random.randint(
                0,
                24
            ),

            "is_new_device": np.random.choice(
                [0, 1],
                p=[0.85, 0.15]
            ),

            "is_new_location": np.random.choice(
                [0, 1],
                p=[0.85, 0.15]
            ),

            "distance_from_home": np.random.exponential(
                scale=15
            ),

            "avg_transaction_amount": avg_amount,

            "transactions_last_1h": transactions_1h,

            "transactions_last_24h": max(
                transactions_24h,
                transactions_1h
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

            "attack_type": attack_type,

            "is_fraud": 1
        }

        rows.append(row)

    return pd.DataFrame(rows)


# 3. ACCOUNT TAKEOVER

ato_df = create_fraud_base(
    N_ATO,
    "account_takeover"
)

# Moderate increase in behavioural deviation
ato_df["behaviour_deviation_score"] = np.clip(
    ato_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.05,
        0.35,
        N_ATO
    ),
    0,
    1
)

# Slight increase in identity risk
ato_df["identity_risk_score"] = np.clip(
    ato_df["identity_risk_score"]
    + np.random.uniform(
        0.02,
        0.25,
        N_ATO
    ),
    0,
    1
)

# Sometimes a new device
ato_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.45, 0.55]
)

# Sometimes a new location
ato_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_ATO,
    p=[0.50, 0.50]
)

# Moderate distance increase
ato_df["distance_from_home"] = np.clip(
    ato_df["distance_from_home"]
    + np.random.uniform(
        0,
        200,
        N_ATO
    ),
    0,
    800
)


# 4. SOCIAL ENGINEERING

app_df = create_fraud_base(
    N_APP,
    "social_engineering"
)

# Increase social engineering risk,
# but keep overlap with legitimate transactions
app_df["social_engineering_risk"] = np.clip(
    app_df["social_engineering_risk"]
    + np.random.uniform(
        0.15,
        0.45,
        N_APP
    ),
    0,
    1
)

# Moderate behavioural change
app_df["behaviour_deviation_score"] = np.clip(
    app_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.05,
        0.30,
        N_APP
    ),
    0,
    1
)

# Most still use familiar device/location
app_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.80, 0.20]
)

app_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_APP,
    p=[0.80, 0.20]
)


# 5. SYNTHETIC IDENTITY

synthetic_df = create_fraud_base(
    N_SYNTHETIC,
    "synthetic_identity"
)

# Moderate identity risk increase
synthetic_df["identity_risk_score"] = np.clip(
    synthetic_df["identity_risk_score"]
    + np.random.uniform(
        0.08,
        0.40,
        N_SYNTHETIC
    ),
    0,
    1
)

# Moderate behavioural change
synthetic_df["behaviour_deviation_score"] = np.clip(
    synthetic_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.05,
        0.30,
        N_SYNTHETIC
    ),
    0,
    1
)

# Some new devices
synthetic_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)

# Some new locations
synthetic_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.75, 0.25]
)

# Verification can still pass
synthetic_df["verification_failed"] = np.random.choice(
    [0, 1],
    N_SYNTHETIC,
    p=[0.70, 0.30]
)


# 6. VELOCITY ATTACK

velocity_df = create_fraud_base(
    N_VELOCITY,
    "velocity_attack"
)

# Increase transaction frequency,
# but avoid making it perfectly separable
velocity_df["transactions_last_1h"] = np.maximum(
    velocity_df["transactions_last_1h"]
    + np.random.randint(
        1,
        5,
        N_VELOCITY
    ),
    1
)

velocity_df["transactions_last_24h"] = np.maximum(
    velocity_df["transactions_last_24h"]
    + np.random.randint(
        3,
        12,
        N_VELOCITY
    ),
    velocity_df["transactions_last_1h"]
)

# Some attacks still use normal devices/locations
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

# Small behavioural increase
velocity_df["behaviour_deviation_score"] = np.clip(
    velocity_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.05,
        0.30,
        N_VELOCITY
    ),
    0,
    1
)


# 7. BEHAVIOUR ANOMALY

behaviour_df = create_fraud_base(
    N_BEHAVIOUR,
    "behaviour_anomaly"
)

# Increase behavioural deviation,
# but maintain overlap
behaviour_df["behaviour_deviation_score"] = np.clip(
    behaviour_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.10,
        0.40,
        N_BEHAVIOUR
    ),
    0,
    1
)

# Moderate distance increase
behaviour_df["distance_from_home"] = np.clip(
    behaviour_df["distance_from_home"]
    + np.random.uniform(
        0,
        180,
        N_BEHAVIOUR
    ),
    0,
    700
)

# Some new devices
behaviour_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.70, 0.30]
)

# Some new locations
behaviour_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_BEHAVIOUR,
    p=[0.65, 0.35]
)


# 8. DEEPFAKE IDENTITY

deepfake_df = create_fraud_base(
    N_DEEPFAKE,
    "deepfake_identity"
)

# Moderate identity-risk increase
deepfake_df["identity_risk_score"] = np.clip(
    deepfake_df["identity_risk_score"]
    + np.random.uniform(
        0.08,
        0.40,
        N_DEEPFAKE
    ),
    0,
    1
)

# Moderate behavioural change
deepfake_df["behaviour_deviation_score"] = np.clip(
    deepfake_df["behaviour_deviation_score"]
    + np.random.uniform(
        0.05,
        0.35,
        N_DEEPFAKE
    ),
    0,
    1
)

# Some new devices
deepfake_df["is_new_device"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.55, 0.45]
)

# Some new locations
deepfake_df["is_new_location"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.60, 0.40]
)

# Verification can still pass
deepfake_df["verification_failed"] = np.random.choice(
    [0, 1],
    N_DEEPFAKE,
    p=[0.75, 0.25]
)


# 9. COMBINE ALL DATA

fraud_df = pd.concat(
    [
        ato_df,
        app_df,
        synthetic_df,
        velocity_df,
        behaviour_df,
        deepfake_df
    ],
    ignore_index=True
)

df = pd.concat(
    [
        normal_df,
        fraud_df
    ],
    ignore_index=True
)


# 10. ADD REALISTIC AMOUNT VARIATION TO SOME FRAUD

fraud_mask = df["is_fraud"] == 1

df.loc[fraud_mask, "amount"] = np.where(
    np.random.random(fraud_mask.sum()) < 0.35,

    df.loc[fraud_mask, "avg_transaction_amount"]
    * np.random.uniform(
        1.1,
        2.2,
        fraud_mask.sum()
    ),

    df.loc[fraud_mask, "amount"]
)


df["amount"] = df["amount"].clip(
    lower=50
)


# 11. KEEP DISTANCE REASONABLE

df["distance_from_home"] = df[
    "distance_from_home"
].clip(
    lower=0,
    upper=1500
)


# 12. SHUFFLE DATASET

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# 13. SAVE DATASET

output_file = "data/fraud_dataset_v4.csv"

df.to_csv(
    output_file,
    index=False
)


# 14. DATASET SUMMARY

print("\n" + "=" * 55)
print("DATASET V4 CREATED")
print("=" * 55)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nTarget distribution:")
print(
    df["is_fraud"].value_counts()
)

print("\nTarget percentages:")
print(
    (df["is_fraud"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nAttack distribution:")
print(
    df["attack_type"].value_counts()
)

print("\nMissing values:")
print(
    df.isnull().sum().sum()
)

print("\nFraud feature means:")
print(
    df.groupby("is_fraud")[
        [
            "amount",
            "transactions_last_1h",
            "transactions_last_24h",
            "behaviour_deviation_score",
            "identity_risk_score",
            "distance_from_home",
            "is_new_device",
            "is_new_location",
            "social_engineering_risk",
            "verification_failed"
        ]
    ].mean().round(3)
)

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 55)
print("V4 GENERATION COMPLETE")
print("=" * 55)