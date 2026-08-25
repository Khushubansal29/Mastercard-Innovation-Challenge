import streamlit as st
import pandas as pd

from model import detect_transaction


# ===================================
# Load Red-Team dataset
# ===================================

attack_data = pd.read_csv(
    "data/fraud_dataset_v3.csv"
)


# ===================================
# Page configuration
# ===================================

st.set_page_config(
    page_title="AI Payment Defense Lab",
    page_icon="🛡️",
    layout="wide"
)

# ===================================
# Session history
# ===================================

if "attack_history" not in st.session_state:
    st.session_state["attack_history"] = []

# ===================================
# Custom styling
# ===================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ===================================
# Title
# ===================================

st.markdown(
    '<div class="main-title">🛡️ AI Payment Defense Lab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Red Team × Blue Team — GenAI Payment Fraud Defense'
    '</div>',
    unsafe_allow_html=True
)


# ===================================
# RED TEAM
# ===================================

st.divider()

st.header("🔴 Red Team — Attack Simulator")

attack_type = st.selectbox(
    "Choose an attack type",
    [
        "Account Takeover",
        "Social Engineering",
        "Synthetic Identity",
        "Velocity Attack",
        "Behaviour Anomaly",
        "Deepfake Identity"
    ]
)

generate_attack = st.button(
    "🔴 Generate Attack",
    type="primary",
    use_container_width=True
)


# ===================================
# Generate attack
# ===================================

if generate_attack:

    attack_mapping = {
        "Account Takeover": "account_takeover",
        "Social Engineering": "social_engineering",
        "Synthetic Identity": "synthetic_identity",
        "Velocity Attack": "velocity_attack",
        "Behaviour Anomaly": "behaviour_anomaly",
        "Deepfake Identity": "deepfake_identity"
    }

    selected_attack = attack_mapping[
        attack_type
    ]

    # Get examples of the selected attack
    attack_samples = attack_data[
        attack_data["attack_type"] == selected_attack
    ]

    # Pick one simulated attack
    selected_transaction = attack_samples.sample(
        1
    ).iloc[0]

    # Convert row into dictionary
    transaction = selected_transaction.to_dict()

    # Remove dataset-only columns
    transaction.pop("customer_id", None)
    transaction.pop("attack_type", None)
    transaction.pop("is_fraud", None)

    # Send transaction to Blue Team
    result = detect_transaction(
        transaction
    )

    # Store results
    st.session_state["transaction"] = transaction
    st.session_state["result"] = result
    st.session_state["attack_type"] = attack_type

    # Save attack to session history

    st.session_state["attack_history"].append({
        "attack_type": attack_type,
        "fraud_probability": result["fraud_probability"],
        "prediction": result["prediction"],
        "amount": transaction["amount"]
    })


# ===================================
# Display transaction
# ===================================

if "transaction" in st.session_state:

    transaction = st.session_state["transaction"]
    result = st.session_state["result"]

    st.divider()

    st.header("⚡ Simulated Transaction")

    st.caption(
        f"Attack simulated: "
        f"**{st.session_state['attack_type']}**"
    )

    # -----------------------------------
    # Transaction metrics — Row 1
    # -----------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Amount",
        f"₹{transaction['amount']:,.2f}"
    )

    col2.metric(
        "Transactions / hour",
        int(transaction["transactions_last_1h"])
    )

    col3.metric(
        "Transactions / 24h",
        int(transaction["transactions_last_24h"])
    )

    col4.metric(
        "Distance from home",
        f"{transaction['distance_from_home']:.1f} km"
    )

    # -----------------------------------
    # Transaction metrics — Row 2
    # -----------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Identity Risk",
        f"{transaction['identity_risk_score']:.1%}"
    )

    col2.metric(
        "Behaviour Deviation",
        f"{transaction['behaviour_deviation_score']:.1%}"
    )

    col3.metric(
        "Social Engineering Risk",
        f"{transaction['social_engineering_risk']:.1%}"
    )

    col4.metric(
        "Account Age",
        f"{int(transaction['account_age_days'])} days"
    )

    # -----------------------------------
    # Additional transaction details
    # -----------------------------------

    st.subheader("Transaction Signals")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "New Device",
        "YES" if transaction["is_new_device"] == 1 else "NO"
    )

    col2.metric(
        "New Location",
        "YES" if transaction["is_new_location"] == 1 else "NO"
    )

    col3.metric(
        "Verification Failed",
        "YES" if transaction["verification_failed"] == 1 else "NO"
    )

    # ===================================
    # BLUE TEAM
    # ===================================

    st.divider()

    st.header("🔵 Blue Team — Detection")

    # Get fraud probability from the model result
    probability = result["fraud_probability"]


    # -----------------------------------
    # Detection result
    # -----------------------------------

    if result["prediction"] == 1:

        st.error(
            f"🚨 FRAUD DETECTED — "
            f"{probability:.1%} risk"
        )

    else:

        st.success(
            f"✅ TRANSACTION APPEARS LEGITIMATE — "
            f"{1 - probability:.1%} confidence"
        )


    # -----------------------------------
    # Risk bar
    # -----------------------------------

    st.progress(
        probability,
        text=f"Fraud Risk: {probability:.1%}"
    )


    # -----------------------------------
    # Risk signals
    # -----------------------------------

    st.subheader("🔍 Risk Signals")

    signals = []


    if transaction["is_new_device"] == 1:

        signals.append(
            "New device detected"
        )


    if transaction["is_new_location"] == 1:

        signals.append(
            "New location detected"
        )


    if transaction["distance_from_home"] > 100:

        signals.append(
            f"Unusual distance from home "
            f"({transaction['distance_from_home']:.1f} km)"
        )


    if transaction["identity_risk_score"] > 0.5:

        signals.append(
            f"Elevated identity risk "
            f"({transaction['identity_risk_score']:.1%})"
        )


    if transaction["behaviour_deviation_score"] > 0.5:

        signals.append(
            f"High behaviour deviation "
            f"({transaction['behaviour_deviation_score']:.1%})"
        )


    if transaction["social_engineering_risk"] > 0.5:

        signals.append(
            f"Elevated social-engineering risk "
            f"({transaction['social_engineering_risk']:.1%})"
        )


    if transaction["transactions_last_1h"] >= 4:

        signals.append(
            f"High transaction velocity "
            f"({int(transaction['transactions_last_1h'])} "
            f"transactions/hour)"
        )


    if transaction["verification_failed"] == 1:

        signals.append(
            "Verification failure detected"
        )


    if transaction["account_age_days"] < 180:

        signals.append(
            f"Relatively new account "
            f"({int(transaction['account_age_days'])} days)"
        )


    # -----------------------------------
    # Display signals
    # -----------------------------------

    if signals:

        for signal in signals:

            st.warning(
                f"⚠️ {signal}"
            )

    else:

        st.success(
            "No major risk signals identified."
        )

# ===================================
# ATTACK PERFORMANCE
# ===================================

st.divider()

st.header("📊 Attack Performance")

try:

    attack_report = pd.read_csv(
        "data/attack_performance_report.csv"
    )

    # -----------------------------------
    # Overall metrics
    # -----------------------------------

    total_tested = attack_report["tested"].sum()
    total_detected = attack_report["detected"].sum()

    overall_recall = (
        total_detected / total_tested
        if total_tested > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
    "Attack Types",
    len(attack_report)
    )

    col2.metric(
    "Fraud Detection",
    f"{overall_recall:.1%}"
    )

    col3.metric(
    "Missed Attacks",
    int(attack_report["missed"].sum())
    )

# Load false-positive result
    try:

        evaluation_data = pd.read_csv(
        "data/evaluation_summary.csv"
        )

        false_positive_rate = evaluation_data[
        "false_positive_rate"
        ].iloc[0]

        col4.metric(
        "False Positive Rate",
        f"{false_positive_rate:.1%}"
        )

    except FileNotFoundError:

        col4.metric(
        "False Positive Rate",
        "N/A"
        )

    st.write("")

    # Convert recall to percentage
    attack_report["recall_percent"] = (
        attack_report["recall"] * 100
    )

    for _, row in attack_report.iterrows():

        col1, col2, col3 = st.columns([3, 1, 4])

        col1.write(
            f"🔴 {row['attack_type'].replace('_', ' ').title()}"
        )

        col2.write(
            f"**{row['recall_percent']:.0f}%**"
        )

        col3.progress(
            row["recall"]
        )

except FileNotFoundError:

    st.info(
        "Attack performance report "
        "will appear after model evaluation."
    )

# ===================================
# LIVE ATTACK HISTORY
# ===================================

st.divider()

st.header("🔄 Live Attack History")

history = st.session_state["attack_history"]


if history:

    history_data = []

    for item in history:

        history_data.append({
            "Attack": item["attack_type"],
            "Amount": f"₹{item['amount']:,.2f}",
            "Risk": f"{item['fraud_probability']:.1%}",
            "Detection": (
                "🚨 Fraud Detected"
                if item["prediction"] == 1
                else "✅ Legitimate"
            )
        })

    history_df = pd.DataFrame(
        history_data
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Generate an attack to start the live history."
    )