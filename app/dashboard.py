import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
bundle = joblib.load("models/fraud_model.pkl")
model = bundle["model"]
features = bundle["features"]

# ==========================
# CUSTOM STYLING
# ==========================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #00ADB5;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.title("💳 Credit Card Fraud Detection")
st.markdown("### 🚀 AI-powered Fraud Detection System")

st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode", ["🔍 Manual Prediction", "📁 Upload CSV"])

# ==========================
# GAUGE FUNCTION
# ==========================
def fraud_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))
    return fig

# ==========================
# MANUAL MODE
# ==========================
if mode == "🔍 Manual Prediction":

    st.subheader("🧾 Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("Transaction Time", value=10000.0)
    with col2:
        amount = st.number_input("Transaction Amount", value=100.0)

    st.markdown("### ⚙️ Feature Inputs")

    features_input = {}
    cols = st.columns(4)

    for i in range(1, 29):
        with cols[i % 4]:
            features_input[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

    if st.button("🚀 Predict Fraud"):

        data = {
            "Time": time,
            "Amount": amount,
            **features_input
        }

        df = pd.DataFrame([data])
        df = df[features]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        st.markdown("---")

        colA, colB = st.columns(2)

        with colA:
            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ Legit Transaction")

        with colB:
            st.plotly_chart(fraud_gauge(prob), use_container_width=True)

# ==========================
# CSV MODE
# ==========================
elif mode == "📁 Upload CSV":

    st.subheader("📂 Upload Transactions CSV")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("### 📄 Data Preview")
        st.dataframe(df.head())

        try:
            df = df[features]

            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]

            df["Prediction"] = preds
            df["Fraud Probability"] = probs

            st.write("### 🔍 Prediction Results")
            st.dataframe(df.head())

            fraud_count = df["Prediction"].sum()
            total = len(df)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Transactions", total)
            with col2:
                st.metric("Fraud Cases", fraud_count)

            st.markdown("### 📊 Fraud Distribution")
            st.bar_chart(df["Prediction"].value_counts())

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("Built with ❤️ | Fraud Detection ML System")