import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import sys
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = glob.glob("models/best_model_*.pkl")
    if not model_path:
        st.error("No trained model found. Please run `python src/train.py` first.")
        st.stop()
    model = joblib.load(model_path[0])
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.markdown("Fill in the customer details below to get an instant churn risk prediction.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Account Info")
    tenure         = st.slider("Tenure (months)", 0, 72, 12)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment        = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly        = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total          = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure), step=1.0)

with col2:
    st.subheader("Services")
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec     = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
    online_bak     = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
    tech_sup       = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()
col3, col4 = st.columns(2)
with col3:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
with col4:
    partner    = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

phone      = st.selectbox("Phone Service", ["Yes", "No"])
multi_line = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):

    # Build a raw dict matching the original CSV columns
    raw = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi_line,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    df = pd.DataFrame([raw])

    # Feature engineering (mirrors data_prep.py)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1,6,12,24,48,72],
        labels=["0-6mo","6-12mo","1-2yr","2-4yr","4-6yr"]
    )
    service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    df["num_addon_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # One-hot encode
    df_enc = pd.get_dummies(df, drop_first=True)

    # Align columns
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols].astype(float)

    # Predict
    if "LogisticRegression" in str(type(model)):
        X = scaler.transform(df_enc)
    else:
        X = df_enc.values

    prob = model.predict_proba(X)[0][1]

    # ── Result ─────────────────────────────────────────────────────────────────
    st.subheader("Prediction Result")

    if prob >= 0.5:
        risk = "🔴 High Risk"
        color = "red"
        msg = "This customer is likely to churn. Consider a retention offer."
    elif prob >= 0.25:
        risk = "🟡 Medium Risk"
        color = "orange"
        msg = "This customer shows some churn signals. Monitor closely."
    else:
        risk = "🟢 Low Risk"
        color = "green"
        msg = "This customer is likely to stay."

    st.metric(label="Churn Probability", value=f"{prob:.1%}")
    st.markdown(f"**Risk Level:** :{color}[{risk}]")
    st.info(msg)

    # Key drivers (based on what we know from SHAP/EDA)
    st.subheader("Key Risk Factors")
    factors = []
    if contract == "Month-to-month":
        factors.append("⚠️ Month-to-month contract — highest churn risk contract type")
    if tenure < 6:
        factors.append("⚠️ Very new customer (< 6 months) — high early churn risk")
    if internet == "Fiber optic":
        factors.append("⚠️ Fiber optic internet — associated with higher churn rates")
    if payment == "Electronic check":
        factors.append("⚠️ Electronic check payment — linked to higher churn in the data")
    if not factors:
        factors.append("✅ No major risk flags detected for this customer")
    for f in factors:
        st.markdown(f)
