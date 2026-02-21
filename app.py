import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import scipy.sparse as sp

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction App")
st.write("This app predicts the probability that a telecom customer will churn.")

# ----------------------------
# Load trained pipeline model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("tuned_random_forest_pipeline.pkl")

model = load_model()

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.30, 0.05)

# ----------------------------
# SHAP Global Summary
# ----------------------------
st.subheader("🔎 Model Explainability (Global)")
st.image(
    "shap_summary.png",
    caption="SHAP Summary Plot (Global Feature Impact on Churn)",
    use_container_width=True
)

# ----------------------------
# User Input Section
# ----------------------------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=50000.0, value=1000.0)

# ----------------------------
# Create input DataFrame
# ----------------------------
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("Predict Churn"):

    # Predict probability
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= threshold)

    st.metric("Churn Probability", f"{prob:.2%}")

    if pred == 1:
        st.error(f"High Risk of Churn (Threshold = {threshold})")
    else:
        st.success(f"Low Risk of Churn (Threshold = {threshold})")

    st.write("### Customer Summary")
    st.dataframe(input_data)

    # ----------------------------
    # SHAP Local Explanation
    # ----------------------------
    with st.expander("Explain this prediction (SHAP)"):
        preprocessor = model.named_steps["preprocessor"]
        rf_clf = model.named_steps["classifier"]

        X_one = preprocessor.transform(input_data)

        if sp.issparse(X_one):
            X_one = X_one.toarray()

        explainer = shap.TreeExplainer(rf_clf)
        sv_one = explainer.shap_values(X_one)

        # Handle SHAP shape differences
        if isinstance(sv_one, list):
            sv_churn = sv_one[1][0]
            base_value = explainer.expected_value[1]
        else:
            sv = np.array(sv_one)
            if sv.ndim == 3:
                sv_churn = sv[0, :, 1]
                base_value = explainer.expected_value[1]
            else:
                sv_churn = sv[0]
                base_value = explainer.expected_value

        feat_names = preprocessor.get_feature_names_out()

        exp = shap.Explanation(
            values=sv_churn,
            base_values=base_value,
            data=X_one[0],
            feature_names=feat_names
        )

        fig = plt.figure()
        shap.plots.waterfall(exp, max_display=12, show=False)
        st.pyplot(fig)
        plt.close(fig)
