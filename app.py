from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_pipeline.joblib"


@st.cache_resource
def load_model_bundle():
    """Load the saved preprocessing-and-model pipeline once."""
    return joblib.load(MODEL_PATH)


st.title("Customer Churn Predictor")
st.write(
    "Enter a customer's service details to estimate their churn risk "
    "and see which features influenced the prediction."
)

try:
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Run the Phase 7 notebook cell to save the model bundle.")
    st.stop()

pipeline = bundle["pipeline"]
threshold = bundle["threshold"]

st.caption(f"Churn decision threshold: {threshold:.2f}")

with st.form("customer_details"):
    st.subheader("Customer details")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12, step=1
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        internet_service = st.selectbox(
            "Internet service", ["Fiber optic", "DSL", "No"]
        )
        phone_service = st.selectbox("Phone service", ["Yes", "No"])

    with col2:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0,
            value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=100000.0,
            value=840.0, step=10.0
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

    with col3:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has a partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])

    st.subheader("Services")

    if internet_service == "No":
        no_internet = ["No internet service"]
        online_security = st.selectbox("Online security", no_internet)
        online_backup = st.selectbox("Online backup", no_internet)
        device_protection = st.selectbox("Device protection", no_internet)
        tech_support = st.selectbox("Tech support", no_internet)
        streaming_tv = st.selectbox("Streaming TV", no_internet)
        streaming_movies = st.selectbox("Streaming movies", no_internet)
    else:
        service_options = ["No", "Yes"]
        service_col1, service_col2, service_col3 = st.columns(3)
        with service_col1:
            online_security = st.selectbox("Online security", service_options)
            online_backup = st.selectbox("Online backup", service_options)
        with service_col2:
            device_protection = st.selectbox("Device protection", service_options)
            tech_support = st.selectbox("Tech support", service_options)
        with service_col3:
            streaming_tv = st.selectbox("Streaming TV", service_options)
            streaming_movies = st.selectbox("Streaming movies", service_options)

    if phone_service == "No":
        multiple_lines = "No phone service"
        st.caption("Multiple lines: No phone service")
    else:
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

    submitted = st.form_submit_button("Predict churn", type="primary")


if submitted:
    # Keep these feature names aligned with the columns used during training.
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }])

    probability = float(pipeline.predict_proba(customer)[0, 1])
    predicted_churn = probability >= threshold

    st.subheader("Prediction")
    metric_col, label_col = st.columns(2)
    metric_col.metric("Churn probability", f"{probability:.1%}")

    if probability >= threshold:
        risk_label = "High"
        label_col.error("High risk — flagged for churn")
    elif probability >= 0.25:
        risk_label = "Medium"
        label_col.warning("Medium risk")
    else:
        risk_label = "Low"
        label_col.success("Low risk")

    st.progress(int(round(probability * 100)))
    st.write(
        f"At the saved threshold of **{threshold:.2f}**, "
        f"the model predicts **{'churn' if predicted_churn else 'stay'}**."
    )

    st.subheader("Why did the model make this prediction?")
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        transformed = preprocessor.transform(customer)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        else:
            transformed = np.asarray(transformed)

        feature_names = preprocessor.get_feature_names_out()
        booster = classifier.get_booster()

        contributions = booster.predict(
            xgb.DMatrix(transformed),
            pred_contribs=True,
        )[0]

        explanation = shap.Explanation(
            values=contributions[:-1],
            base_values=contributions[-1],
            data=transformed[0],
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption(
            "SHAP contributions show how features push the model output up or down. "
            "Numeric feature values in this chart are scaled by preprocessing."
        )
    except Exception as error:
        st.warning(f"Prediction succeeded, but the SHAP chart could not be shown: {error}")


with st.expander("About this model"):
    st.write(
        "This project uses the IBM Telco Customer Churn dataset and an XGBoost "
        "classifier inside a preprocessing pipeline. The pipeline imputes missing "
        "values, scales numeric features, and one-hot encodes categories."
    )
    st.write(f"Decision threshold: **{threshold:.2f}**")
    st.write("Held-out test results from this run:")
    st.write(bundle["test_metrics"])
    st.caption(
        "The threshold reflects a recall-focused choice. The model's predictions "
        "show patterns in this dataset and do not prove that a feature causes churn."
    )