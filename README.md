# Customer Churn Prediction

A machine learning project that estimates whether a telecom customer may leave the service. It includes exploratory data analysis, model comparison, and a Streamlit app for making predictions.

## Project highlights

- Compared Logistic Regression, Random Forest, and XGBoost.
- XGBoost achieved a cross-validation ROC-AUC of about 0.847.
- Uses a 0.45 threshold to flag potential churners.
- Uses SHAP to help explain predictions.

## Dataset

The project uses the Telco Customer Churn dataset, containing customer account, service, billing, and churn information.

## Run locally

From the project folder, install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
