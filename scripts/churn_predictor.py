# churn_predictor.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

# 🚀 Load Model & Scaler from MLflow and use the latest
logged_model_uri = "runs:/<REPLACE_WITH_RUN_ID>/model"  # Update dynamically or use latest
model = mlflow.pyfunc.load_model(logged_model_uri)
scaler = joblib.load("../models/scaler.joblib")

# 🧾 Input Schema
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# 🔌 FastAPI App
app = FastAPI()

# 🧠 Prediction Endpoint
@app.post("/predict")
def predict_churn(data: CustomerFeatures):
    input_df = pd.DataFrame([data.model_dump()])
    input_encoded = pd.get_dummies(input_df).reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return {"churn": bool(prediction), "probability": round(probability, 4)}
