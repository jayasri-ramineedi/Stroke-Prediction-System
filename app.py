import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict
import joblib

# Load Model
model = joblib.load("stroke_prediction_model.pkl")

# FastAPI App
app = FastAPI(
    title="Stroke Prediction API",
    description="""
    Machine Learning API for Stroke Risk Prediction.

    Features:
    - Missing Value Handling
    - One-Hot Encoding
    - Robust Scaling
    - Logistic Regression Model
    - Stroke Risk Probability
    """,
    version="1.0.0"
)

# Input Schema
class Stroke(BaseModel):
    age: int
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    avg_RestingBP: float
    heart_cholesterol: float
    avg_MaxHR: float
    avg_Oldpeak: float
    heart_disease_rate: float
    gender: str
    ever_married: str
    work_type: str
    Residence_type: str
    smoking_status: str
    smoke_flag: int
    alco_flag: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 50,
                "hypertension": 0,
                "heart_disease": 0,
                "avg_glucose_level": 69.92,
                "bmi": 18.7,
                "avg_RestingBP": 125.71,
                "heart_cholesterol": 219.28,
                "avg_MaxHR": 140.57,
                "avg_Oldpeak": 0.52,
                "heart_disease_rate": 0.28,
                "gender": "Female",
                "ever_married": "Yes",
                "work_type": "Self-employed",
                "Residence_type": "Urban",
                "smoking_status": "formerly smoked",
                "smoke_flag": 1,
                "alco_flag": 0
            }
        }
    )

# Redirect to Swagger Docs
@app.get("/", tags=["Home"])
def root():
    return RedirectResponse(url="/docs")

# Health Check Endpoint
@app.get("/health", tags=["Health Check"])
def health():
    return {
        "message": "Stroke Prediction System Running Successfully"
    }

# Prediction Endpoint
@app.post("/stroke-prediction", tags=["Prediction"])
def predict(stroke: Stroke):

    try:
        data = pd.DataFrame([{
            "age": stroke.age,
            "hypertension": stroke.hypertension,
            "heart_disease": stroke.heart_disease,
            "avg_glucose_level": stroke.avg_glucose_level,
            "bmi": stroke.bmi,
            "avg_RestingBP": stroke.avg_RestingBP,
            "heart_cholesterol": stroke.heart_cholesterol,
            "avg_MaxHR": stroke.avg_MaxHR,
            "avg_Oldpeak": stroke.avg_Oldpeak,
            "heart_disease_rate": stroke.heart_disease_rate,
            "gender": stroke.gender,
            "ever_married": stroke.ever_married,
            "work_type": stroke.work_type,
            "Residence_type": stroke.Residence_type,
            "smoking_status": stroke.smoking_status,
            "smoke_flag": stroke.smoke_flag,
            "alco_flag": stroke.alco_flag
        }])

        prediction = int(model.predict(data)[0])

        probability = float(
            model.predict_proba(data)[0][1]
        )

        risk_level = (
            "High Stroke Risk"
            if prediction == 1
            else "Low Stroke Risk"
        )

        return {
            "prediction": prediction,
            "risk_level": risk_level,
            "stroke_probability_percent": round(probability * 100, 2),
            "status": "Success"
        }

    except Exception as e:
        return {
            "status": "Error",
            "message": str(e)
        }