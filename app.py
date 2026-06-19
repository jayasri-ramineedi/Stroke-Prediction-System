import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib

app = FastAPI(
    title ="🧠 Stroke Risk Prediction System",
    description="""
    Machine Learning API for Stroke Risk Prediction.

    Features:
    - Symptom-Based Stroke Risk Evaluation
    - Age-Based Stroke Analysis
    - Data Cleaning & Duplicate Removal
    - Standard Scaling for Age Feature
    - Logistic Regression Model
    - Stroke Risk Probability Prediction
    """,
    version="1.0.0"  )
model = joblib.load("stroke_prediction_model.pkl")

class Stroke(BaseModel):
    chest_pain: int
    shortness_of_breath: int
    irregular_heartbeat: int
    fatigue_weakness: int
    dizziness: int
    swelling_edema: int
    neck_jaw_shoulder_back_pain: int
    excessive_sweating: int
    persistent_cough: int
    nausea_vomiting: int
    high_blood_pressure: int
    chest_discomfort_activity: int
    cold_hands_feet: int
    snoring_sleep_apnea: int
    anxiety_feeling_of_doom: int
    age: int 

@app.get("/", tags=["Home"])
def root():
    return RedirectResponse(url="/docs")
    
@app.get("/health", tags=["Health Check"])
def health():
    return{
        "message": "Stroke Prediction System Running Successfully"
    }

@app.post("/stroke-prediction", tags=["Prediction"])
def predict(stroke: Stroke):
    data = pd.DataFrame([{
    "Chest Pain": stroke.chest_pain,
    "Shortness of Breath": stroke.shortness_of_breath,
    "Irregular Heartbeat": stroke.irregular_heartbeat,
    "Fatigue & Weakness": stroke.fatigue_weakness,
    "Dizziness": stroke.dizziness,
    "Swelling (Edema)": stroke.swelling_edema,
    "Pain in Neck/Jaw/Shoulder/Back": stroke.neck_jaw_shoulder_back_pain,
    "Excessive Sweating": stroke.excessive_sweating,
    "Persistent Cough": stroke.persistent_cough,
    "Nausea/Vomiting": stroke.nausea_vomiting,
    "High Blood Pressure": stroke.high_blood_pressure,
    "Chest Discomfort (Activity)": stroke.chest_discomfort_activity,
    "Cold Hands/Feet": stroke.cold_hands_feet,
    "Snoring/Sleep Apnea": stroke.snoring_sleep_apnea,
    "Anxiety/Feeling of Doom": stroke.anxiety_feeling_of_doom,
    "Age": stroke.age
}])
    
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    risk_level = (
    "High Stroke Risk"
    if prediction == 1
    else "Low Stroke Risk"
)
    
    return {
        "status": "Success",
        "prediction": prediction,
        "risk_level": risk_level,
        "stroke_probability_percent": round(probability * 100, 2)
    }
