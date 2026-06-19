🧠 Stroke Risk Prediction System

# Overview

The Stroke Risk Prediction System is a Machine Learning project that predicts whether a person is at risk of stroke based on symptoms and age.

The model is deployed using FastAPI and provides predictions through a REST API.


# Features

✅ Stroke Risk Prediction

✅ Probability Score Prediction

✅ Duplicate Record Removal

✅ Feature Scaling

✅ Model Comparison and Evaluation

✅ FastAPI REST API

✅ Interactive Swagger Documentation


# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Uvicorn
* Joblib
* Pydantic


# Machine Learning Workflow

# Data Preprocessing

* Duplicate Record Removal
* Feature Selection
* Train-Test Split
* StandardScaler for Age Feature
* Pipeline-Based Model Training

# Models Evaluated

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

# Final Model

* Logistic Regression
* Pipeline-Based Training and Prediction


# Model Performance

# Logistic Regression

* Accuracy: 100%
* ROC-AUC Score: 1.00

The model was trained and evaluated using a train-test split approach and integrated with a preprocessing pipeline.


# Input Features

* Chest Pain
* Shortness of Breath
* Irregular Heartbeat
* Fatigue & Weakness
* Dizziness
* Swelling (Edema)
* Pain in Neck/Jaw/Shoulder/Back
* Excessive Sweating
* Persistent Cough
* Nausea/Vomiting
* High Blood Pressure
* Chest Discomfort (Activity)
* Cold Hands/Feet
* Snoring/Sleep Apnea
* Anxiety/Feeling of Doom
* Age


# Target Variable

At Risk (Binary)

* 0 = Low Stroke Risk
* 1 = High Stroke Risk


# API Endpoints

# GET /

Returns project information.

# GET /health

Returns API health status.

# POST /stroke-prediction

Predicts stroke risk and returns:

* Prediction (0 or 1)
* Risk Level
* Probability Score


# Project Structure

Stroke-Risk-Prediction-System/

├── app.py

├── stroke_prediction_model.pkl

├── stroke_cleaned.csv

├── requirements.txt

└── README.md


# Installation

Install required packages:

pip install -r requirements.txt


# Run the Application

Start the FastAPI server:
uvicorn app:app --reload


# Open API Documentation

http://127.0.0.1:8000/docs



# Example Response

{
  "prediction": 1,
  "risk_level": "High Stroke Risk",
  "stroke_probability_percent": 96.42
}


# Author

Jayasri Ramineedi

Aspiring AI & Machine Learning Engineer

Python | Scikit-Learn | FastAPI | Machine Learning
