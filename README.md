🧠 Stroke Prediction System

Overview:
 
The Stroke Prediction System is a Machine Learning project that predicts whether a person is at risk of stroke based on health and lifestyle factors.

The model is deployed using FastAPI and provides predictions through a REST API.



Features

✅ Stroke Risk Prediction

✅ Probability Score Prediction

✅ Missing Value Handling

✅ Feature Scaling

✅ Categorical Feature Encoding

✅ FastAPI REST API

✅ Interactive Swagger Documentation



Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Uvicorn
* Joblib
* Pydantic


 Machine Learning Workflow

Data Preprocessing

* Missing Value Imputation using Median Strategy for Numerical Features
* Missing Value Imputation using Most Frequent Strategy for Categorical Features
* RobustScaler for Numerical Features
* OneHotEncoder for Categorical Features

Model
* Logistic Regression
* Class Weight = Balanced
* Pipeline-Based Training and Prediction



Model Performance

* Accuracy: 74.5%
* ROC-AUC Score: 0.82

The model was optimized for handling imbalanced stroke data by using class weight balancing.



Input Features

* age
* hypertension
* heart_disease
* avg_glucose_level
* bmi
* avg_RestingBP
* heart_cholesterol
* avg_MaxHR
* avg_Oldpeak
* heart_disease_rate
* gender
* ever_married
* work_type
* Residence_type
* smoking_status
* smoke_flag
* alco_flag



API Endpoints

GET /

Redirects to Swagger API Documentation.

GET /health

Returns API health status.

POST /stroke-prediction

Predicts stroke risk and returns:

* Prediction (0 or 1)
* Risk Level
* Probability Score



Project Structure

Stroke-Prediction-System/

├── app.py

├── stroke_prediction_model.pkl

├── stroke_final.csv

├── requirements.txt

└── README.md



Installation

Install required packages:

pip install -r requirements.txt



Run the Application

Start the FastAPI server:

uvicorn app:app --reload



Open API Documentation

http://127.0.0.1:8000/docs



Example Response

{
"prediction": 1,
"risk_level": "High Stroke Risk",
"probability": 57.35
}



Author

Jayasri Ramineedi

Aspiring Data Scientist | Machine Learning Enthusiast

Python | Scikit-Learn | FastAPI | Machine Learning
