import streamlit as st
import requests
import json

# Page Configuration

st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide"
)

# Title

st.title("🧠 Stroke Risk Prediction System")
st.markdown(
    "Enter patient details below to predict stroke risk using the deployed Machine Learning model."
)

# API URL

API_URL = "https://stroke-prediction-system-lbqb.onrender.com/stroke-prediction"

# Input Form

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox("Chest Pain", [0, 1])
    shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1])
    irregular_heartbeat = st.selectbox("Irregular Heartbeat", [0, 1])
    fatigue_weakness = st.selectbox("Fatigue & Weakness", [0, 1])
    dizziness = st.selectbox("Dizziness", [0, 1])
    swelling_edema = st.selectbox("Swelling (Edema)", [0, 1])
    neck_jaw_shoulder_back_pain = st.selectbox(
        "Pain in Neck/Jaw/Shoulder/Back", [0, 1]
    )
    excessive_sweating = st.selectbox("Excessive Sweating", [0, 1])

with col2:
    persistent_cough = st.selectbox("Persistent Cough", [0, 1])
    nausea_vomiting = st.selectbox("Nausea/Vomiting", [0, 1])
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
    chest_discomfort_activity = st.selectbox(
        "Chest Discomfort (Activity)", [0, 1]
    )
    cold_hands_feet = st.selectbox("Cold Hands/Feet", [0, 1])
    snoring_sleep_apnea = st.selectbox("Snoring/Sleep Apnea", [0, 1])
    anxiety_feeling_of_doom = st.selectbox(
        "Anxiety/Feeling of Doom", [0, 1]
    )
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

# Prediction Button

if st.button("🔍 Predict Stroke Risk"):

    payload = {
        "chest_pain": chest_pain,
        "shortness_of_breath": shortness_of_breath,
        "irregular_heartbeat": irregular_heartbeat,
        "fatigue_weakness": fatigue_weakness,
        "dizziness": dizziness,
        "swelling_edema": swelling_edema,
        "neck_jaw_shoulder_back_pain": neck_jaw_shoulder_back_pain,
        "excessive_sweating": excessive_sweating,
        "persistent_cough": persistent_cough,
        "nausea_vomiting": nausea_vomiting,
        "high_blood_pressure": high_blood_pressure,
        "chest_discomfort_activity": chest_discomfort_activity,
        "cold_hands_feet": cold_hands_feet,
        "snoring_sleep_apnea": snoring_sleep_apnea,
        "anxiety_feeling_of_doom": anxiety_feeling_of_doom,
        "age": age
    }

    try:

        with st.spinner("Analyzing patient data..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Completed Successfully")

            st.subheader("📊 Prediction Results")

            prediction = result["prediction"]
            risk_level = result["risk_level"]
            probability = result["stroke_probability_percent"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:
                st.metric(
                    "Risk Level",
                    risk_level
                )

            with col3:
                st.metric(
                    "Stroke Probability",
                    f"{probability}%"
                )

            if prediction == 1:
                st.error(
                    f"⚠️ High Stroke Risk Detected ({probability}%)"
                )
            else:
                st.success(
                    f"✅ Low Stroke Risk ({probability}%)"
                )

        else:
            st.error(
                f"API Error: {response.status_code}"
            )

    except Exception as e:
        st.error(
            f"Connection Error: {e}"
        )

# Footer

st.markdown("---")
st.caption(
    "Stroke Risk Prediction System | Machine Learning + FastAPI + Streamlit"
)
