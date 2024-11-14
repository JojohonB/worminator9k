#streamlit preparation.
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load the trained model.
MODEL_PATH = 'model/model.joblib'
model = joblib.load(MODEL_PATH)

#user input ratio calculations.
def calculate_user_ratio_features(measurements):
    for key in measurements:
        if measurements[key] == 0:
            measurements[key] = 0.1  

    ratio_features = {
        'WHR': measurements["waistcircumference"] / measurements["buttockcircumference"],
        'HBS': measurements["hipbreadth"] / measurements["stature"],
        'CS': measurements["chestcircumference"] / measurements["stature"],
        'FSR': measurements["forearmcircumferenceflexed"] / measurements["stature"],
        'CBR': measurements["calfcircumference"] / measurements["buttockcircumference"],
        'BBSR': measurements["biacromialbreadth"] / measurements["stature"],
        'BBHB': measurements["biacromialbreadth"] / measurements["hipbreadth"],
        'ANKLS': measurements["anklecircumference"] / measurements["stature"],
        'FLS': measurements["forearmhandlength"] / measurements["stature"],
        'WCS': measurements["wristcircumference"] / measurements["stature"]
    }
    return ratio_features

st.title("Gender Prediction Model")

st.write("Enter your measurements (in millimeters):")

stature = st.number_input("Stature", min_value=0.0, format="%.2f")
biacromialbreadth = st.number_input("Biacromial Breadth", min_value=0.0, format="%.2f")
chestcircumference = st.number_input("Chest Circumference", min_value=0.0, format="%.2f")
buttockcircumference = st.number_input("Buttock Circumference", min_value=0.0, format="%.2f")
waistcircumference = st.number_input("Waist Circumference", min_value=0.0, format="%.2f")
hipbreadth = st.number_input("Hip Breadth", min_value=0.0, format="%.2f")
forearmcircumferenceflexed = st.number_input("Forearm Circumference Flexed", min_value=0.0, format="%.2f")
wristcircumference = st.number_input("Wrist Circumference", min_value=0.0, format="%.2f")
calfcircumference = st.number_input("Calf Circumference", min_value=0.0, format="%.2f")
anklecircumference = st.number_input("Ankle Circumference", min_value=0.0, format="%.2f")
footlength = st.number_input("Foot Length", min_value=0.0, format="%.2f")
forearmhandlength = st.number_input("Forearm-Hand Length", min_value=0.0, format="%.2f")

measurements = {
    "stature": stature,
    "biacromialbreadth": biacromialbreadth,
    "chestcircumference": chestcircumference,
    "buttockcircumference": buttockcircumference,
    "waistcircumference": waistcircumference,
    "hipbreadth": hipbreadth,
    "forearmcircumferenceflexed": forearmcircumferenceflexed,
    "wristcircumference": wristcircumference,
    "calfcircumference": calfcircumference,
    "anklecircumference": anklecircumference,
    "footlength": footlength,
    "forearmhandlength": forearmhandlength
}

if st.button("Predict"):
    # Check if all inputs are provided
    if any(value == 0.0 for value in measurements.values()):
        st.error("Please provide all measurements.")
    else:
        # Calculate ratio features
        user_ratios = calculate_user_ratio_features(measurements)
        # Combine ratios with stature for final model input
        user_input = {**user_ratios, 'stature': measurements["stature"]}
        user_df = pd.DataFrame([user_input])
        # Make prediction
        passing_probability = model.predict_proba(user_df)[0][0] * 100  # Assuming class 0 is female
        st.success(f"Probability of passing as female: {passing_probability:.2f}%")
