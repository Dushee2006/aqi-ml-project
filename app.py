import streamlit as st
import joblib
import numpy as np

# Load trained model
rf_model = joblib.load("models/random_forest_model.pkl")

st.title("🌍 AQI Prediction App")

st.write("Enter pollutant values to predict AQI category.")

# Pollutant Inputs
pm2_5 = st.number_input("PM2.5", min_value=0.0)
pm10 = st.number_input("PM10", min_value=0.0)
co = st.number_input("CO", min_value=0.0)
no2 = st.number_input("NO2", min_value=0.0)
o3 = st.number_input("O3", min_value=0.0)
so2 = st.number_input("SO2", min_value=0.0)

# Time Inputs
hour = st.slider("Hour of Day", 0, 23)
day = st.slider("Day of Month", 1, 31)
month = st.slider("Month", 1, 12)
day_of_week = st.slider("Day of Week (0=Monday)", 0, 6)

if st.button("Predict AQI"):

    input_data = np.array([[pm2_5, pm10, co, no2, o3, so2,
                            hour, day, month, day_of_week]])

    prediction = rf_model.predict(input_data)[0]

    aqi_labels = {
        1: "Good 🟢",
        2: "Fair 🟡",
        3: "Moderate 🟠",
        4: "Poor 🔴"
    }

    st.success(f"Predicted AQI Category: {aqi_labels.get(prediction)}")