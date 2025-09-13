# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Recommendation App", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation_model.pkl")
    return model

model = load_model()

# Define feature names in same order as training
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate conditions to get a recommended crop.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Create input DataFrame
input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

if st.button("🌱 Recommend Crop"):
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

    # Show top 3 probabilities if model supports it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        crops = model.classes_
        crop_probs = sorted(zip(crops, probs), key=lambda x: x[1], reverse=True)[:3]
        
        st.write("### 🔝 Top 3 Crop Predictions")
        for crop, prob in crop_probs:
            st.write(f"- {crop}: {prob:.2%}")
