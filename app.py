import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model_path = "model.joblib"
scaler_path = "scaler.joblib"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Streamlit app UI
st.title("Insurance Cost Prediction")

st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10, 50, 25)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
children = st.sidebar.slider("Children", 0, 10, 1)

# Convert categorical features to numerical values
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
region = region_map[region]

# Prepare the input data for prediction
input_data = np.array([[age, sex, bmi, smoker, region, children]])

# Scale the input data using the saved scaler
scaled_data = scaler.transform(input_data)

# Make prediction
prediction = model.predict(scaled_data)

# Show the prediction result
st.subheader("Prediction")
st.write(f"The estimated insurance charge is: ${prediction[0]:.2f}")

if __name__ == "__main__":
    st.write("Use the sidebar to input values and predict the insurance charge.")
