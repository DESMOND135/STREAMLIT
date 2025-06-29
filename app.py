import streamlit as st
import cloudpickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open("regmodel.pkl", "rb") as f:
    model = cloudpickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = cloudpickle.load(f)

# Streamlit app UI
st.title("Insurance Cost Prediction")

st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10, 50, 25)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
children = st.sidebar.slider("Children", 0, 10, 1)

# Convert categorical features
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
region = region_map[region]

# Prepare and scale input
input_data = np.array([[age, sex, bmi, smoker, region, children]])
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)

# Output
st.subheader("Prediction")
st.write(f"The estimated insurance charge is: ${prediction[0]:.2f}")
