import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px

# Load the trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

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

# Prepare and scale input data for prediction
input_data = np.array([[age, sex, bmi, smoker, region, children]])
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)

# Output prediction
st.subheader("Prediction")
st.write(f"The estimated insurance charge is: ${prediction[0]:.2f}")

# Create a graph for different values of age and charges based on various features
ages = np.linspace(18, 100, 100)
charges = model.predict(scaler.transform(np.column_stack([ages, np.zeros(100), np.ones(100)*25, np.zeros(100), np.zeros(100), np.ones(100)])))

# Create a DataFrame for Plotly graph
df = pd.DataFrame({
    'Age': ages,
    'Charges': charges
})

# Create the Plotly graph
fig = px.line(df, x="Age", y="Charges", title="Insurance Charges by Age")

# Display the Plotly chart
st.plotly_chart(fig)

# Add interactive graph for smoker status and region
smoker_values = [0, 1]
smoker_charges = [model.predict(scaler.transform(np.column_stack([ages, np.zeros(100), np.ones(100)*25, np.ones(100)*smoker, np.zeros(100), np.ones(100)]))) for smoker in smoker_values]

smoker_charges_df = pd.DataFrame({
    'Age': np.tile(ages, 2),
    'Charges': np.concatenate(smoker_charges),
    'Smoker': ['No']*100 + ['Yes']*100
})

fig_smoker = px.line(smoker_charges_df, x="Age", y="Charges", color="Smoker", title="Insurance Charges by Age and Smoker Status")
st.plotly_chart(fig_smoker)
