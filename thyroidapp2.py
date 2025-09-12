import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('thyroid_model.pkl')
scaler = joblib.load('thyroid_scaler.pkl')

st.title("🧠 Thyroid Disease Classifier")

st.write("Enter the following values to predict the thyroid condition:")

# Collect user inputs (Attribute6 dropped)
attr1 = st.number_input("Attribute 1", value=0.0)
attr2 = st.number_input("Attribute 2", value=0.0)
attr3 = st.number_input("Attribute 3", value=0.0)
attr4 = st.number_input("Attribute 4", value=0.0)
attr5 = st.number_input("Attribute 5", value=0.0)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[attr1, attr2, attr3, attr4, attr5]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"✅ Predicted Thyroid Class: {prediction}")
