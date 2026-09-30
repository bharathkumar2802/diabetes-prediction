import streamlit as st
import numpy as np
import tensorflow as tf
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retinopathy Prediction",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

model = tf.keras.models.load_model("model.keras")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Diabetic Retinopathy Prediction")

st.write(
    "Enter the patient information below "
    "to generate a prediction."
)

st.warning(
    "This application is for educational purposes "
    "and is not a medical diagnostic tool."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Patient Information")


age = st.number_input(
    "Age",
    min_value=1.0,
    max_value=120.0,
    value=50.0
)


systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=0.0,
    value=120.0
)


diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=0.0,
    value=80.0
)


cholesterol = st.number_input(
    "Cholesterol",
    min_value=0.0,
    value=200.0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict"):

    # Create input data
    input_data = np.array([
        [
            age,
            systolic_bp,
            diastolic_bp,
            cholesterol
        ]
    ])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    # Convert probability to class
    prediction = int(probability >= 0.5)

    # Convert class number to original label
    prediction_label = label_encoder.inverse_transform(
        [prediction]
    )[0]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("Prediction Result")

    st.write(
        "Predicted Class:"
    )

    st.success(
        prediction_label
    )

    st.write(
        f"Probability: {probability * 100:.2f}%"
    )


    # Progress bar
    st.progress(
        float(probability)
    )