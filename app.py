
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="centered"
)

# Load trained model
model = joblib.load("cardio_model.pkl")

st.title("❤️ CardioPredict")
st.subheader("Cardiovascular Disease Prediction")

st.write(
    "An educational machine-learning project using "
    "health-related and lifestyle data."
)

st.warning(
    "Educational use only. This experimental model "
    "cannot diagnose disease or determine personal medical risk."
)

st.divider()
st.header("Enter Sample Details")

age = st.number_input(
    "Age (years)", min_value=18, max_value=100, value=50
)

gender = st.selectbox("Gender", ["Female", "Male"])

height = st.number_input(
    "Height (cm)", min_value=100, max_value=250, value=165
)

weight = st.number_input(
    "Weight (kg)", min_value=20, max_value=300, value=65
)

st.subheader("Blood Pressure")

ap_hi = st.number_input(
    "Systolic blood pressure", min_value=1, max_value=300, value=120
)

ap_lo = st.number_input(
    "Diastolic blood pressure", min_value=1, max_value=250, value=80
)

st.subheader("Other Features")

cholesterol = st.selectbox("Cholesterol level", [1, 2, 3])
gluc = st.selectbox("Glucose level", [1, 2, 3])
smoke = st.selectbox("Smoking", ["No", "Yes"])
alco = st.selectbox("Alcohol consumption", ["No", "Yes"])
active = st.selectbox("Physically active", ["Yes", "No"])

if st.button("Run Experimental Prediction", type="primary"):

    if ap_hi < ap_lo:
        st.error("Please check the blood pressure values.")
    else:
        input_data = pd.DataFrame([{
            "age": int(age * 365.25),
            "gender": 1 if gender == "Female" else 2,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Model Output")

        if prediction == 1:
            st.write(
                "The model classified this input as "
                "'cardiovascular disease recorded' in the dataset."
            )
        else:
            st.write(
                "The model classified this input as "
                "'no cardiovascular disease recorded' in the dataset."
            )

        st.caption(
            "This is an experimental classification, not a medical diagnosis."
        )

st.divider()
st.caption("Student project | Python, Pandas, Scikit-learn and Streamlit")
