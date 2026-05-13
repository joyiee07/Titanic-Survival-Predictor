import streamlit as st
import joblib
import numpy as np


model = joblib.load("titanic_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered"
)

# Load trained model
model = joblib.load("titanic_model.pkl")

# Title
st.title("🚢 Titanic Survival Prediction System")

st.write(
    "This machine learning app predicts whether a passenger "
    "would survive the Titanic disaster using a Random Forest Classifier."
)

st.header("Enter Passenger Information")

# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

# Convert sex to numeric
sex_value = 0 if sex == "Male" else 1

age = st.slider(
    "Age",
    1,
    80,
    25
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=50.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

# Convert embarked to numeric
embarked_value = {
    "S": 0,
    "C": 1,
    "Q": 2
}[embarked]

# Prediction
if st.button("Predict Survival"):

    try:
        # Arrange input exactly like training features
        input_data = np.array([[
            pclass,
            sex_value,
            age,
            fare,
            embarked_value
        ]])

        # Predict
        prediction = model.predict(input_data)

        # Result
        if prediction[0] == 1:
            st.success("✅ The passenger is predicted to SURVIVE.")
        else:
            st.error("❌ The passenger is predicted NOT to survive.")

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.write("---")
st.write("Artificial Intelligence 5.0")
st.write("Machine Learning Classification Project")