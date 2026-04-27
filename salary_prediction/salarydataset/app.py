import streamlit as st
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("random_forest_salary_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Employee Salary Prediction App")

st.write("Enter employee details to predict the salary")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=65, step=1)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

job_title = st.text_input(
    "Job Title (Example: Data Analyst, Software Engineer)"
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    step=1
)

# Prediction Button
if st.button("Predict Salary"):

    # Create dataframe
    new_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job_title],
        "Years of Experience": [experience]
    })

    # Categorical columns
    cat_cols = ['Gender', 'Education Level', 'Job Title']

    # Encode categorical features
    encoded = encoder.transform(new_data[cat_cols])

    encoded_cols = encoder.get_feature_names_out(cat_cols)

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols)

    # Combine numerical + encoded columns
    final_data = pd.concat([new_data.drop(columns=cat_cols), encoded_df], axis=1)

    # Scale data
    scaled_data = scaler.transform(final_data)

    # Predict salary
    prediction = model.predict(scaled_data)

    # Display result
    st.success(f"Estimated Salary: ₹ {prediction[0]:,.2f}")