import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained model and scaler
try:
    model = joblib.load('rfc_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the notebook to train and save them first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Define the application title
st.title("Customer Churn Prediction")

# Create input fields for user data
st.sidebar.header("Customer Details:")

credit_score = st.sidebar.number_input("Credit Score", min_value=0, max_value=1000, value=650)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=20, value=5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=0.0, format="%.2f")
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.sidebar.checkbox("Has Credit Card?", value=True)
is_active_member = st.sidebar.checkbox("Is Active Member?", value=True)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

# Predict button
if st.sidebar.button("Predict Churn"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card else 0],
        'IsActiveMember': [1 if is_active_member else 0],
        'EstimatedSalary': [estimated_salary]
    })

    # --- Preprocessing ---
    # Label Encoding for Geography and Gender
    # IMPORTANT: Ensure the encoding matches what was used in the notebook
    # For Geography: France:0, Germany:1, Spain:2 (as per common alphabetical order if not specified)
    # For Gender: Female:0, Male:1 (as per common alphabetical order if not specified)
    # It's safer to fit LabelEncoders on the known categories from the notebook
    # and save/load them, or hardcode the mapping if it's simple and stable.

    # Assuming the notebook used alphabetical order for LabelEncoders:
    # Geography: France -> 0, Germany -> 1, Spain -> 2
    # Gender: Female -> 0, Male -> 1
    # The notebook actually did: France:0, Spain:2, Germany:1 (from the unique values [0 2 1])
    # And Female:0, Male:1 (from unique values [0 1])

    # Correcting the mapping based on notebook's output [0 2 1] for ['France', 'Spain', 'Germany']
    # France: 0, Spain: 2, Germany: 1
    geography_mapping = {"France": 0, "Spain": 2, "Germany": 1}
    input_data['Geography'] = input_data['Geography'].map(geography_mapping)


    gender_encoder = LabelEncoder()
    gender_encoder.fit(["Female", "Male"]) # Fit in alphabetical order based on typical notebook output
    input_data['Gender'] = gender_encoder.transform(input_data['Gender'])


    # Scaling numerical features
    numerical_cols = ['CreditScore', 'Balance', 'EstimatedSalary']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    # --- Prediction ---
    try:
        prediction_proba = model.predict_proba(input_data)[:, 1] # Probability of churn
        prediction = model.predict(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.warning(f"Customer is LIKELY to churn (Probability: {prediction_proba[0]:.2f})")
        else:
            st.success(f"Customer is UNLIKELY to churn (Probability: {prediction_proba[0]:.2f})")
            
        # Log prediction (optional, but good practice)
        # For simplicity, not writing to prediction_logs.csv in this subtask
        # to avoid file access conflicts if the notebook is also running.

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.info("Adjust the customer details in the sidebar and click 'Predict Churn'.")
