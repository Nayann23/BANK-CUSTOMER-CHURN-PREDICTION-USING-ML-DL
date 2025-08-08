import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('churn_predict_model.pkl')
scaler = joblib.load('scaler.pkl')  # Remove this line if you didn’t save the scaler

st.title("🏦 Customer Churn Prediction App")

st.write("""
This app predicts whether a **bank customer is likely to leave** the bank, based on their profile.
Please fill in the details below:
""")

# User input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0.0, value=10000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)

has_credit_card = st.selectbox("Has Credit Card?", options=["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", options=["Yes", "No"])

estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
gender = st.selectbox("Gender", options=["Female", "Male"])

# Convert user-friendly inputs to numeric values used in model
has_credit_card = 1 if has_credit_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0
gender_male = 1 if gender == "Male" else 0

# Geography one-hot encoding (drop_first=True was used)
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
# France is the base case, so both will be 0

# Combine all inputs in the correct order
features = np.array([[credit_score, age, tenure, balance,
                      num_of_products, has_credit_card, is_active_member,
                      estimated_salary, geography_germany, geography_spain, gender_male]])

# Apply scaling
scaled_features = scaler.transform(features)  # Remove this line if not using scaler

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(scaled_features)  # Use `features` if not scaling
    probability = model.predict_proba(scaled_features)[0][1]

    if prediction[0] == 1:
        st.error(f"🚨 This customer is **likely to churn**.\n\nChurn Probability: **{probability:.2f}**")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\nChurn Probability: **{probability:.2f}**")
