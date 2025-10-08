import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime
import os

model = load_model('model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

csv_file = "user_predictions.csv"
if not os.path.exists(csv_file):
    df_init = pd.DataFrame(columns=[
        "Name","CreditScore","Age","Tenure","Balance","NumOfProducts",
        "HasCrCard","IsActiveMember","EstimatedSalary",
        "Geography","Gender","ChurnProbability","Prediction","Timestamp"
    ])
    df_init.to_csv(csv_file,index=False)

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide",        # Makes the page use the full width
    initial_sidebar_state="auto"
)

st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict whether they will churn or stay.")

def validate_name(name):
    return name.replace(" ","").isalpha()

name = st.text_input("Enter Your Name (alphabets only)", placeholder="e.g., John Doe")
if name and not validate_name(name):
    st.warning("❌ Name must contain only alphabetic characters.")

# Row 1: Credit Score, Age, Tenure
row1 = st.columns(3)
credit_score = row1[0].number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
age = row1[1].number_input("Age", min_value=18, max_value=100, value=35, step=1)
tenure = row1[2].number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5, step=1)

# Row 2: Balance, Num of Products, Estimated Salary
row2 = st.columns(3)
balance = row2[0].number_input("Account Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0, format="%.2f")
num_of_products = row2[1].number_input("Number of Bank Products", min_value=1, max_value=4, value=2, step=1)
estimated_salary = row2[2].number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0, format="%.2f")

# Row 3: Has Credit Card, Is Active Member, Gender
row3 = st.columns(3)
has_cr_card = row3[0].selectbox("Has Credit Card?", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
is_active_member = row3[1].selectbox("Is Active Member?", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
gender = row3[2].selectbox("Gender", options=["Male","Female"])
gender_male = 1 if gender=="Male" else 0

# Row 4: Geography
row4 = st.columns(3)
geography = row4[0].selectbox("Geography", options=["France","Germany","Spain"])
geography_france = 1 if geography=="France" else 0
geography_germany = 1 if geography=="Germany" else 0
geography_spain = 1 if geography=="Spain" else 0

if st.button("🔍 Predict Churn"):
    if not name or not validate_name(name):
        st.error("Please enter a valid name (alphabets only).")
    else:
        user_input = np.array([[credit_score, age, tenure, balance, num_of_products,
                                has_cr_card, is_active_member, estimated_salary,
                                geography_germany, geography_spain, gender_male]])
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)[0][0]
        prediction_label = "Churn" if prediction >= 0.5 else "Stay"

        if prediction >= 0.5:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
        st.markdown(f"**Churn Probability:** {prediction:.2%}")

        new_data = pd.DataFrame([[name, credit_score, age, tenure, balance, num_of_products,
                                  has_cr_card, is_active_member, estimated_salary,
                                  geography, gender, round(float(prediction),4), prediction_label,
                                  datetime.now().strftime('%Y-%m-%d %H:%M:%S')]],
                                columns=["Name","CreditScore","Age","Tenure","Balance",
                                         "NumOfProducts","HasCrCard","IsActiveMember",
                                         "EstimatedSalary","Geography","Gender",
                                         "ChurnProbability","Prediction","Timestamp"])
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, new_data], ignore_index=True)
        df_combined.drop_duplicates(subset=["Name","Timestamp"], keep='last').to_csv(csv_file,index=False)
        st.success(f"📬 Thank you **{name}**, your response has been recorded on {new_data['Timestamp'][0]}.")

