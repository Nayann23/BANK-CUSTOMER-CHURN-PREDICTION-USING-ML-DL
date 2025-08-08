import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load model and scaler
# ----------------------------
model = load_model('model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ----------------------------
# CSV file setup
# ----------------------------
csv_file = "user_predictions.csv"
if not os.path.exists(csv_file):
    df_init = pd.DataFrame(columns=[
        "Name", "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography", "Gender", "ChurnProbability", "Prediction", "Timestamp"
    ])
    df_init.to_csv(csv_file, index=False)

# ----------------------------
# UI Title
# ----------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict whether they will churn or stay.")

# ----------------------------
# Helper Functions
# ----------------------------
def get_numeric_input(label, min_value, max_value, placeholder):
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        step=1.0,
        format="%.2f",
        placeholder=placeholder
    )

def validate_name(name):
    return name.replace(" ", "").isalpha()

# ----------------------------
# Name Input
# ----------------------------
name = st.text_input("Enter Your Name (alphabets only)", placeholder="e.g., John Doe")
if name and not validate_name(name):
    st.warning("❌ Name must contain only alphabetic characters.")

# ----------------------------
# Input Fields
# ----------------------------
credit_score = get_numeric_input("Credit Score", 300.0, 900.0, "e.g., 600")
age = get_numeric_input("Age", 18.0, 100.0, "e.g., 35")
tenure = get_numeric_input("Tenure (Years with Bank)", 0.0, 20.0, "e.g., 5")
balance = get_numeric_input("Account Balance", 0.0, 250000.0, "e.g., 50000")
num_of_products = get_numeric_input("Number of Bank Products", 1.0, 4.0, "e.g., 2")
has_cr_card = st.selectbox("Has Credit Card?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.selectbox("Is Active Member?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = get_numeric_input("Estimated Salary", 0.0, 300000.0, "e.g., 75000")

# Gender Encoding
gender = st.selectbox("Gender", options=["Male", "Female"])
gender_male = 1 if gender == "Male" else 0

# Geography Encoding
geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
geography_france = 1 if geography == "France" else 0
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0


# ----------------------------
# Predict Button
# ----------------------------
if st.button("🔍 Predict Churn"):

    if not name or not validate_name(name):
        st.error("Please enter a valid name (alphabets only) before proceeding.")
    else:
        user_input = np.array([[credit_score, age, tenure, balance, num_of_products,
                                has_cr_card, is_active_member, estimated_salary,
                                geography_germany, geography_spain, gender_male]])
        
        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)[0][0]
        prediction_label = "Churn" if prediction >= 0.5 else "Stay"

        # Display result
        if prediction >= 0.5:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
        st.markdown(f"**Churn Probability:** {prediction:.2%}")

        # Save result
        new_data = pd.DataFrame([[
            name, credit_score, age, tenure, balance, num_of_products,
            has_cr_card, is_active_member, estimated_salary,
            geography, gender, round(float(prediction), 4), prediction_label,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]], columns=[
            "Name", "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography", "Gender", "ChurnProbability", "Prediction", "Timestamp"
        ])

        # Append and deduplicate based on Name+Timestamp
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, new_data], ignore_index=True)
        df_combined.drop_duplicates(subset=["Name", "Timestamp"], keep='last').to_csv(csv_file, index=False)

        # ✅ Confirmation Message
        st.success(f"📬 Thank you **{name}**, your response has been recorded successfully on {new_data['Timestamp'][0]}.")



# ----------------------------
# Generate Clean Aggregate Graphs (Dynamic EDA)
# ----------------------------
st.markdown("---")
if st.button("📊 Generate User Statistics Charts"):
    df = pd.read_csv(csv_file)

    if df.empty:
        st.warning("No user data available to generate charts.")
    else:
        st.markdown("### 📈 Overall Customer Trends (All Users)")

        col1, col2 = st.columns(2)

        # 1️⃣ PIE CHART: Churn vs Stay Distribution
        with col1:
            fig1, ax1 = plt.subplots()
            churn_counts = df['Prediction'].value_counts()
            ax1.pie(
                churn_counts, 
                labels=churn_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=["skyblue", "tomato"]
            )
            ax1.set_title("Churn vs Stay")
            st.pyplot(fig1)

        # 2️⃣ BOX PLOT: Estimated Salary vs Prediction
        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Prediction', y='EstimatedSalary', data=df, ax=ax2, palette="Set2")
            ax2.set_title("Estimated Salary vs Prediction")
            st.pyplot(fig2)

        # 3️⃣ VIOLIN PLOT: Age vs Prediction
        st.markdown("### 🎻 Age Distribution vs Prediction")
        fig3, ax3 = plt.subplots()
        sns.violinplot(x='Prediction', y='Age', data=df, ax=ax3, palette="pastel")
        ax3.set_title("Age vs Churn/Stay")
        st.pyplot(fig3)

        # 4️⃣ HISTOGRAM: Balance Distribution (Optional EDA)
        st.markdown("### 💰 Balance Distribution (All Users)")
        fig4, ax4 = plt.subplots()
        sns.histplot(data=df, x='Balance', hue='Prediction', bins=15, kde=True, ax=ax4, palette='coolwarm')
        ax4.set_title("Balance Distribution by Prediction")
        st.pyplot(fig4)
