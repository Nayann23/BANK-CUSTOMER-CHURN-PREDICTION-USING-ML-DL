# README.md

# 🏦 Customer Churn Prediction App

## Overview
This is a simple web-based application built using **Streamlit** that predicts whether a bank customer is likely to churn (leave the bank). The prediction is based on customer attributes like age, credit score, balance, salary, etc. The app uses a pre-trained machine learning model (`churn_predict_model.pkl`) and a data scaler (`scaler.pkl`) to make accurate predictions.

## Features
- Interactive form for inputting customer data
- One-click churn prediction
- Displays both the churn result and the churn probability
- Encodes categorical variables and scales inputs for compatibility with the trained model

## Technologies Used
- **Python**
- **Streamlit** (for web interface)
- **Scikit-learn** (used during model creation)
- **Joblib** (for loading serialized models and scalers)
- **NumPy** (for numerical array management)

## Installation

1. **Clone the repository:**
   ```bash
   git clone (https://github.com/Nayann23/BANK-CUSTOMER-CHURN-PREDICTION-USING-ML-DL.git)
   cd churn-prediction-app
   ```

2. **Install dependencies:**
   Make sure you have Python installed (>= 3.7), then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist:**
   Place the following files in the project directory:
   - `churn_predict_model.pkl` (your pre-trained model)
   - `scaler.pkl` (your scaler, optional if not using scaling)

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Fill in the form:**
   Enter customer data such as:
   - Credit Score
   - Age
   - Tenure
   - Account Balance
   - Number of Products
   - Credit Card status
   - Active Membership status
   - Estimated Salary
   - Geography
   - Gender

3. **Predict Churn:**
   Click the **Predict Churn** button to receive a churn prediction and associated probability.

## Input Encoding Details

- **Gender:** Male = 1, Female = 0
- **Has Credit Card:** Yes = 1, No = 0
- **Is Active Member:** Yes = 1, No = 0
- **Geography:**
  - Germany = `[1, 0]`
  - Spain = `[0, 1]`
  - France = `[0, 0]` (base case)

## Notes

- The scaler is applied to normalize the feature values before prediction. If your model was trained without scaling, remove the scaler-related lines.
- Categorical variables are one-hot encoded to match the training model.

## Author

**Nayan**

## Support

If you found this project valuable, consider giving it a ⭐️ on the repository. It helps others discover the project and motivates further development!

---
