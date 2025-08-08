# 🏦 Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to **churn or stay**, using a machine learning model built with TensorFlow/Keras and deployed with a **Streamlit** user interface.

---

## 📂 Project Structure

```
📁 root/
├── churn_model.ipynb           # Model training notebook
├── model.keras                 # Trained model
├── scaler.pkl                  # Scaler used during preprocessing
├── app.py                      # Streamlit app
├── users.csv                   # Stored user predictions
└── README.md                   # Documentation (this file)
```

---

## 📊 1. Model Training (`churn_model.ipynb`)

### ✅ Features Used:
- Credit Score
- Age
- Tenure
- Balance
- Number of Bank Products
- Has Credit Card?
- Is Active Member?
- Estimated Salary
- Geography (France, Germany, Spain)
- Gender (Male/Female)

### 🧠 Model Architecture:
- Input Layer
- Dense(16, ReLU) → Dropout
- Dense(8, ReLU)
- Output Layer: Dense(1, Sigmoid)

### ⚙️ Tools & Techniques:
- Data preprocessing (Label Encoding, One-Hot)
- Scaling with StandardScaler
- Binary Crossentropy Loss
- Optimizer: Adam
- Epochs: 100

### 🧾 Output Files:
- `model.keras` — Saved Keras model
- `scaler.pkl` — Saved StandardScaler for consistent input scaling

---

## 🌐 2. Streamlit App (`app.py`)

### 🎯 Purpose:
An interactive frontend where users input customer data and get churn predictions.

### ✨ Features:
- Validated inputs with placeholders (e.g., no names in numeric fields)
- Predictions with churn probability
- Result saved along with name and timestamp
- CSV-based user history
- Dynamic EDA-style visualizations on user data:
  - 📊 Pie chart (Churn vs Stay)
  - 📦 Box plot (Estimated Salary vs Prediction)
  - 🎻 Violin plot (Age vs Prediction)

---

## 📁 CSV Output (`user_predictions.csv`)

Each time a prediction is made, a new row is appended:

| Name         | CreditScore | Age | Tenure | ... | ChurnProbability | Prediction | Timestamp           |
|--------------|--------------|-----|--------|-----|------------------|------------|----------------------|
| Nayan Darokar| 700.0        | 24  | 5      | ... | 0.0187           | Stay       | 2025-08-08 12:48:47  |

---

## 🚀 How to Run

### 🔧 Requirements:
Install packages using:

```bash
pip install streamlit tensorflow scikit-learn pandas matplotlib seaborn
```

### ▶️ Start App:
```bash
streamlit run app.py
```

---

## 💡 Future Enhancements
- Authentication for user access
- Admin dashboard
- More complex model like XGBoost or ensemble stacking

---

## 👨‍💻 Author
Made with ❤️ by Nayan
