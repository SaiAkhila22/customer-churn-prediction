# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a service.

The project uses customer demographic, service, contract, and billing information to identify customers who are at risk of churn.

An interactive Streamlit web application is developed where users can enter customer details and receive a real-time churn prediction.

---

## 🎯 Objectives

- Predict whether a customer is likely to churn.
- Analyze customer churn patterns using Exploratory Data Analysis.
- Compare different Machine Learning classification models.
- Identify important factors affecting customer churn.
- Build an interactive web application for real-time prediction.

---

## 📂 Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming Services
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## 🔍 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Encoding
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Random Forest
   ↓
XGBoost
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Save Best Model
   ↓
Streamlit Web Application
   ↓
Customer Churn Prediction