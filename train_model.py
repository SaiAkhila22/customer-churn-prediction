import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
df.dropna(inplace=True)

# Convert Churn into numbers
# No = 0
# Yes = 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Remove customerID because it is not useful for prediction
df.drop("customerID", axis=1, inplace=True)

# Display cleaned data
print("Cleaned Dataset:")
print(df.head())

# Display new dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Exploratory Data Analysis
# -------------------------------

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.savefig("plots/churn_distribution.png")
plt.show()


# 2. Churn by Contract
plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.savefig("plots/churn_by_contract.png")
plt.show()


# 3. Churn by Tenure
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Churn by Customer Tenure")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure (Months)")
plt.savefig("plots/churn_by_tenure.png")
plt.show()


# 4. Churn by Monthly Charges
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Churn by Monthly Charges")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")
plt.savefig("plots/churn_by_monthly_charges.png")
plt.show()


# 5. Churn by Internet Service
plt.figure(figsize=(8, 5))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.savefig("plots/churn_by_internet_service.png")
plt.show()

print("\nEDA graphs created successfully!")

# -------------------------------
# Step 4: Prepare Data for ML
# -------------------------------

# Separate input features (X) and target (y)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

print("\nFeatures after encoding:")
print(X.head())

print("\nNumber of features:")
print(X.shape[1])

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# -------------------------------
# Step 6: Logistic Regression
# -------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scale the data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = logistic_model.predict(X_test_scaled)

# Prediction probabilities
y_prob_lr = logistic_model.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression model trained successfully!")

print("First 10 predictions:")
print(y_pred_lr[:10])

# -------------------------------
# Step 7: Evaluate Logistic Regression
# -------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Calculate metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_prob_lr)

print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_lr)
print("Precision:", precision_lr)
print("Recall   :", recall_lr)
print("F1 Score :", f1_lr)
print("ROC-AUC  :", roc_auc_lr)

# -------------------------------
# Step 8: Random Forest
# -------------------------------

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Prediction probabilities
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest model trained successfully!")

print("First 10 predictions:")
print(y_pred_rf[:10])

# -------------------------------
# Step 9: Evaluate Random Forest
# -------------------------------

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n--- Random Forest Results ---")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1 Score :", f1_rf)
print("ROC-AUC  :", roc_auc_rf)


# Compare both models
print("\n--- Model Comparison ---")

print("\nLogistic Regression")
print("Accuracy :", accuracy_lr)
print("Precision:", precision_lr)
print("Recall   :", recall_lr)
print("F1 Score :", f1_lr)
print("ROC-AUC  :", roc_auc_lr)

print("\nRandom Forest")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1 Score :", f1_rf)
print("ROC-AUC  :", roc_auc_rf)

# -------------------------------
# Step 10: XGBoost
# -------------------------------

from xgboost import XGBClassifier

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Prediction probabilities
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost model trained successfully!")

print("First 10 predictions:")
print(y_pred_xgb[:10])

# -------------------------------
# Step 11: Evaluate XGBoost
# -------------------------------

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_prob_xgb)

print("\n--- XGBoost Results ---")
print("Accuracy :", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall   :", recall_xgb)
print("F1 Score :", f1_xgb)
print("ROC-AUC  :", roc_auc_xgb)


# -------------------------------
# Compare All 3 Models
# -------------------------------

print("\n================ MODEL COMPARISON ================")

print("\nLogistic Regression")
print("Accuracy :", accuracy_lr)
print("Precision:", precision_lr)
print("Recall   :", recall_lr)
print("F1 Score :", f1_lr)
print("ROC-AUC  :", roc_auc_lr)

print("\nRandom Forest")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1 Score :", f1_rf)
print("ROC-AUC  :", roc_auc_rf)

print("\nXGBoost")
print("Accuracy :", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall   :", recall_xgb)
print("F1 Score :", f1_xgb)
print("ROC-AUC  :", roc_auc_xgb)

# -------------------------------
# Step 12: Confusion Matrix
# -------------------------------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Create confusion matrix for XGBoost
cm = confusion_matrix(y_test, y_pred_xgb)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()
plt.title("XGBoost Confusion Matrix")
plt.savefig("plots/confusion_matrix.png")
plt.show()


# -------------------------------
# ROC Curve
# -------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc_xgb:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()

plt.savefig("plots/roc_curve.png")
plt.show()

print("\nConfusion Matrix and ROC Curve created successfully!")

# -------------------------------
# Step 13: Feature Importance
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance from XGBoost
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top 10 features
print("\n--- Top 10 Important Features ---")
print(feature_importance.head(10))


# Create graph
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"].head(10)[::-1],
    feature_importance["Importance"].head(10)[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Affecting Customer Churn")

plt.tight_layout()

plt.savefig("plots/feature_importance.png")
plt.show()

print("\nFeature importance graph created successfully!")

# -------------------------------
# Step 14: Save Model
# -------------------------------

import joblib

# Save the XGBoost model
joblib.dump(
    xgb_model,
    "models/churn_model.pkl"
)

# Save the feature names
joblib.dump(
    X.columns.tolist(),
    "models/features.pkl"
)

print("\nModel saved successfully!")
print("Saved file: models/churn_model.pkl")

print("Features saved successfully!")
print("Saved file: models/features.pkl")