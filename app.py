import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL AND FEATURES
# =========================================================

model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("📊 Customer Churn Prediction System")

st.write(
    "A Machine Learning application that predicts whether "
    "a customer is likely to leave the service."
)

st.divider()


# =========================================================
# DASHBOARD METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset", "Telco Customer Churn")
col2.metric("ML Model", "XGBoost")
col3.metric("Prediction", "Churn / No Churn")
col4.metric("Application", "Streamlit")

st.divider()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔮 Churn Prediction",
        "📈 EDA Analysis",
        "ℹ️ About Project"
    ]
)


# =========================================================
# TAB 1 - CHURN PREDICTION
# =========================================================

with tab1:

    st.header("🔮 Predict Customer Churn")

    st.write(
        "Enter the customer's information below."
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
            key="gender"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            key="senior_citizen"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            key="partner"
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
            key="dependents"
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12,
            key="tenure"
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            key="phone_service"
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ],
            key="multiple_lines"
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ],
            key="internet_service"
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="online_security"
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="online_backup"
        )

    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="device_protection"
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="tech_support"
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="streaming_tv"
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ],
            key="streaming_movies"
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key="contract"
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            key="paperless_billing"
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key="payment_method"
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=50.0,
            key="monthly_charges"
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=500.0,
            key="total_charges"
        )

    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Predict Churn",
        use_container_width=True,
        key="predict_button"
    )

    if predict_button:

        # Create customer dataframe

        customer = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        # Convert categorical columns into numerical columns

        customer = pd.get_dummies(
            customer,
            drop_first=True
        )

        # Make the input data have exactly the same
        # columns as the training data

        customer = customer.reindex(
            columns=features,
            fill_value=0
        )

        # Make prediction

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(
            customer
        )[0][1]

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.error(
                    "🔴 Customer is likely to churn"
                )

            else:

                st.success(
                    "🟢 Customer is unlikely to churn"
                )

        with result_col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )


# =========================================================
# TAB 2 - EDA ANALYSIS
# =========================================================

with tab2:

    st.header("📈 Exploratory Data Analysis")

    st.write(
        "These graphs show important patterns found in "
        "the customer churn dataset."
    )

    # -----------------------------------------------------
    # CHURN DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("1. Customer Churn Distribution")

    st.image(
        "plots/churn_distribution.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHURN BY CONTRACT
    # -----------------------------------------------------

    st.subheader("2. Churn by Contract Type")

    st.image(
        "plots/churn_by_contract.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHURN BY TENURE
    # -----------------------------------------------------

    st.subheader("3. Churn by Customer Tenure")

    st.image(
        "plots/churn_by_tenure.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHURN BY MONTHLY CHARGES
    # -----------------------------------------------------

    st.subheader("4. Churn by Monthly Charges")

    st.image(
        "plots/churn_by_monthly_charges.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHURN BY INTERNET SERVICE
    # -----------------------------------------------------

    st.subheader("5. Churn by Internet Service")

    st.image(
        "plots/churn_by_internet_service.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader("6. Top Features Affecting Churn")

    st.image(
        "plots/feature_importance.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader("7. Confusion Matrix")

    st.image(
        "plots/confusion_matrix.png",
        use_container_width=True
    )

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    st.subheader("8. ROC Curve")

    st.image(
        "plots/roc_curve.png",
        use_container_width=True
    )


# =========================================================
# TAB 3 - ABOUT PROJECT
# =========================================================

with tab3:

    st.header("ℹ️ About the Project")

    st.write(
        """
        ### Customer Churn Prediction Using Machine Learning

        This project predicts whether a customer is likely to
        leave a service based on their demographic and service
        information.

        ### Technologies Used

        - Python
        - Pandas
        - NumPy
        - Matplotlib
        - Seaborn
        - Scikit-learn
        - XGBoost
        - Joblib
        - Streamlit

        ### Machine Learning Models

        1. Logistic Regression
        2. Random Forest
        3. XGBoost

        ### Evaluation Metrics

        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC

        ### Main Features

        - Customer churn prediction
        - Churn probability
        - Exploratory Data Analysis
        - Confusion Matrix
        - ROC Curve
        - Feature Importance
        - Interactive Streamlit interface

        ### Project Goal

        The goal of this project is to identify customers
        who are likely to churn so that companies can take
        appropriate retention actions.
        """
    )

    st.success(
        "✅ Customer Churn Prediction System is ready!"
    )