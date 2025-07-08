import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_main


def main():
    st.title("🎯 Customer Churn Prediction App")
    
    # Sidebar for description
    with st.sidebar:
        st.markdown("### About This App")
        st.markdown("""
        This app predicts whether a customer is likely to churn (leave the service) based on their characteristics and usage patterns.
        
        **How it works:**
        1. Enter customer information in the form
        2. Click 'Predict Churn' 
        3. Get instant prediction results
        
        The model uses machine learning to analyze patterns and predict customer behavior.
        """)
        
        st.markdown("### Model Performance")
        st.info("Current Model Accuracy: ~80.7%")
        
        st.markdown("### Feature Descriptions")
        with st.expander("View all features"):
            st.markdown("""
            - **SeniorCitizen**: Is the customer a senior citizen?
            - **Tenure**: Number of months with the company
            - **MonthlyCharges**: Monthly billing amount
            - **TotalCharges**: Total amount charged
            - **Gender**: Customer's gender
            - **Partner**: Does customer have a partner?
            - **Dependents**: Does customer have dependents?
            - **PhoneService**: Does customer have phone service?
            - **MultipleLines**: Does customer have multiple phone lines?
            - **InternetService**: Type of internet service
            - **OnlineSecurity**: Online security subscription
            - **OnlineBackup**: Online backup subscription
            - **DeviceProtection**: Device protection subscription
            - **TechSupport**: Technical support subscription
            - **StreamingTV**: TV streaming subscription
            - **StreamingMovies**: Movie streaming subscription
            - **Contract**: Contract duration
            - **PaperlessBilling**: Paperless billing preference
            - **PaymentMethod**: Payment method used
            """)

    # Main content area
    st.markdown("### 📝 Customer Information")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        gender = st.radio("Gender:", ["male", "female"])
        SeniorCitizen = st.checkbox("Senior Citizen")
        Partner = st.checkbox("Has Partner")
        Dependents = st.checkbox("Has Dependents")
        
        st.markdown("#### Service Information")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=1.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0, step=10.0)
        
        PhoneService = st.checkbox("Phone Service")
        MultipleLines = st.checkbox("Multiple Lines") if PhoneService else False
        
        InternetService = st.checkbox("Internet Service")
        
    with col2:
        st.markdown("#### Internet Services")
        if InternetService:
            OnlineSecurity = st.checkbox("Online Security")
            OnlineBackup = st.checkbox("Online Backup")
            DeviceProtection = st.checkbox("Device Protection")
            TechSupport = st.checkbox("Tech Support")
            StreamingTV = st.checkbox("Streaming TV")
            StreamingMovies = st.checkbox("Streaming Movies")
        else:
            OnlineSecurity = OnlineBackup = DeviceProtection = TechSupport = StreamingTV = StreamingMovies = False
        
        st.markdown("#### Billing & Contract")
        contract_options = {
            0: "Month-to-month",
            1: "One year", 
            2: "Two year"
        }
        Contract = st.selectbox("Contract Duration:", options=list(contract_options.keys()), 
                               format_func=lambda x: contract_options[x])
        
        PaperlessBilling = st.checkbox("Paperless Billing")
        
        payment_options = {
            0: "Credit card (automatic)",
            1: "Bank transfer (automatic)", 
            2: "Electronic check",
            3: "Mailed check"
        }
        PaymentMethod = st.selectbox("Payment Method:", options=list(payment_options.keys()),
                                    format_func=lambda x: payment_options[x])

    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

    if predict_button:
        with st.spinner("Analyzing customer data..."):
            try:
                # Load the prediction service
                service = prediction_service_loader(
                    pipeline_name="continuous_deployment_pipeline",
                    pipeline_step_name="mlflow_model_deployer_step",
                    running=False,
                )
                
                if service is None:
                    st.error("No prediction service found. Please run the deployment pipeline first.")
                    if st.button("Deploy Model"):
                        run_main()
                    return
                
                # Prepare the data
                data_point = {
                    'SeniorCitizen': int(SeniorCitizen),
                    'tenure': tenure, 
                    'MonthlyCharges': MonthlyCharges, 
                    'TotalCharges': TotalCharges,
                    'gender': 1 if gender == "male" else 0,
                    'Partner': int(Partner),
                    'Dependents': int(Dependents),
                    'PhoneService': int(PhoneService),
                    'MultipleLines': int(MultipleLines), 
                    'InternetService': int(InternetService),
                    'OnlineSecurity': int(OnlineSecurity),
                    'OnlineBackup': int(OnlineBackup),
                    'DeviceProtection': int(DeviceProtection),
                    'TechSupport': int(TechSupport),
                    'StreamingTV': int(StreamingTV),
                    'StreamingMovies': int(StreamingMovies),
                    'Contract': int(Contract), 
                    'PaperlessBilling': int(PaperlessBilling),
                    'PaymentMethod': int(PaymentMethod)
                }

                # Convert to DataFrame
                data_point_df = pd.DataFrame([data_point])
                
                # Make prediction
                prediction = service.predict(data_point_df)
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    if prediction[0] == 1:
                        st.error("🚨 **HIGH RISK: Customer likely to churn**")
                        st.markdown("**Recommendation:** Consider retention strategies")
                    else:
                        st.success("✅ **LOW RISK: Customer likely to stay**")
                        st.markdown("**Recommendation:** Continue current service")
                
                with col2:
                    st.metric("Churn Probability", f"{prediction[0] * 100:.1f}%")
                
                # Show input summary
                with st.expander("📋 Customer Data Summary"):
                    st.json(data_point)
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                logging.error(e)

if __name__ == "__main__":
    main()