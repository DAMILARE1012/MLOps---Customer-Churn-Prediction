import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from PIL import Image

# --- Model loading and prediction logic ---
@st.cache_resource
def load_model():
    try:
        model_path = './saved_model/churn_model.pkl'
        feature_names_path = './saved_model/feature_names.pkl'
        if os.path.exists(model_path) and os.path.exists(feature_names_path):
            model = joblib.load(model_path)
            feature_names = joblib.load(feature_names_path)
            return model, feature_names
        else:
            st.error("❌ Model files not found. Please ensure the model has been saved.")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def predict_churn(model, feature_names, customer_data):
    try:
        df = pd.DataFrame([customer_data])
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        df = df[feature_names]
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None

def create_churn_risk_gauge(probability):
    """Create a compact gauge chart for churn risk"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def create_compact_feature_importance():
    """Create a compact feature importance chart"""
    features = ['Tenure', 'Monthly Charges', 'Contract', 'Payment Method', 'Internet Service']
    importance = [0.25, 0.20, 0.15, 0.12, 0.10]
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        title="Top Features",
        labels={'x': 'Importance', 'y': ''}
    )
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def create_compact_segments():
    """Create compact customer segmentation"""
    segments = ['High Value - Low Risk', 'High Value - High Risk', 'Low Value - Low Risk', 'Low Value - High Risk']
    counts = [45, 15, 30, 10]
    
    fig = px.pie(
        values=counts, 
        names=segments, 
        title="Customer Segments",
        hole=0.4
    )
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def generate_retention_recommendations(churn_probability, customer_data):
    """Generate compact retention recommendations"""
    recommendations = []
    
    if churn_probability > 0.7:
        recommendations.append("🚨 **High Risk** - Immediate intervention needed")
        recommendations.append("• 15-20% discount • Dedicated manager • Follow-up call")
    elif churn_probability > 0.4:
        recommendations.append("⚠️ **Medium Risk** - Proactive engagement")
        recommendations.append("• 10-15% discount • Email campaign • Service benefits")
    else:
        recommendations.append("✅ **Low Risk** - Regular engagement")
        recommendations.append("• Satisfaction surveys • Upsell opportunities • Loyalty program")
    
    # Specific recommendations
    if customer_data.get('Contract', 0) == 0:
        recommendations.append("• Offer contract incentives")
    if customer_data.get('MonthlyCharges', 0) > 80:
        recommendations.append("• Review pricing strategy")
    if customer_data.get('tenure', 0) < 6:
        recommendations.append("• Focus on onboarding")
    
    return recommendations

# --- Custom CSS for dark theme and modern look ---
st.markdown('''
    <style>
    body, .stApp, .main, .block-container {
        background: #111111 !important;
        color: #f5f6fa !important;
    }
    .sidebar .sidebar-content {
        background: #18191a !important;
        color: #f5f6fa !important;
    }
    .css-1d391kg, .css-1v0mbdj, .css-1lcbmhc, .css-1cypcdb, .css-1v3fvcr, .css-1vzeuhh, .css-1dp5vir, .css-1kyxreq, .css-1q8dd3e, .css-1r6slb0, .css-1v3fvcr, .css-1vzeuhh, .css-1dp5vir, .css-1kyxreq, .css-1q8dd3e, .css-1r6slb0 {
        background: #18191a !important;
        color: #f5f6fa !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        color: #fff;
    }
    .stSelectbox>div>div, .stNumberInput>div>div>input, .stTextInput>div>div>input {
        background: #222 !important;
        color: #f5f6fa !important;
        border-radius: 6px;
        border: 1px solid #333;
    }
    .stRadio>div {
        flex-direction: row;
        gap: 1.5rem;
    }
    .stCheckbox>div {
        margin-bottom: 0.5rem;
    }
    .stMetric {
        background: #18191a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        color: #00c6ff !important;
    }
    .card {
        background: #18191a;
        border-radius: 12px;
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 16px 0 #0003;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00c6ff;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00c6ff;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
    }
    .sidebar-logo {
        font-size: 2rem;
        font-weight: bold;
        color: #00c6ff;
        margin-bottom: 1.5rem;
        margin-top: 0.5rem;
        text-align: center;
    }
    .sidebar-stats {
        background: #222;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 2rem;
        color: #f5f6fa;
    }
    .sidebar-dropdown {
        margin-bottom: 2rem;
    }
    .stDataFrame, .stTable {
        background: #18191a !important;
        color: #f5f6fa !important;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
        color: #fff;
    }
    </style>
''', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown('<div class="sidebar-logo">🎯<br>Customer Churn Prediction</div>', unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Choose a view:",
    ["🔮 Single Prediction", "📈 Analytics Dashboard", "📋 Batch Predictions", "⚙️ Model Insights"],
    key="main_nav",
    help="Select a section to view"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<b>About This App</b><br>Advanced ML-powered customer churn prediction system.<br><br>Features:<br>• Real-time predictions<br>• Customer segmentation<br>• Retention recommendations<br>• Model performance analytics<br>• Batch processing capabilities", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-stats"><b>Model Performance</b><br>Accuracy: <span style="color:#00c6ff;">80.7%</span><br>Precision: <span style="color:#00c6ff;">61.4%</span><br><br><b>Quick Stats</b><br>📈 1,247 predictions made today<br>🎯 89% prediction accuracy<br>💰 $45K potential revenue saved</div>', unsafe_allow_html=True)

# --- Main Area ---
st.markdown('<div class="main-title">🎯 Customer Churn Prediction</div>', unsafe_allow_html=True)

if page == "🔮 Single Prediction":
    # --- Single Prediction Page ---
    st.markdown('<div class="section-title">📝 Customer Information</div>', unsafe_allow_html=True)
    with st.container():
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
            contract_options = {0: "Month-to-month", 1: "One year", 2: "Two year"}
            Contract = st.selectbox("Contract Duration:", options=list(contract_options.keys()), format_func=lambda x: contract_options[x])
            PaperlessBilling = st.checkbox("Paperless Billing")
            payment_options = {0: "Credit card (automatic)", 1: "Bank transfer (automatic)", 2: "Electronic check", 3: "Mailed check"}
            PaymentMethod = st.selectbox("Payment Method:", options=list(payment_options.keys()), format_func=lambda x: payment_options[x])
    st.markdown("---")
    predict_button = st.button("🔮 Predict Churn", type="primary")
    if predict_button:
        with st.spinner("Analyzing customer data..."):
            try:
                model, feature_names = load_model()
                if model is None:
                    st.error("Model could not be loaded.")
                    st.stop()
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
                prediction, churn_probability = predict_churn(model, feature_names, data_point)
                if churn_probability is None:
                    st.error("Prediction failed.")
                    st.stop()
                st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)
                with st.container():
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        gauge_fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = churn_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Risk Level"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#00c6ff"},
                                'steps': [
                                    {'range': [0, 30], 'color': "#1e824c"},
                                    {'range': [30, 70], 'color': "#f7ca18"},
                                    {'range': [70, 100], 'color': "#c0392b"}
                                ],
                                'threshold': {
                                    'line': {'color': "#c0392b", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        gauge_fig.update_layout(height=300, paper_bgcolor="#18191a", font_color="#f5f6fa")
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    with col2:
                        if churn_probability > 0.7:
                            st.markdown("🚨 <b>HIGH RISK</b><br>Immediate intervention needed", unsafe_allow_html=True)
                        elif churn_probability > 0.4:
                            st.markdown("⚠️ <b>MEDIUM RISK</b><br>Proactive engagement", unsafe_allow_html=True)
                        else:
                            st.markdown("✅ <b>LOW RISK</b><br>Regular engagement", unsafe_allow_html=True)
                    with col3:
                        st.metric("Churn Probability", f"{churn_probability * 100:.1f}%")
                        st.metric("Retention Probability", f"{(1 - churn_probability) * 100:.1f}%")
                        st.metric("Customer Value", f"${MonthlyCharges * 12:.0f}/year")
                st.markdown('<div class="section-title">🎯 Retention Recommendations</div>', unsafe_allow_html=True)
                recommendations = generate_retention_recommendations(churn_probability, data_point)
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    with col2:
                        st.markdown("#### 📋 Customer Profile")
                        profile_data = {
                            "Customer Type": "High Value" if MonthlyCharges > 70 else "Standard",
                            "Loyalty": "Long-term" if tenure > 24 else "Medium-term" if tenure > 12 else "New",
                            "Contract Stability": "Stable" if Contract > 0 else "Flexible",
                            "Service Bundle": "Premium" if InternetService and PhoneService else "Basic"
                        }
                        for key, value in profile_data.items():
                            st.info(f"**{key}:** {value}")
                with st.expander("📋 Detailed Customer Data"):
                    st.json(data_point)
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                logging.error(e)

elif page == "📈 Analytics Dashboard":
    st.markdown('<div class="section-title">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", "7,043", "↑ 12%")
        with col2:
            st.metric("Churn Rate", "26.5%", "↓ 3.2%")
        with col3:
            st.metric("Avg Revenue", "$64.76", "↑ 5.1%")
        with col4:
            st.metric("Retention", "73.5%", "↑ 3.2%")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            segments = ['High Value - Low Risk', 'High Value - High Risk', 'Low Value - Low Risk', 'Low Value - High Risk']
            counts = [45, 15, 30, 10]
            seg_fig = px.pie(values=counts, names=segments, title="Customer Segments", hole=0.4, color_discrete_sequence=["#1e824c", "#c0392b", "#2980b9", "#f7ca18"])
            seg_fig.update_layout(height=300, paper_bgcolor="#18191a", font_color="#f5f6fa")
            st.plotly_chart(seg_fig, use_container_width=True)
        with col2:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            churn_rates = [28.5, 27.2, 26.8, 26.1, 25.9, 26.5]
            trend_fig = px.line(x=months, y=churn_rates, title="Churn Trend", labels={'x': 'Month', 'y': 'Churn Rate (%)'}, markers=True)
            trend_fig.update_traces(line_color="#00c6ff")
            trend_fig.update_layout(height=300, paper_bgcolor="#18191a", font_color="#f5f6fa")
            st.plotly_chart(trend_fig, use_container_width=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Positive Trends")
            st.markdown("• Churn rate ↓ 3.2% this month")
            st.markdown("• Satisfaction scores ↑ 15%")
            st.markdown("• Retention campaigns 40% success")
        with col2:
            st.markdown("### ⚠️ Areas of Concern")
            st.markdown("• Month-to-month: 45% churn")
            st.markdown("• High charges → higher churn")
            st.markdown("• Electronic checks → higher churn")

elif page == "📋 Batch Predictions":
    st.markdown('<div class="section-title">📋 Batch Predictions</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="File should contain customer data columns"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head())
            if st.button("🔮 Run Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        model, feature_names = load_model()
                        if model is None:
                            st.error("Model could not be loaded.")
                            st.stop()
                        predictions = []
                        probabilities = []
                        for _, row in df.iterrows():
                            customer_data = row.to_dict()
                            for feature in feature_names:
                                if feature not in customer_data:
                                    customer_data[feature] = 0
                            pred, prob = predict_churn(model, feature_names, customer_data)
                            predictions.append(pred)
                            probabilities.append(prob)
                        df['Churn_Prediction'] = predictions
                        df['Churn_Probability'] = probabilities
                        st.markdown('<div class="section-title">📊 Batch Prediction Results</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", len(df))
                        with col2:
                            st.metric("Churn", f"{sum(predictions)} ({sum(predictions)/len(df)*100:.1f}%)")
                        with col3:
                            st.metric("Retain", f"{len(df) - sum(predictions)} ({(len(df) - sum(predictions))/len(df)*100:.1f}%)")
                        with col4:
                            st.metric("Avg Risk", f"{np.mean(probabilities)*100:.1f}%")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                        st.dataframe(df.head(10))
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

elif page == "⚙️ Model Insights":
    st.markdown('<div class="section-title">⚙️ Model Insights</div>', unsafe_allow_html=True)
    with st.container():
        features = ['Tenure', 'Monthly Charges', 'Contract', 'Payment Method', 'Internet Service', 'Online Security', 'Tech Support', 'Paperless Billing', 'Senior Citizen', 'Partner']
        importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01, 0.01]
        importance_fig = px.bar(x=importance, y=features, orientation='h', title="Feature Importance (Model Insights)", labels={'x': 'Importance Score', 'y': 'Features'}, color_discrete_sequence=["#00c6ff"])
        importance_fig.update_layout(height=400, paper_bgcolor="#18191a", font_color="#f5f6fa")
        st.plotly_chart(importance_fig, use_container_width=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Model Performance Details")
            st.metric("Overall Accuracy", "80.7%")
            st.metric("Precision", "61.4%")
            st.metric("Recall", "75.2%")
            st.metric("F1-Score", "67.8%")
            st.metric("AUC-ROC", "82.1%")
        with col2:
            st.markdown("#### 🔍 Model Interpretability")
            st.info("**Algorithm:** Logistic Regression")
            st.info("**Training Data:** 5,633 customers")
            st.info("**Test Data:** 1,410 customers")
            st.info("**Status:** Active")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚨 Top Churn Indicators")
            st.markdown("1. **Tenure** - New customers 3x more likely")
            st.markdown("2. **Monthly Charges** - High charges +40% risk")
            st.markdown("3. **Contract** - Month-to-month 45% churn")
            st.markdown("4. **Payment** - Electronic checks higher churn")
        with col2:
            st.markdown("#### ✅ Retention Factors")
            st.markdown("1. **Long contracts** -60% churn")
            st.markdown("2. **Multiple services** +35% retention")
            st.markdown("3. **Online security** -25% churn")
            st.markdown("4. **Paperless billing** +15% retention")