# Customer Churn Prediction - Production Deployment

## 🚀 Production-Ready Deployment

This branch contains the production-ready version of the Customer Churn Prediction application, optimized for deployment on Railway.

### ✨ Key Features

- **Local Model Loading**: Uses joblib to load pre-trained models (no MLflow dependencies)
- **Mobile Responsive**: Optimized for all device sizes
- **Compact Interface**: Single viewport design with tabbed navigation
- **Real-time Predictions**: Instant churn risk assessment
- **Batch Processing**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: Key metrics and insights
- **Model Insights**: Feature importance and performance metrics

### 📦 What's Included

- `app.py` - Production Streamlit application
- `saved_model/` - Pre-trained model files
- `requirements.txt` - Production dependencies
- `railway.json` - Railway deployment configuration

### 🛠️ Dependencies

- `streamlit` - Web application framework
- `scikit-learn` - Machine learning library
- `joblib` - Model serialization
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations

### 🚀 Deployment on Railway

1. **Connect Repository**: Link this GitHub repository to Railway
2. **Select Branch**: Choose the `production` branch
3. **Auto-Deploy**: Railway will automatically detect and deploy the Streamlit app
4. **Environment Variables**: No additional configuration needed
5. **Access URL**: Railway will provide a public URL for your app

### 📱 App Features

#### 🔮 Single Prediction
- Customer information input
- Real-time churn risk assessment
- Visual risk gauge
- Personalized recommendations

#### 📊 Analytics Dashboard
- Key performance metrics
- Customer segmentation
- Churn trend analysis
- Actionable insights

#### 📋 Batch Predictions
- CSV file upload
- Bulk customer analysis
- Results download
- Summary statistics

#### ⚙️ Model Insights
- Feature importance analysis
- Model performance metrics
- Top churn indicators
- Retention factors

### 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### 📈 Model Performance

- **Accuracy**: 80.7%
- **Precision**: 61.4%
- **Recall**: 75.2%
- **F1-Score**: 67.8%

### 🎯 Key Insights

**Top Churn Indicators:**
1. Tenure (new customers 3x more likely to churn)
2. Monthly Charges (high charges +40% risk)
3. Contract Type (month-to-month 45% churn)
4. Payment Method (electronic checks higher churn)

**Retention Factors:**
1. Long-term contracts (-60% churn)
2. Multiple services (+35% retention)
3. Online security (-25% churn)
4. Paperless billing (+15% retention)

### 🔄 Updates

This production branch is separate from the development branch to ensure stability. Updates are deployed through:

1. Development in the `main` branch
2. Testing and validation
3. Merge to `production` branch
4. Automatic deployment on Railway

### 📞 Support

For issues or questions about the production deployment, please check the development branch for detailed documentation and troubleshooting guides. 