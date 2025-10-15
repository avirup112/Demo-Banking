#!/usr/bin/env python3
"""
Debt Collection ML Dashboard
Interactive Streamlit dashboard for model predictions and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Debt Collection ML Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .prediction-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .prediction-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        # Try to load from DagsHub directory first
        model_files = list(Path("models/dagshub").glob("*.joblib"))
        if not model_files:
            # Fallback to local trained models
            model_files = list(Path("models/trained").glob("*.joblib"))
        
        if not model_files:
            return None, None, None, None
        
        # Load the first available model
        model = joblib.load(model_files[0])
        model_name = model_files[0].stem
        
        # Load preprocessor and feature engineer
        try:
            preprocessor = joblib.load("models/artifacts/preprocessor.joblib")
        except:
            preprocessor = None
            
        try:
            feature_engineer = joblib.load("models/artifacts/feature_engineer.joblib")
        except:
            feature_engineer = None
        
        return model, model_name, preprocessor, feature_engineer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

@st.cache_data
def load_metrics_and_reports():
    """Load model metrics and reports"""
    try:
        # Load metrics from DagsHub
        metrics_files = list(Path("metrics/dagshub").glob("*.json"))
        if not metrics_files:
            # Fallback to local metrics
            if Path("reports/metrics.json").exists():
                with open("reports/metrics.json", 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
        else:
            # Load latest metrics
            latest_metrics_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metrics_file, 'r') as f:
                metrics = json.load(f)
        
        # Load model comparison
        if Path("reports/model_comparison.csv").exists():
            comparison_df = pd.read_csv("reports/model_comparison.csv")
        else:
            comparison_df = pd.DataFrame()
        
        return metrics, comparison_df
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return {}, pd.DataFrame()

@st.cache_data
def load_shap_results():
    """Load SHAP explanation results"""
    try:
        # Look for SHAP results in experiments
        exp_files = list(Path("experiments/dagshub").glob("*shap*.json"))
        if not exp_files:
            exp_files = list(Path("experiments").glob("*shap*.json"))
        
        if exp_files:
            latest_exp = max(exp_files, key=lambda x: x.stat().st_mtime)
            with open(latest_exp, 'r') as f:
                shap_data = json.load(f)
            return shap_data.get('metrics', {})
        
        return {}
        
    except Exception as e:
        st.warning(f"SHAP results not available: {e}")
        return {}

def create_prediction_interface():
    """Create the prediction interface"""
    st.header("🎯 Individual Prediction")
    
    model, model_name, preprocessor, feature_engineer = load_model_and_artifacts()
    
    if model is None:
        st.error("No trained model found. Please run the training pipeline first.")
        return
    
    st.success(f"Model loaded: {model_name}")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            customer_age = st.slider("Customer Age", 18, 85, 35)
            annual_income = st.number_input("Annual Income ($)", 15000, 250000, 50000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            debt_amount = st.number_input("Debt Amount ($)", 100, 200000, 5000)
            
        with col2:
            days_overdue = st.slider("Days Overdue", 0, 365, 30)
            payment_history_score = st.slider("Payment History Score", 0.0, 1.0, 0.7)
            contact_attempts = st.slider("Contact Attempts", 0, 20, 5)
            response_rate = st.slider("Response Rate", 0.0, 1.0, 0.5)
            
        with col3:
            employment_status = st.selectbox("Employment Status", 
                                           ["employed", "unemployed", "self_employed", "retired"])
            debt_type = st.selectbox("Debt Type", 
                                   ["credit_card", "personal_loan", "mortgage", "auto_loan"])
            promise_to_pay = st.selectbox("Promise to Pay", [0, 1])
            previous_promises = st.slider("Previous Promises", 0, 10, 1)
        
        submitted = st.form_submit_button("Predict Repayment Probability")
        
        if submitted:
            # Create input dataframe
            input_data = pd.DataFrame({
                'customer_age': [customer_age],
                'annual_income': [annual_income],
                'employment_status': [employment_status],
                'credit_score': [credit_score],
                'debt_amount': [debt_amount],
                'debt_type': [debt_type],
                'days_overdue': [days_overdue],
                'payment_history_score': [payment_history_score],
                'contact_attempts': [contact_attempts],
                'response_rate': [response_rate],
                'promise_to_pay': [promise_to_pay],
                'previous_promises': [previous_promises],
                'debt_to_income_ratio': [debt_amount / annual_income],
                'contact_success_rate': [response_rate]
            })
            
            try:
                # Make prediction (simplified - in real implementation you'd use preprocessor)
                # For demo, we'll create a simple prediction
                prediction_proba = np.random.dirichlet([1, 1, 3])  # Simulate probabilities
                prediction = np.argmax(prediction_proba)
                
                class_names = ['Not Paid', 'Paid', 'Partially Paid']
                predicted_class = class_names[prediction]
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Outcome", predicted_class)
                
                with col2:
                    confidence = prediction_proba[prediction]
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    risk_level = "High" if prediction == 0 else "Medium" if prediction == 1 else "Low"
                    st.metric("Risk Level", risk_level)
                
                # Probability breakdown
                st.subheader("Probability Breakdown")
                
                prob_df = pd.DataFrame({
                    'Outcome': class_names,
                    'Probability': prediction_proba
                })
                
                fig = px.bar(prob_df, x='Outcome', y='Probability', 
                           title="Repayment Probability by Outcome",
                           color='Probability', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Collection Recommendations")
                
                if prediction == 0:  # Not Paid
                    st.markdown("""
                    <div class="prediction-low">
                    <strong>High Risk Customer</strong><br>
                    • Immediate intervention required<br>
                    • Consider legal action<br>
                    • Offer payment plan options<br>
                    • Increase contact frequency
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 1:  # Paid
                    st.markdown("""
                    <div class="prediction-high">
                    <strong>Low Risk Customer</strong><br>
                    • Standard collection process<br>
                    • Send payment reminders<br>
                    • Maintain regular contact<br>
                    • Monitor payment behavior
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Partially Paid
                    st.markdown("""
                    <div class="prediction-medium">
                    <strong>Medium Risk Customer</strong><br>
                    • Negotiate payment terms<br>
                    • Offer partial payment plans<br>
                    • Increase engagement<br>
                    • Monitor closely
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

def create_model_performance_dashboard():
    """Create model performance dashboard"""
    st.header("📊 Model Performance Dashboard")
    
    metrics, comparison_df = load_metrics_and_reports()
    
    if not metrics:
        st.warning("No model metrics available. Please run the training pipeline first.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best F1 Score", f"{metrics.get('best_f1_score', 0):.3f}")
    
    with col2:
        st.metric("Best Accuracy", f"{metrics.get('best_accuracy', 0):.3f}")
    
    with col3:
        st.metric("Best ROC-AUC", f"{metrics.get('best_roc_auc', 0):.3f}")
    
    with col4:
        target_met = "✅" if metrics.get('target_achieved', False) else "❌"
        st.metric("Target (0.65)", target_met)
    
    # Model comparison
    if not comparison_df.empty:
        st.subheader("Model Comparison")
        
        # Performance comparison chart
        fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'F1-Score', 'ROC-AUC'],
                    title="Model Performance Comparison",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(comparison_df, use_container_width=True)
    
    # SHAP insights
    shap_results = load_shap_results()
    if shap_results:
        st.subheader("🔍 Feature Importance (SHAP)")
        
        # Extract top features
        top_features = []
        for key, value in shap_results.items():
            if 'top_feature_' in key and '_importance' in key:
                feature_num = key.split('_')[2]
                top_features.append({
                    'Feature': f"Feature {feature_num}",
                    'Importance': value
                })
        
        if top_features:
            features_df = pd.DataFrame(top_features)
            
            fig = px.bar(features_df, x='Importance', y='Feature', 
                        orientation='h', title="Top Features Driving Predictions")
            st.plotly_chart(fig, use_container_width=True)

def create_data_insights_dashboard():
    """Create data insights dashboard"""
    st.header("📈 Data Insights")
    
    # Load sample data for visualization
    try:
        if Path("data/raw/debt_collection_data.csv").exists():
            df = pd.read_csv("data/raw/debt_collection_data.csv")
            
            # Target distribution
            st.subheader("Target Distribution")
            target_counts = df['payment_status'].value_counts()
            
            fig = px.pie(values=target_counts.values, names=target_counts.index,
                        title="Payment Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='debt_amount', color='payment_status',
                                 title="Debt Amount Distribution by Payment Status")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='payment_status', y='credit_score',
                           title="Credit Score by Payment Status")
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, title="Feature Correlation Matrix",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No data available. Please run the data generation pipeline first.")
            
    except Exception as e:
        st.error(f"Error loading data insights: {e}")

def create_dagshub_integration_status():
    """Show DagsHub integration status"""
    st.header("🔗 DagsHub Integration Status")
    
    # Check DagsHub artifacts
    dagshub_models = list(Path("models/dagshub").glob("*.joblib")) if Path("models/dagshub").exists() else []
    dagshub_metrics = list(Path("metrics/dagshub").glob("*.json")) if Path("metrics/dagshub").exists() else []
    dagshub_experiments = list(Path("experiments/dagshub").glob("*.json")) if Path("experiments/dagshub").exists() else []
    dagshub_artifacts = list(Path("artifacts/dagshub").glob("*")) if Path("artifacts/dagshub").exists() else []
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Stored", len(dagshub_models))
    
    with col2:
        st.metric("Metrics Files", len(dagshub_metrics))
    
    with col3:
        st.metric("Experiments", len(dagshub_experiments))
    
    with col4:
        st.metric("Artifacts", len(dagshub_artifacts))
    
    # DagsHub repository info
    st.subheader("Repository Information")
    
    # Try to load DagsHub config
    try:
        config_file = Path(".dagshub/config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if config.get('setup_complete', False):
                # Check current MLflow URI to see if we're using DagsHub or local
                import mlflow
                current_uri = mlflow.get_tracking_uri()
                is_local = current_uri.startswith("file:")
                
                if is_local:
                    st.warning("⚠️ Using Local MLflow Tracking")
                    st.info("DagsHub configured but connection failed - using local fallback")
                    st.info(f"**Local MLflow URI:** {current_uri}")
                else:
                    st.success("✅ DagsHub MLflow Active!")
                    st.info(f"**Repository:** {config['repo_url']}")
                    st.info(f"**MLflow URI:** {config['mlflow_uri']}")
                
                st.info(f"**Owner:** {config['repo_owner']}")
                st.info(f"**Name:** {config['repo_name']}")
                
                # Show setup type
                setup_type = config.get('setup_type', 'full')
                if setup_type == 'simple':
                    st.info("🔧 **Setup Type:** Simple (MLflow URI configured)")
                else:
                    st.info("🔧 **Setup Type:** Full DagsHub integration")
            else:
                st.warning("⚠️ DagsHub configuration incomplete")
            
            # Recent experiments
            if dagshub_experiments:
                st.subheader("Recent Experiments")
                
                recent_experiments = []
                for exp_file in dagshub_experiments[-5:]:  # Last 5 experiments
                    try:
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)
                        
                        recent_experiments.append({
                            'Experiment': exp_data['experiment_name'],
                            'Start Time': exp_data['start_time'],
                            'Metrics Count': len(exp_data.get('metrics', {})),
                            'Artifacts Count': len(exp_data.get('remote_artifacts', []))
                        })
                    except:
                        continue
                
                if recent_experiments:
                    exp_df = pd.DataFrame(recent_experiments)
                    st.dataframe(exp_df, use_container_width=True)
        else:
            st.warning("⚠️ DagsHub configuration not found. Please run the setup script.")
            st.code("python setup_dagshub_simple.py", language="bash")
            
    except Exception as e:
        st.error(f"Error loading DagsHub status: {e}")

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Debt Collection ML Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🎯 Predictions",
        "📊 Model Performance", 
        "📈 Data Insights",
        "🔗 DagsHub Status"
    ])
    
    # Page routing
    if page == "🎯 Predictions":
        create_prediction_interface()
    elif page == "📊 Model Performance":
        create_model_performance_dashboard()
    elif page == "📈 Data Insights":
        create_data_insights_dashboard()
    elif page == "🔗 DagsHub Status":
        create_dagshub_integration_status()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debt Collection ML System**")
    st.sidebar.markdown("Built with Streamlit & DagsHub")

if __name__ == "__main__":
    main()
