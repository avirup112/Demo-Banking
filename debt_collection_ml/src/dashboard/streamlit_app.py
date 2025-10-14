#!/usr/bin/env python3
"""
Debt Collection ML Dashboard
Interactive Streamlit dashboard for model insights and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
try:
    from explainability.shap_explainer import DebtCollectionExplainer
    from utils.dagshub_integration import DagsHubTracker
except ImportError as e:
    st.error(f"Import error: {e}")

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
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load model results and data"""
    try:
        # Load results summary
        results_path = Path("reports/results_summary.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}
        
        # Load model comparison
        comparison_path = Path("reports/optimized_model_comparison.csv")
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
        else:
            comparison_df = pd.DataFrame()
        
        # Load raw data
        data_path = Path("data/raw/debt_collection_data.csv")
        if data_path.exists():
            raw_data = pd.read_csv(data_path)
        else:
            raw_data = pd.DataFrame()
        
        # Load feature importance
        importance_path = Path("reports/explainability/feature_importance.csv")
        if importance_path.exists():
            feature_importance = pd.read_csv(importance_path)
        else:
            feature_importance = pd.DataFrame()
        
        return results, comparison_df, raw_data, feature_importance
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_model(model_path: str):
    """Load trained model"""
    try:
        if Path(model_path).exists():
            return joblib.load(model_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_collection_propensity_chart(data):
    """Create collection propensity distribution chart"""
    if data.empty or 'payment_status' not in data.columns:
        return go.Figure()
    
    # Count distribution
    status_counts = data['payment_status'].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Collection Propensity Distribution",
        color_discrete_map={
            'full_payment': '#28a745',
            'partial_payment': '#ffc107', 
            'no_payment': '#dc3545'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance chart"""
    if feature_importance.empty:
        return go.Figure()
    
    # Take top 15 features
    top_features = feature_importance.head(15)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title="Top Drivers of Repayment (Feature Importance)",
        labels={'importance': 'SHAP Importance Score', 'feature': 'Features'}
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_model_performance_chart(comparison_df):
    """Create model performance comparison chart"""
    if comparison_df.empty:
        return go.Figure()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                marker_color=colors[i]
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_debt_analysis_charts(data):
    """Create debt amount and age analysis charts"""
    if data.empty:
        return go.Figure(), go.Figure()
    
    # Debt amount distribution by payment status
    fig1 = px.box(
        data,
        x='payment_status',
        y='debt_amount',
        title="Debt Amount Distribution by Payment Status",
        color='payment_status',
        color_discrete_map={
            'full_payment': '#28a745',
            'partial_payment': '#ffc107', 
            'no_payment': '#dc3545'
        }
    )
    fig1.update_layout(height=400)
    
    # Days overdue vs payment status
    if 'days_overdue' in data.columns:
        fig2 = px.histogram(
            data,
            x='days_overdue',
            color='payment_status',
            title="Days Overdue Distribution by Payment Status",
            nbins=30,
            color_discrete_map={
                'full_payment': '#28a745',
                'partial_payment': '#ffc107', 
                'no_payment': '#dc3545'
            }
        )
        fig2.update_layout(height=400)
    else:
        fig2 = go.Figure()
    
    return fig1, fig2

def prediction_interface():
    """Create prediction interface"""
    st.subheader("🔮 Make Predictions")
    
    # Load model
    model_path = "models/optimized/random_forest_optimized.joblib"
    model = load_model(model_path)
    
    if model is None:
        st.warning("No trained model found. Please run the training pipeline first.")
        return
    
    st.write("Enter customer information to predict payment propensity:")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_age = st.slider("Customer Age", 18, 85, 45)
        annual_income = st.number_input("Annual Income ($)", 15000, 250000, 60000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        debt_amount = st.number_input("Debt Amount ($)", 100, 200000, 10000)
        days_overdue = st.slider("Days Overdue", 0, 365, 30)
    
    with col2:
        employment_status = st.selectbox("Employment Status", 
                                       ["employed", "unemployed", "self_employed", "retired"])
        debt_type = st.selectbox("Debt Type", 
                                ["credit_card", "personal_loan", "mortgage", "auto_loan"])
        payment_history_score = st.slider("Payment History Score", 0.0, 1.0, 0.7)
        contact_attempts = st.slider("Contact Attempts", 1, 20, 5)
        response_rate = st.slider("Response Rate", 0.0, 1.0, 0.5)
    
    with col3:
        promise_to_pay = st.selectbox("Promise to Pay", [0, 1])
        previous_promises = st.slider("Previous Promises", 0, 10, 1)
        debt_to_income_ratio = debt_amount / annual_income if annual_income > 0 else 0
        st.write(f"Debt-to-Income Ratio: {debt_to_income_ratio:.3f}")
        contact_success_rate = st.slider("Contact Success Rate", 0.0, 1.0, 0.4)
    
    if st.button("Predict Payment Propensity", type="primary"):
        try:
            # Create feature vector (simplified - would need proper preprocessing)
            features = np.array([[
                customer_age, annual_income, credit_score, debt_amount, days_overdue,
                payment_history_score, contact_attempts, response_rate, promise_to_pay,
                previous_promises, debt_to_income_ratio, contact_success_rate,
                1, 1  # Placeholder for encoded categorical features
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            
            # Display results
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            class_names = ['Full Payment', 'No Payment', 'Partial Payment']
            predicted_class = class_names[prediction]
            
            with col1:
                st.metric("Predicted Outcome", predicted_class)
            
            with col2:
                st.metric("Confidence", f"{prediction_proba[prediction]:.1%}")
            
            with col3:
                risk_level = "Low" if prediction == 0 else "High" if prediction == 1 else "Medium"
                st.metric("Risk Level", risk_level)
            
            # Probability breakdown
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': class_names,
                'Probability': prediction_proba
            })
            
            fig = px.bar(prob_df, x='Outcome', y='Probability', 
                        title="Payment Outcome Probabilities",
                        color='Probability',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Debt Collection ML Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    results, comparison_df, raw_data, feature_importance = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Overview", 
        "📈 Model Performance", 
        "🔍 Feature Analysis",
        "🔮 Make Predictions",
        "📋 Data Explorer"
    ])
    
    if page == "📊 Overview":
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if results and 'best_f1_score' in results:
                st.metric("Best F1 Score", f"{results['best_f1_score']:.3f}")
            else:
                st.metric("Best F1 Score", "N/A")
        
        with col2:
            if results and 'best_model' in results:
                st.metric("Best Model", results['best_model'])
            else:
                st.metric("Best Model", "N/A")
        
        with col3:
            if results and 'dataset_size' in results:
                st.metric("Dataset Size", f"{results['dataset_size']:,}")
            else:
                st.metric("Dataset Size", "N/A")
        
        with col4:
            if results and 'models_trained' in results:
                st.metric("Models Trained", results['models_trained'])
            else:
                st.metric("Models Trained", "N/A")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Collection propensity distribution
            fig1 = create_collection_propensity_chart(raw_data)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Model performance comparison
            fig2 = create_model_performance_chart(comparison_df)
            st.plotly_chart(fig2, use_container_width=True)
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Analysis")
        
        if not comparison_df.empty:
            # Performance metrics table
            st.subheader("Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance trends
            fig = create_model_performance_chart(comparison_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model details
            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]
                st.subheader(f"Best Model: {best_model['Model']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{best_model['Accuracy']:.3f}")
                with col2:
                    st.metric("F1-Score", f"{best_model['F1-Score']:.3f}")
                with col3:
                    st.metric("ROC-AUC", f"{best_model['ROC-AUC']:.3f}")
        else:
            st.warning("No model performance data available. Please run the training pipeline.")
    
    elif page == "🔍 Feature Analysis":
        st.header("Feature Importance Analysis")
        
        if not feature_importance.empty:
            # Feature importance chart
            fig = create_feature_importance_chart(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top features table
            st.subheader("Top 20 Most Important Features")
            st.dataframe(feature_importance.head(20), use_container_width=True)
            
            # Business insights
            st.subheader("Key Business Insights")
            if len(feature_importance) > 0:
                top_feature = feature_importance.iloc[0]
                st.info(f"**Most Important Feature:** {top_feature['feature']} "
                       f"(Importance: {top_feature['importance']:.4f})")
                
                st.write("**Top 5 Drivers of Repayment:**")
                for i, row in feature_importance.head(5).iterrows():
                    st.write(f"{i+1}. **{row['feature']}** - Impact Score: {row['importance']:.4f}")
        else:
            st.warning("No feature importance data available. Please run the explainability analysis.")
    
    elif page == "🔮 Make Predictions":
        prediction_interface()
    
    elif page == "📋 Data Explorer":
        st.header("Data Explorer")
        
        if not raw_data.empty:
            # Data overview
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(raw_data))
            with col2:
                st.metric("Features", len(raw_data.columns))
            with col3:
                st.metric("Missing Values", raw_data.isnull().sum().sum())
            
            # Data sample
            st.subheader("Data Sample")
            st.dataframe(raw_data.head(100), use_container_width=True)
            
            # Distribution analysis
            st.subheader("Distribution Analysis")
            
            # Debt analysis charts
            fig1, fig2 = create_debt_analysis_charts(raw_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation analysis
            if len(raw_data.select_dtypes(include=[np.number]).columns) > 1:
                st.subheader("Feature Correlations")
                numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
                corr_matrix = raw_data[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              title="Feature Correlation Matrix",
                              color_continuous_scale='RdBu',
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available. Please run the data generation pipeline.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Debt Collection ML System**")
    st.sidebar.markdown("Built with Streamlit & Plotly")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()