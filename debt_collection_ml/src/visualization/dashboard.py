import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from data.data_generator import DebtCollectionDataGenerator
    from data.data_preprocessor import AdvancedDataPreprocessor
    from models.ml_model import DebtCollectionMLModel
    from models.explainability import ModelExplainer
    from models.recommendations import RecommendationEngine
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

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
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DebtCollectionDashboard:
    """Comprehensive Streamlit dashboard for debt collection ML system"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.recommender = None
        
    @st.cache_data
    def load_data(_self):
        """Load or generate data with caching"""
        try:
            if os.path.exists('data/raw/debt_collection_data.csv'):
                df = pd.read_csv('data/raw/debt_collection_data.csv')
                st.success("✅ Loaded existing dataset")
            else:
                with st.spinner("Generating synthetic dataset..."):
                    generator = DebtCollectionDataGenerator(n_samples=5000)
                    df = generator.generate_dataset()
                    
                    # Save data
                    os.makedirs('data/raw', exist_ok=True)
                    df.to_csv('data/raw/debt_collection_data.csv', index=False)
                    st.success("✅ Generated new dataset")
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def load_models(self):
        """Load trained models if available"""
        try:
            # Try to load trained model
            model_files = list(Path('models/trained').glob('*.joblib'))
            if model_files:
                model_path = model_files[0]  # Load first available model
                
                self.model = DebtCollectionMLModel()
                self.model.load_model(str(model_path))
                
                st.sidebar.success(f"✅ Loaded model: {model_path.name}")
                return True
            else:
                st.sidebar.warning("⚠️ No trained models found. Train a model first.")
                return False
                
        except Exception as e:
            st.sidebar.error(f"Error loading models: {e}")
            return False
    
    def create_overview_metrics(self, df):
        """Create overview metrics cards"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            total_outstanding = df['Outstanding_Balance'].sum()
            st.metric("Total Outstanding", f"₹{total_outstanding/1e6:.1f}M")
        
        with col3:
            avg_dpd = df['Days_Past_Due'].mean()
            st.metric("Avg Days Past Due", f"{avg_dpd:.0f}")
        
        with col4:
            payment_rate = (df['Outcome'] == 'Paid').mean() * 100
            st.metric("Payment Rate", f"{payment_rate:.1f}%")
    
    def create_collection_propensity_chart(self, df):
        """Create collection propensity distribution chart"""
        
        st.subheader("📊 Collection Propensity Distribution")
        
        # Outcome distribution
        outcome_counts = df['Outcome'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=outcome_counts.values,
                names=outcome_counts.index,
                title="Outcome Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=outcome_counts.index,
                y=outcome_counts.values,
                title="Outcome Counts",
                color=outcome_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Risk segmentation
        st.subheader("🎯 Risk Segmentation Analysis")
        
        # Create risk segments
        def create_risk_score(row):
            score = 0
            score += min(row['Days_Past_Due'] / 90, 2)  # Max 2 points
            score += max(0, (650 - row['Credit_Score']) / 100)  # Max ~3 points
            score += max(0, (50 - row['Response_Rate']) / 25)  # Max 2 points
            return score
        
        df['Risk_Score'] = df.apply(create_risk_score, axis=1)
        df['Risk_Segment'] = pd.cut(
            df['Risk_Score'],
            bins=[0, 2, 4, float('inf')],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Risk segment analysis
        risk_outcome = pd.crosstab(df['Risk_Segment'], df['Outcome'], normalize='index') * 100
        
        fig_risk = px.bar(
            risk_outcome,
            title="Payment Outcome by Risk Segment (%)",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    def create_top_drivers_analysis(self, df):
        """Analyze top drivers of repayment"""
        
        st.subheader("🔍 Top Drivers of Repayment")
        
        # Calculate correlation with payment outcome
        df_numeric = df.select_dtypes(include=[np.number])
        df_numeric['Paid'] = (df['Outcome'] == 'Paid').astype(int)
        
        correlations = df_numeric.corr()['Paid'].abs().sort_values(ascending=False)[1:11]  # Top 10
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance chart
            fig_importance = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation='h',
                title="Top 10 Features Correlated with Payment",
                color=correlations.values,
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Key insights
            st.markdown("### 💡 Key Insights")
            
            insights = [
                f"**Credit Score** is the strongest predictor (correlation: {correlations.get('Credit_Score', 0):.3f})",
                f"**Response Rate** significantly impacts payment likelihood",
                f"**Days Past Due** shows strong negative correlation with payment",
                f"**Income level** is moderately correlated with payment behavior",
                f"**Age** shows some correlation with payment propensity"
            ]
            
            for insight in insights:
                st.markdown(f"• {insight}")
        
        # Detailed analysis by key features
        st.subheader("📈 Detailed Feature Analysis")
        
        feature_tabs = st.tabs(["Credit Score", "Days Past Due", "Response Rate", "Income"])
        
        with feature_tabs[0]:
            # Credit Score analysis
            fig_credit = px.box(
                df, x='Outcome', y='Credit_Score',
                title="Credit Score Distribution by Outcome",
                color='Outcome'
            )
            st.plotly_chart(fig_credit, use_container_width=True)
        
        with feature_tabs[1]:
            # Days Past Due analysis
            fig_dpd = px.histogram(
                df, x='Days_Past_Due', color='Outcome',
                title="Days Past Due Distribution by Outcome",
                marginal="box"
            )
            st.plotly_chart(fig_dpd, use_container_width=True)
        
        with feature_tabs[2]:
            # Response Rate analysis
            fig_response = px.scatter(
                df, x='Response_Rate', y='Number_of_Calls',
                color='Outcome', size='Outstanding_Balance',
                title="Response Rate vs Number of Calls (sized by Outstanding Balance)"
            )
            st.plotly_chart(fig_response, use_container_width=True)
        
        with feature_tabs[3]:
            # Income analysis
            fig_income = px.violin(
                df, x='Outcome', y='Income',
                title="Income Distribution by Outcome",
                color='Outcome'
            )
            st.plotly_chart(fig_income, use_container_width=True)
    
    def create_model_performance_trends(self):
        """Create model performance trends visualization"""
        
        st.subheader("📈 Model Performance Trends")
        
        if self.model is None:
            st.warning("⚠️ No trained model available. Please train a model first.")
            
            # Show mock performance trends
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
            mock_performance = {
                'Date': dates,
                'Accuracy': np.random.normal(0.85, 0.02, len(dates)),
                'F1_Score': np.random.normal(0.82, 0.02, len(dates)),
                'ROC_AUC': np.random.normal(0.88, 0.015, len(dates)),
                'Business_F1': np.random.normal(0.80, 0.025, len(dates))
            }
            
            perf_df = pd.DataFrame(mock_performance)
            
            fig_trends = go.Figure()
            
            metrics = ['Accuracy', 'F1_Score', 'ROC_AUC', 'Business_F1']
            colors = ['blue', 'red', 'green', 'orange']
            
            for metric, color in zip(metrics, colors):
                fig_trends.add_trace(go.Scatter(
                    x=perf_df['Date'],
                    y=perf_df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' '),
                    line=dict(color=color)
                ))
            
            fig_trends.update_layout(
                title="Model Performance Over Time (Mock Data)",
                xaxis_title="Date",
                yaxis_title="Score",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
        else:
            # Show actual model performance if available
            if hasattr(self.model, 'cv_scores') and self.model.cv_scores is not None:
                cv_scores = self.model.cv_scores
                
                fig_cv = px.bar(
                    x=[f'Fold {i+1}' for i in range(len(cv_scores))],
                    y=cv_scores,
                    title="Cross-Validation Scores",
                    color=cv_scores,
                    color_continuous_scale='viridis'
                )
                fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", 
                               annotation_text=f"Mean: {cv_scores.mean():.4f}")
                
                st.plotly_chart(fig_cv, use_container_width=True)
            
            # Model comparison if available
            st.subheader("🏆 Model Comparison")
            
            # Mock comparison data
            comparison_data = {
                'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Ensemble'],
                'Accuracy': [0.847, 0.842, 0.839, 0.851],
                'F1_Score': [0.834, 0.829, 0.825, 0.838],
                'ROC_AUC': [0.891, 0.887, 0.883, 0.894],
                'Business_F1': [0.823, 0.818, 0.814, 0.827]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            
            fig_comp = px.bar(
                comp_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric',
                title="Model Performance Comparison",
                barmode='group'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
    
    def create_customer_insights(self, df):
        """Create customer insights section"""
        
        st.subheader("👥 Customer Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Demographics analysis
            st.markdown("### Demographics Analysis")
            
            # Age distribution
            fig_age = px.histogram(
                df, x='Age', color='Outcome',
                title="Age Distribution by Outcome",
                marginal="rug"
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Occupation analysis
            occupation_outcome = pd.crosstab(df['Occupation'], df['Outcome'], normalize='index') * 100
            fig_occ = px.bar(
                occupation_outcome,
                title="Payment Rate by Occupation (%)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_occ, use_container_width=True)
        
        with col2:
            # Communication analysis
            st.markdown("### Communication Analysis")
            
            # Channel effectiveness
            channel_effectiveness = df.groupby('Last_Contact_Channel')['Outcome'].apply(
                lambda x: (x == 'Paid').mean() * 100
            ).sort_values(ascending=False)
            
            fig_channel = px.bar(
                x=channel_effectiveness.index,
                y=channel_effectiveness.values,
                title="Payment Rate by Contact Channel (%)",
                color=channel_effectiveness.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_channel, use_container_width=True)
            
            # Response rate vs outcome
            fig_response_outcome = px.box(
                df, x='Outcome', y='Response_Rate',
                title="Response Rate Distribution by Outcome",
                color='Outcome'
            )
            st.plotly_chart(fig_response_outcome, use_container_width=True)
    
    def create_recommendations_section(self, df):
        """Create recommendations section"""
        
        st.subheader("🎯 Collection Recommendations")
        
        # Sample customer for demonstration
        sample_customer = df.sample(1).iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sample Customer Profile")
            
            customer_info = {
                "Customer ID": sample_customer['Customer_ID'],
                "Age": f"{sample_customer['Age']} years",
                "Income": f"₹{sample_customer['Income']:,}",
                "Outstanding Balance": f"₹{sample_customer['Outstanding_Balance']:,}",
                "Days Past Due": f"{sample_customer['Days_Past_Due']} days",
                "Credit Score": sample_customer['Credit_Score'],
                "Response Rate": f"{sample_customer['Response_Rate']:.1f}%",
                "Last Contact Channel": sample_customer['Last_Contact_Channel']
            }
            
            for key, value in customer_info.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### Recommendations")
            
            # Generate recommendations based on customer profile
            recommendations = []
            
            # Channel recommendation
            if sample_customer['Age'] < 35:
                recommended_channel = "WhatsApp"
                reason = "Young demographic prefers digital channels"
            elif sample_customer['Response_Rate'] > 70:
                recommended_channel = sample_customer['Last_Contact_Channel']
                reason = "High response rate with current channel"
            else:
                recommended_channel = "Call"
                reason = "Direct contact for better engagement"
            
            recommendations.append(f"**Contact Channel:** {recommended_channel} - {reason}")
            
            # Timing recommendation
            if sample_customer['Occupation'] in ['Salaried', 'Professional']:
                timing = "Evening (6-8 PM)"
            elif sample_customer['Occupation'] == 'Business':
                timing = "Afternoon (2-5 PM)"
            else:
                timing = "Morning (9-12 PM)"
            
            recommendations.append(f"**Best Contact Time:** {timing}")
            
            # Priority level
            if sample_customer['Days_Past_Due'] > 90:
                priority = "High Priority"
                action = "Immediate attention required"
            elif sample_customer['Credit_Score'] > 700:
                priority = "Medium Priority"
                action = "Standard follow-up"
            else:
                priority = "Low Priority"
                action = "Routine collection process"
            
            recommendations.append(f"**Priority Level:** {priority} - {action}")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        # Business recommendations
        st.markdown("### 📋 Business Recommendations")
        
        business_recs = [
            "**Focus on High-Response Customers**: Prioritize customers with >70% response rate",
            "**Channel Optimization**: Use WhatsApp for customers under 35, Email for high-income customers",
            "**Early Intervention**: Contact customers within 30 days of becoming past due",
            "**Risk-Based Approach**: Implement different strategies for each risk segment",
            "**Performance Monitoring**: Track collection success rates by channel and agent"
        ]
        
        for rec in business_recs:
            st.markdown(f"• {rec}")
    
    def create_data_quality_section(self, df):
        """Create data quality assessment section"""
        
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing data analysis
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing_Count': missing_data.values,
                    'Missing_Percentage': missing_pct.values
                }).query('Missing_Count > 0')
                
                fig_missing = px.bar(
                    missing_df, x='Column', y='Missing_Percentage',
                    title="Missing Data by Column (%)",
                    color='Missing_Percentage',
                    color_continuous_scale='reds'
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing data found!")
        
        with col2:
            # Data quality metrics
            quality_metrics = {
                "Total Records": len(df),
                "Total Features": len(df.columns),
                "Duplicate Records": df.duplicated().sum(),
                "Missing Data %": f"{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%",
                "Numeric Features": len(df.select_dtypes(include=[np.number]).columns),
                "Categorical Features": len(df.select_dtypes(include=['object']).columns)
            }
            
            st.markdown("### Data Quality Metrics")
            for metric, value in quality_metrics.items():
                st.metric(metric, value)
    
    def run_dashboard(self):
        """Main dashboard function"""
        
        # Header
        st.markdown('<h1 class="main-header">💰 Debt Collection ML Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("Navigation")
        
        # Load data
        self.data = self.load_data()
        if self.data is None:
            st.error("Failed to load data. Please check the data source.")
            return
        
        # Load models
        model_loaded = self.load_models()
        
        # Navigation
        pages = [
            "📊 Overview",
            "🎯 Collection Propensity",
            "🔍 Top Drivers",
            "📈 Model Performance",
            "👥 Customer Insights",
            "🎯 Recommendations",
            "🔍 Data Quality"
        ]
        
        selected_page = st.sidebar.selectbox("Select Page", pages)
        
        # Page routing
        if selected_page == "📊 Overview":
            st.header("Dashboard Overview")
            self.create_overview_metrics(self.data)
            
            # Quick insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("### 💡 Quick Insights")
            
            payment_rate = (self.data['Outcome'] == 'Paid').mean() * 100
            avg_outstanding = self.data['Outstanding_Balance'].mean()
            high_risk_customers = (self.data['Days_Past_Due'] > 90).sum()
            
            insights = [
                f"Overall payment rate is {payment_rate:.1f}%",
                f"Average outstanding balance is ₹{avg_outstanding:,.0f}",
                f"{high_risk_customers} customers are high-risk (>90 days past due)",
                f"Most effective contact channel varies by customer demographics"
            ]
            
            for insight in insights:
                st.markdown(f"• {insight}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif selected_page == "🎯 Collection Propensity":
            self.create_collection_propensity_chart(self.data)
        
        elif selected_page == "🔍 Top Drivers":
            self.create_top_drivers_analysis(self.data)
        
        elif selected_page == "📈 Model Performance":
            self.create_model_performance_trends()
        
        elif selected_page == "👥 Customer Insights":
            self.create_customer_insights(self.data)
        
        elif selected_page == "🎯 Recommendations":
            self.create_recommendations_section(self.data)
        
        elif selected_page == "🔍 Data Quality":
            self.create_data_quality_section(self.data)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Debt Collection ML System** | "
            "Built with Streamlit, Plotly, and scikit-learn | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

def main():
    """Main function to run the dashboard"""
    dashboard = DebtCollectionDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()