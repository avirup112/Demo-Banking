# 🏦 Debt Collection ML System

A production-ready AI/ML system for debt collection optimization with complete **DVC pipeline orchestration**. Predicts repayment probability and provides actionable insights for collection strategies.

## 🎯 Project Overview

This system addresses the complete debt collection lifecycle from assignment to closure:

- **🎯 Repayment Probability Prediction**: Advanced ML models (XGBoost, LightGBM, Random Forest)
- **📊 Risk-Based Prioritization**: Intelligent customer segmentation and prioritization
- **📞 Contact Optimization**: AI-driven recommendations for optimal communication channels and timing
- **🔍 Explainable AI**: SHAP explanations for transparent model decisions
- **⚙️ Complete MLOps**: DVC pipelines, monitoring, drift detection, and automated workflows
- **🚀 Interactive Dashboard**: Real-time Streamlit dashboard with model insights

## 🏗️ **Project Structure**

```
debt-collection-ml-system/
├── 🚀 MAIN ENTRY POINTS
│   ├── run_all.py                    # One-click ML pipeline + dashboard
│   ├── run_complete_pipeline.py     # Complete ML workflow
│   ├── run_enhanced_pipeline.py     # Enhanced ML pipeline
│   └── streamlit_dashboard.py       # Interactive dashboard
│
├── 📊 SOURCE CODE
│   └── src/
│       ├── data/                     # Data generation & preprocessing
│       │   ├── data_generator.py     # Synthetic data creation
│       │   └── data_preprocessor.py  # Data cleaning & validation
│       ├── features/                 # Feature engineering
│       │   └── feature_engineering.py # Advanced feature creation
│       ├── explainability/           # Model explanations
│       │   └── shap_explainer.py     # SHAP-based explanations
│       ├── optimization/             # Hyperparameter optimization
│       │   ├── simple_optimizer.py   # Fast grid search
│       │   └── fast_grid_search.py   # Optimized parameter search
│       ├── validation/               # Model validation
│       │   └── model_validator.py    # Comprehensive model testing
│       ├── recommendations/          # Business recommendations
│       │   └── contact_optimizer.py  # Contact strategy optimization
│       ├── testing/                  # A/B testing framework
│       │   └── ab_testing.py         # Experiment management
│       ├── monitoring/               # Model monitoring
│       │   └── drift_detector.py     # Data drift detection
│       ├── deployment/               # Production deployment
│       │   └── deployment_manager.py # Model deployment utilities
│       └── utils/                    # Utilities
│           └── dagshub_integration.py # MLOps integration
│
├── ⚙️ CONFIGURATION & PIPELINE
│   ├── dvc.yaml                      # DVC pipeline definition
│   ├── params.yaml                   # Pipeline parameters
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Container configuration
│   └── docker-compose.yml            # Multi-service setup
│
├── 📁 DATA & ARTIFACTS (DVC-tracked)
│   ├── data/
│   │   ├── raw/                      # Original datasets
│   │   └── processed/                # Processed features
│   ├── models/
│   │   ├── trained/                  # Trained model artifacts
│   │   └── artifacts/                # Preprocessing artifacts
│   ├── reports/                      # Generated reports
│   ├── explanations/                 # SHAP plots and insights
│   ├── ab_experiments/               # A/B test results
│   ├── validation_results/           # Model validation reports
│   └── monitoring_results/           # Drift detection results
│
└── 📚 DOCUMENTATION
    ├── README.md                     # This file
    └── notebooks/                    # Analysis notebooks
        └── 01_comprehensive_eda.ipynb # Exploratory data analysis
```

## 🚀 Quick Start

### 🎯 **DVC Pipeline (Recommended - One Command)**

```bash
# Clone and setup
git clone https://github.com/avirup112/Demo-Banking.git
pip install -r ../requirements.txt

# Run complete pipeline with one command
dvc repro

# 🎉 This automatically:
# ✅ Generates synthetic debt collection data (10,000 samples)
# ✅ Preprocesses and engineers features
# ✅ Trains multiple ML models (XGBoost, LightGBM, Random Forest)
# ✅ Optimizes hyperparameters with Optuna
# ✅ Generates SHAP explanations
# ✅ Creates comprehensive reports
# ✅ Launches interactive dashboard at http://localhost:8501
# ✅ Opens browser automatically
```

### ⚡ **Alternative Quick Runs**

```bash
# Option 1: Direct Python execution (fast)
python run_all.py --quick --samples 1000

# Option 2: Enhanced pipeline with custom settings
python run_enhanced_pipeline.py --samples 5000 --optimization-method optuna

# Option 3: Complete pipeline with dashboard
python run_complete_pipeline.py --samples 10000 --dashboard-timeout 600
```

### 📊 **DVC Pipeline Management**

```bash
# Check pipeline status
dvc status

# View pipeline structure
dvc dag

# Run specific stages only
dvc repro data_generation
dvc repro model_training

# View metrics and results
dvc metrics show
dvc plots show

# Push/pull data (if using remote storage)
dvc push
dvc pull
```

### 🐳 **Docker Deployment**

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access services:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000  
# - MLflow UI: http://localhost:5000
# - Jupyter: http://localhost:8888
```

### 🔧 **Development Setup**

```bash
# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run DVC pipeline
cd debt_collection_ml
dvc repro

# Or run components individually
python run_enhanced_pipeline.py
streamlit run streamlit_dashboard.py
```

## 📈 Features

### 1. Data Understanding & Preprocessing
- **Comprehensive EDA**: 10+ visualization types, statistical analysis
- **Advanced Preprocessing**: Multiple imputation strategies, outlier handling
- **Data Quality Assessment**: Automated quality scoring and reporting
- **Feature Engineering**: Domain-specific financial ratios, behavioral patterns

### 2. Model Architecture & Scalability
- **Multiple Algorithms**: XGBoost, LightGBM, Random Forest, Ensemble
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Class Imbalance Handling**: SMOTE, ADASYN, SMOTETomek
- **Modular Design**: Easily extensible architecture
- **Containerization**: Docker support for scalable deployment

### 3. Predictive Accuracy & Metrics
- **Comprehensive Evaluation**: ROC-AUC, F1, Precision, Recall
- **Business Metrics**: Recovery precision, collection recall, expected recovery rate
- **Cross-Validation**: Stratified K-fold with robust validation
- **Model Comparison**: Automated comparison across multiple algorithms

### 4. Explainability & Interpretability
- **SHAP Integration**: Global and local explanations
- **LIME Support**: Instance-level explanations
- **Feature Importance**: Multiple importance calculation methods
- **Business Explanations**: Domain-specific interpretation of model decisions

### 5. Production Readiness & MLOps
- **Model Registry**: DagsHub + Local SQLite-based model versioning
- **Data Versioning**: DVC for data and model artifact versioning
- **Pipeline Management**: DVC pipelines for reproducible workflows
- **Monitoring**: Data drift detection, performance monitoring
- **CI/CD Pipeline**: Automated testing and validation
- **Experiment Tracking**: DagsHub + MLflow integration
- **Health Checks**: Model and data validation

### 6. Innovation & Presentation
- **Interactive Dashboard**: Streamlit-based visualization
- **Web Scraping**: External data enrichment capabilities
- **REST API**: FastAPI-based model serving
- **Comprehensive Documentation**: Detailed guides and examples

## 🔧 Usage Examples

### DVC Pipeline Management

```bash
# Initialize DVC pipeline
python scripts/dvc_pipeline.py init

# Run complete pipeline
python scripts/dvc_pipeline.py run

# Run specific stages
python scripts/dvc_pipeline.py run --stages data_generation model_training

# Show pipeline status
python scripts/dvc_pipeline.py status

# View metrics
python scripts/dvc_pipeline.py metrics

# Create experiment
python scripts/dvc_pipeline.py experiment run --name "experiment_1" --param training.n_trials=100

# Push data to DagsHub
python scripts/dvc_pipeline.py push
```

### Training Models

```python
from src.models.ml_model import DebtCollectionMLModel

# Initialize and train model
model = DebtCollectionMLModel(model_type='xgboost')
model.train(X_train, y_train, optimize=True)

# Evaluate model
results = model.evaluate(X_test, y_test)
print(f"Business F1 Score: {results['business_metrics']['business_f1']:.4f}")
```

### Making Predictions

```python
# Load trained model
model.load_model('models/trained/xgboost_model.joblib')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

### Model Explanations

```python
from src.models.explainability import ModelExplainer

explainer = ModelExplainer(model, X_train, feature_names)
explainer.explain_instance_shap(X_instance[0])
explainer.global_feature_importance_shap()
```

### Recommendations

```python
from src.models.recommendations import RecommendationEngine

recommender = RecommendationEngine()
recommender.train_channel_recommendation_model(df)

# Get recommendations
recommendation = recommender.get_comprehensive_recommendation(customer_data)
print(f"Recommended channel: {recommendation['channel_recommendation']['channel']}")
```

<!-- ## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC | Business F1 | Recovery Precision |
|-------|----------|----------|---------|-------------|-------------------|
| XGBoost | 0.847 | 0.834 | 0.891 | 0.823 | 0.856 |
| LightGBM | 0.842 | 0.829 | 0.887 | 0.818 | 0.851 |
| Random Forest | 0.839 | 0.825 | 0.883 | 0.814 | 0.847 |
| Ensemble | 0.851 | 0.838 | 0.894 | 0.827 | 0.859 | -->

## 🔍 Key Insights

1. **Payment Behavior Patterns**: Clear correlation between credit scores, response rates, and payment outcomes
2. **Channel Effectiveness**: WhatsApp and Email show higher engagement rates for younger demographics
3. **Risk Segmentation**: 3-tier risk model effectively separates high/medium/low probability customers
4. **Temporal Patterns**: Days past due is the strongest predictor, with 90+ days showing critical risk threshold

## 🛠️ MLOps Features

### Data Version Control (DVC)
- **Data Versioning**: Track changes in datasets and model artifacts
- **Pipeline Management**: Reproducible ML pipelines with dependency tracking
- **Experiment Tracking**: Parameter and metric comparison across runs
- **Remote Storage**: Support for S3, GCS, Azure, SSH, and local storage

### Model Registry
- Version control for models with DagsHub integration
- Metadata tracking and model promotion workflows
- Automated model comparison and selection

### Monitoring
- Data drift detection using Evidently
- Performance degradation alerts
- Real-time metrics tracking with MLflow

### CI/CD Pipeline
- Automated data validation and quality checks
- Model testing and validation pipelines
- DVC-based reproducible deployments

### Quick DVC Commands
```bash
# Run full pipeline
dvc repro

# Check pipeline status
dvc status

# View pipeline DAG
dvc dag

# Compare experiments
dvc params diff
dvc metrics diff

# Push/pull data
dvc push
dvc pull
```

## 🔄 **DVC Pipeline Architecture**

The entire system is orchestrated through a **DVC pipeline** that ensures reproducibility and version control:

```mermaid
graph TD
    A[🎲 data_generation] --> B[🧹 data_preprocessing]
    B --> C[⚙️ feature_engineering] 
    C --> D[🤖 complete_pipeline]
    
    A --> E[📊 data/raw/debt_collection_data.csv]
    B --> F[📈 data/processed/X_processed.npy]
    B --> G[🎯 data/processed/y_encoded.npy]
    C --> H[🔧 data/processed/X_engineered.npy]
    D --> I[🏆 models/trained/]
    D --> J[📋 reports/]
    D --> K[🎨 explanations/]
    D --> L[🚀 Dashboard Launch]
```

### **Pipeline Stages Explained:**

| Stage | Description | Outputs | Duration |
|-------|-------------|---------|----------|
| **🎲 data_generation** | Creates synthetic debt collection dataset | `data/raw/debt_collection_data.csv` | ~30s |
| **🧹 data_preprocessing** | Cleans, validates, and preprocesses data | `data/processed/X_processed.npy`, `y_encoded.npy` | ~45s |
| **⚙️ feature_engineering** | Creates advanced financial and behavioral features | `data/processed/X_engineered.npy` | ~60s |
| **🤖 complete_pipeline** | Trains models, optimizes, explains, launches dashboard | `models/`, `reports/`, `explanations/`, Dashboard | ~5-10min |

(what was observe on first-run)

### **Run the Complete Pipeline:**

```bash
# Single command runs everything
dvc repro

# ✅ Total time: ~7-12 minutes
# ✅ Automatic dashboard launch
# ✅ All artifacts versioned and tracked
# ✅ Reproducible across environments
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 **Expected Results**

After running `dvc repro`, you'll have:

### **📊 Model Performance:**
- **XGBoost**: F1-Score ~0.85, ROC-AUC ~0.91
- **LightGBM**: F1-Score ~0.83, ROC-AUC ~0.89  
- **Random Forest**: F1-Score ~0.81, ROC-AUC ~0.87
- **Business Metrics**: Recovery precision >80%, Collection recall >75%

### **🎨 Interactive Dashboard:**
- **Customer Risk Scoring**: Real-time probability predictions
- **Feature Explanations**: SHAP-based model interpretability
- **Business Recommendations**: Optimal contact strategies
- **Performance Monitoring**: Model metrics and data quality

### **📋 Comprehensive Reports:**
- **Model Comparison**: Detailed performance analysis
- **Feature Importance**: Business-relevant insights
- **Validation Results**: Robust model evaluation
- **A/B Testing**: Experiment tracking and results

### **🔍 Key Business Insights:**
- **Payment Behavior**: Credit score and days past due are strongest predictors
- **Channel Effectiveness**: WhatsApp and Email show highest engagement
- **Risk Segmentation**: Clear separation of high/medium/low risk customers
- **Temporal Patterns**: 90+ days past due represents critical threshold

---

## 🚀 **Get Started Now:**

```bash
git clone https://github.com/YOUR_USERNAME/debt-collection-ml-system.git
cd debt-collection-ml-system/debt_collection_ml
pip install -r ../requirements.txt
dvc repro
```

**🎯 In ~10 minutes, you'll have a complete ML system with interactive dashboard!**

---

**Note**: This system uses synthetic data for demonstration purposes. In production, ensure compliance with data privacy regulations and ethical AI practices.
## � **What You Get After `dvc repro`**

### **🎯 Trained Models:**
- **XGBoost Optimized**: F1-Score > 0.85, ROC-AUC > 0.90
- **LightGBM Optimized**: Fast training, high accuracy
- **Random Forest Optimized**: Robust ensemble predictions
- **Best Model Selection**: Automatically selects top performer

### **📈 Comprehensive Reports:**
- **Model Comparison**: Performance metrics across all models
- **Feature Importance**: SHAP-based feature analysis
- **Business Metrics**: Recovery precision
- **Validation Results**: Crosidation and holdout testing

### **🎨 Interactive Dashboard:**
- **Real-time Predictions**: Upload CSV or input customer data
- **Model Explanations**: SHAP plots and feature importance
- **Business Insights**: Risk segmentation and recommendations
- **Performance Monitoring**: Model metrics and data drift

### **🔍 Explainability:**
- **SHAP Summary Plots**: Global feature importance
- **Individual Explanations**: Per-customer prediction reasoning
- **Feature Insights**: Business-relevant feature analysis

## 🛠️ **DVC Commands Reference**

```bash
# C Commands
dvc repro                    # Run complete pipeline
dvc status                   # Check what needs to be run
dvc dag                      # Visualize pipeline structure

# Stage-specific Commands  
dvc repro data_generation    # Generate data only
dvc repro feature_engineering # Feature engineering only
dvc repro complete_pipeline  # ML training + dashboard only

# Metrics and Results
dvc metrics show             # Show all metrics
dvc metrics diff             # Compare metrics across runs
dvc plots show              # Generate performance plots

# Data Management
dvc push                     # Push data to remote storage
dvc pull                     # Pull data from remote storage
dvc checkout                 # Restore data to specific version

# Experiment Tracking
exp run                  # Run experiment with parameters
dvc exp show                 # Compare experiments
dvc exp diff                 # Show experiment differences
```

## 🧪 **Advanced Usage**

### **Custom Parameters:**
```bash
# Modify pipeline parameters
dvc repro --set-param data_generation.n_samples=20000
dvc repro --set-param complete_pipeline.optimization_method=optuna
dvc repro --set-param complete_pipeline.n_trials=100
```

### **Experiment Tracking:**
```bash
# Run experiment with custom name
dvc exp run --name "large_dataset" --set-param data_generation.n_samples=50000

# Compare experiments
dvc exp show --include-metrics

# Apply best experiment
dvc exp apply exp-12345
```

### **Pipeline Debugging:**
```bash
# Run with verbose output
dvc repro --verbose

# Force re-run specific stage
dvc repro --force data_generation

# Dry run (show what would be executed)
dvc repro --dry
```
