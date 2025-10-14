#!/usr/bin/env python3
"""
Quick setup script for debt collection ML system
This script handles all dependencies and setup automatically
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Essential packages for the system
    essential_packages = [
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "lightgbm==4.0.0",
        "PyYAML==6.0.1",
        "joblib==1.3.2",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.15.0",
        "streamlit==1.25.0",
        "fastapi==0.101.1",
        "uvicorn==0.23.2",
        "shap==0.42.1",
        "optuna==3.3.0",
        "imbalanced-learn==0.11.0",
        "dvc==3.22.0",
        "dagshub==0.3.1",
        "mlflow==2.5.0",
        "evidently==0.4.2"
    ]
    
    for package in essential_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            # Continue with other packages
    
    print("✅ Package installation completed!")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw", "data/processed", "data/external",
        "models/trained", "models/artifacts", "models/experiments", 
        "logs", "reports", "monitoring/metrics", "experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def setup_dvc_simple():
    """Simple DVC setup without external dependencies"""
    
    try:
        # Check if DVC is available
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
        print("✅ DVC is available")
        
        # Initialize DVC if not already done
        if not Path(".dvc").exists():
            subprocess.run(["dvc", "init"], check=True)
            print("✅ DVC initialized")
        else:
            print("✅ DVC already initialized")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  DVC not available, skipping DVC setup")
        print("   You can install DVC later with: pip install dvc")

def create_basic_files():
    """Create basic configuration files"""
    
    # Create basic params.yaml
    params_content = """
generate_data:
  n_samples: 10000
  random_state: 42

preprocess:
  imputation_strategy: knn
  scaling_method: standard
  encoding_method: onehot
  handle_outliers: true
  outlier_method: iqr

feature_engineering:
  include_polynomial: true
  polynomial_degree: 2
  include_pca: false
  feature_selection: true
  selection_k: 50

training:
  optimize_hyperparameters: true
  n_trials: 30
  cv_folds: 5
  test_size: 0.2
  random_state: 42
"""
    
    with open("params.yaml", "w") as f:
        f.write(params_content.strip())
    
    # Create .env file
    env_content = """
# Debt Collection ML Configuration
DAGSHUB_OWNER=your_dagshub_username
DAGSHUB_REPO=debt-collection-ml
MLFLOW_TRACKING_URI=http://localhost:5000
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    
    print("✅ Configuration files created!")

def run_basic_pipeline():
    """Run basic pipeline without DVC"""
    
    print("Running basic ML pipeline...")
    
    try:
        # Generate data
        print("1. Generating data...")
        subprocess.run([sys.executable, "scripts/generate_data.py"], check=True)
        
        # Preprocess data
        print("2. Preprocessing data...")
        subprocess.run([sys.executable, "scripts/preprocess_data.py"], check=True)
        
        # Feature engineering
        print("3. Engineering features...")
        subprocess.run([sys.executable, "scripts/feature_engineering.py"], check=True)
        
        # Train models (without optimization for speed)
        print("4. Training models...")
        subprocess.run([sys.executable, "scripts/train_model_pipeline.py"], check=True)
        
        print("✅ Basic pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed at step: {e}")
        print("You can run individual steps manually:")
        print("  python scripts/generate_data.py")
        print("  python scripts/preprocess_data.py") 
        print("  python scripts/feature_engineering.py")
        print("  python scripts/train_model_pipeline.py")

def main():
    """Main setup function"""
    
    print("🚀 Setting up Debt Collection ML System...")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Setup DVC (optional)
    setup_dvc_simple()
    
    # Step 4: Create basic files
    create_basic_files()
    
    # Step 5: Ask user if they want to run pipeline
    response = input("\n🤔 Do you want to run the basic ML pipeline now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_basic_pipeline()
    else:
        print("\n📋 Setup completed! You can run the pipeline later with:")
        print("   python scripts/train_model_pipeline.py")
        print("   OR: dvc repro (if DVC is set up)")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📊 Next steps:")
    print("1. Start dashboard: streamlit run src/visualization/dashboard.py")
    print("2. Start API: uvicorn src.api.main:app --reload")
    print("3. View results in reports/ directory")
    
    if Path(".dvc").exists():
        print("4. Check DVC status: dvc status")
        print("5. View pipeline: dvc dag")

if __name__ == "__main__":
    main()