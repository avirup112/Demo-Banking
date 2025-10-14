#!/usr/bin/env python3
"""
Setup script for DagsHub integration
This script helps you set up DagsHub for experiment tracking
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dagshub_integration import setup_dagshub_integration

def main():
    parser = argparse.ArgumentParser(description='Setup DagsHub integration for debt collection ML')
    parser.add_argument('--repo-owner', required=True, help='Your DagsHub username')
    parser.add_argument('--repo-name', default='debt-collection-ml', help='Repository name on DagsHub')
    parser.add_argument('--create-config', action='store_true', help='Create DagsHub configuration files')
    
    args = parser.parse_args()
    
    print("Setting up DagsHub integration...")
    print(f"Repository: {args.repo_owner}/{args.repo_name}")
    
    # Setup DagsHub integration
    tracker = setup_dagshub_integration(args.repo_owner, args.repo_name)
    
    if args.create_config:
        # Create additional configuration files
        
        # Initialize DVC
        os.system("dvc init --no-scm")
        
        # Create DVC remote
        dvc_remote_url = f"https://dagshub.com/{args.repo_owner}/{args.repo_name}.dvc"
        os.system(f"dvc remote add -d origin {dvc_remote_url}")
        
        print("DVC initialized with DagsHub remote!")
        
        # Create .gitignore additions for DagsHub
        gitignore_additions = """
# DagsHub and MLflow
mlruns/
.dagshub/
*.db
*.sqlite

# Model artifacts
models/trained/*.joblib
models/experiments/

# Data files (use DVC instead)
data/raw/*.csv
data/processed/*.npy

# Logs
logs/*.log
"""
        
        gitignore_path = Path('.gitignore')
        if gitignore_path.exists():
            with open(gitignore_path, 'a') as f:
                f.write('\n' + gitignore_additions)
        else:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_additions.strip())
        
        print("Configuration files created!")
    
    # Create example experiment script
    example_script = f"""#!/usr/bin/env python3
\"\"\"
Example experiment script using DagsHub integration
\"\"\"

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dagshub_integration import DagsHubTracker
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def run_example_experiment():
    # Initialize DagsHub tracker
    tracker = DagsHubTracker("{args.repo_owner}", "{args.repo_name}")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 3, 1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start experiment
    with tracker.start_run("example_experiment"):
        # Log parameters
        params = {{
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }}
        tracker.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        metrics = {{
            "accuracy": accuracy,
            "f1_score": f1
        }}
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(model, "example_model")
        
        print(f"Experiment completed! Accuracy: {{accuracy:.4f}}, F1: {{f1:.4f}}")
        print(f"View results at: https://dagshub.com/{args.repo_owner}/{args.repo_name}.mlflow")

if __name__ == "__main__":
    run_example_experiment()
"""
    
    with open('scripts/example_dagshub_experiment.py', 'w') as f:
        f.write(example_script)
    
    print(f"""
DagsHub Setup Complete!
======================

Next Steps:
1. Create a repository on DagsHub: https://dagshub.com/repo/create
   - Repository name: {args.repo_name}
   - Make it public for free tier

2. Initialize git and push to DagsHub:
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://dagshub.com/{args.repo_owner}/{args.repo_name}.git
   git push -u origin main

3. Run the example experiment:
   python scripts/example_dagshub_experiment.py

4. View your experiments:
   https://dagshub.com/{args.repo_owner}/{args.repo_name}.mlflow

5. Run the DVC pipeline:
   python scripts/dvc_pipeline.py init
   python scripts/dvc_pipeline.py run

6. Run individual training:
   python scripts/train_model_pipeline.py --optimize --dagshub-owner {args.repo_owner}

Files created:
- .dagshub.yaml (DagsHub configuration)
- scripts/example_dagshub_experiment.py (Example experiment)
""")
    
    if args.create_config:
        print("- .dvcignore (DVC ignore file)")
        print("- .gitignore (Updated with DagsHub entries)")

if __name__ == "__main__":
    main()