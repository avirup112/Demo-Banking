#!/usr/bin/env python3
"""
Simple DagsHub repository setup script
This script helps you configure DagsHub for your debt collection ML project
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def setup_dagshub_repo():
    """Setup DagsHub repository configuration"""
    
    print("🚀 Setting up DagsHub for Debt Collection ML Project")
    print("=" * 50)
    
    # Get user input
    repo_owner = input("Enter your DagsHub username: ").strip()
    repo_name = input("Enter repository name (default: debt-collection-ml): ").strip()
    
    if not repo_name:
        repo_name = "debt-collection-ml"
    
    if not repo_owner:
        print("❌ Repository owner is required!")
        return
    
    print(f"\n📁 Repository: {repo_owner}/{repo_name}")
    print(f"🌐 URL: https://dagshub.com/{repo_owner}/{repo_name}")
    
    # Create DagsHub configuration
    dagshub_config = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "repo_url": f"https://dagshub.com/{repo_owner}/{repo_name}",
        "data_remote": f"https://dagshub.com/{repo_owner}/{repo_name}.dvc",
        "setup_date": "2025-10-14"
    }
    
    # Save configuration
    config_path = Path(".dagshub_config.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(dagshub_config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")
    
    # Initialize DVC if not already done
    if not Path(".dvc").exists():
        print("\n📦 Initializing DVC...")
        run_command("dvc init --no-scm")
        print("✅ DVC initialized")
    
    # Add DagsHub as DVC remote
    print("\n🔗 Setting up DVC remote...")
    dvc_remote_url = f"https://dagshub.com/{repo_owner}/{repo_name}.dvc"
    
    # Remove existing remote if it exists
    run_command("dvc remote remove origin", cwd=".")
    
    # Add new remote
    result = run_command(f"dvc remote add -d origin {dvc_remote_url}")
    if result is not None:
        print("✅ DVC remote configured")
    
    # Create .gitignore additions
    gitignore_additions = """
# DagsHub and DVC
.dagshub/
*.dvc
/data/raw/*.csv
/data/processed/*.npy
/models/trained/*.joblib
/experiments/
.dvcignore

# Logs and temporary files
logs/*.log
temp/
*.tmp
"""
    
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'a') as f:
            f.write('\n' + gitignore_additions)
    else:
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_additions.strip())
    
    print("✅ .gitignore updated")
    
    # Create DVC pipeline configuration
    dvc_yaml_content = f"""
# DVC Pipeline for Debt Collection ML
# Repository: {repo_owner}/{repo_name}

stages:
  data_generation:
    cmd: python src/data/data_generator.py --output data/raw/synthetic_debt_data.csv --size 10000
    deps:
    - src/data/data_generator.py
    outs:
    - data/raw/synthetic_debt_data.csv
    
  data_preprocessing:
    cmd: python src/data/data_preprocessor.py --input data/raw/synthetic_debt_data.csv --output data/processed/
    deps:
    - src/data/data_preprocessor.py
    - data/raw/synthetic_debt_data.csv
    outs:
    - data/processed/X_train.npy
    - data/processed/X_test.npy
    - data/processed/y_train.npy
    - data/processed/y_test.npy
    
  feature_engineering:
    cmd: python src/features/feature_engineering.py --input data/processed/ --output data/features/
    deps:
    - src/features/feature_engineering.py
    - data/processed/X_train.npy
    - data/processed/X_test.npy
    outs:
    - data/features/X_train_features.npy
    - data/features/X_test_features.npy
    
  model_training:
    cmd: python run_complete_pipeline.py --dagshub-owner {repo_owner} --dagshub-repo {repo_name}
    deps:
    - run_complete_pipeline.py
    - data/features/X_train_features.npy
    - data/features/X_test_features.npy
    - data/processed/y_train.npy
    - data/processed/y_test.npy
    outs:
    - models/trained/best_model.joblib
    - reports/results_summary.json
    metrics:
    - reports/metrics.json
"""
    
    with open("dvc.yaml", 'w') as f:
        f.write(dvc_yaml_content.strip())
    
    print("✅ DVC pipeline configuration created")
    
    # Update the DagsHub integration file
    integration_code = f'''
# Auto-generated DagsHub configuration
REPO_OWNER = "{repo_owner}"
REPO_NAME = "{repo_name}"
REPO_URL = "https://dagshub.com/{repo_owner}/{repo_name}"

# Initialize DagsHub tracker
from src.utils.dagshub_integration import DagsHubTracker
tracker = DagsHubTracker(REPO_OWNER, REPO_NAME)
'''
    
    with open("dagshub_config.py", 'w') as f:
        f.write(integration_code)
    
    print("✅ DagsHub integration code created")
    
    print(f"""
🎉 DagsHub Setup Complete!
========================

Next Steps:
1. Create repository on DagsHub:
   👉 Go to: https://dagshub.com/repo/create
   👉 Repository name: {repo_name}
   👉 Make it public (free tier)

2. Add DagsHub as git remote:
   git remote add dagshub https://dagshub.com/{repo_owner}/{repo_name}.git

3. Push your code:
   git add .
   git commit -m "Setup DagsHub integration"
   git push dagshub main

4. Run the DVC pipeline:
   dvc repro

5. Push data to DagsHub:
   dvc push

6. View your project:
   👉 Repository: https://dagshub.com/{repo_owner}/{repo_name}
   👉 Data: https://dagshub.com/{repo_owner}/{repo_name}/data
   👉 Experiments: https://dagshub.com/{repo_owner}/{repo_name}/experiments

Files created:
- .dagshub_config.json (DagsHub configuration)
- dagshub_config.py (Integration code)
- dvc.yaml (DVC pipeline)
- .gitignore (Updated)

Happy experimenting! 🚀
""")

if __name__ == "__main__":
    setup_dagshub_repo()