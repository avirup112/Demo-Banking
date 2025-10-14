#!/usr/bin/env python3
"""
Setup script for DVC integration
This script helps you set up DVC for data versioning and pipeline management
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.dvc_integration import setup_dvc_project, DVCManager

def main():
    parser = argparse.ArgumentParser(description='Setup DVC for debt collection ML project')
    parser.add_argument('--remote-url', help='DVC remote storage URL (optional)')
    parser.add_argument('--remote-type', choices=['s3', 'gcs', 'azure', 'ssh', 'local'], 
                       default='local', help='Type of remote storage')
    parser.add_argument('--init-only', action='store_true', help='Only initialize DVC without pipeline')
    
    args = parser.parse_args()
    
    print("Setting up DVC for debt collection ML project...")
    
    if args.init_only:
        # Just initialize DVC
        dvc_manager = DVCManager()
        dvc_manager.init_dvc(args.remote_url)
        print("DVC initialized successfully!")
        return
    
    # Full setup with pipeline
    setup_dvc_project(remote_url=args.remote_url)
    
    # Additional setup based on remote type
    if args.remote_type and args.remote_url:
        print(f"\nRemote storage configured: {args.remote_type}")
        
        if args.remote_type == 's3':
            print("""
AWS S3 Setup:
- Ensure AWS credentials are configured
- Install: pip install 'dvc[s3]'
- Example URL: s3://my-bucket/dvc-storage
""")
        elif args.remote_type == 'gcs':
            print("""
Google Cloud Storage Setup:
- Ensure GCS credentials are configured
- Install: pip install 'dvc[gs]'
- Example URL: gs://my-bucket/dvc-storage
""")
        elif args.remote_type == 'azure':
            print("""
Azure Blob Storage Setup:
- Ensure Azure credentials are configured
- Install: pip install 'dvc[azure]'
- Example URL: azure://container/path
""")
        elif args.remote_type == 'ssh':
            print("""
SSH Remote Setup:
- Ensure SSH access is configured
- Install: pip install 'dvc[ssh]'
- Example URL: ssh://user@host/path/to/dvc-storage
""")
        elif args.remote_type == 'local':
            print("""
Local Remote Setup:
- Using local directory for remote storage
- Good for testing and development
""")
    
    # Create example workflows
    create_example_workflows()
    
    print(f"""
DVC Setup Complete!
==================

Quick Start Commands:
1. Add data to tracking:
   dvc add data/raw/debt_collection_data.csv

2. Run full pipeline:
   dvc repro

3. View pipeline DAG:
   dvc dag

4. Check pipeline status:
   dvc status

5. Compare experiments:
   dvc params diff
   dvc metrics diff

6. Push data to remote:
   dvc push

7. Pull data from remote:
   dvc pull

Files Created:
- dvc.yaml (Pipeline definition)
- params.yaml (Parameters)
- .dvcignore (DVC ignore rules)
- scripts/generate_data.py
- scripts/preprocess_data.py  
- scripts/feature_engineering.py

Next Steps:
1. Generate initial data: python scripts/generate_data.py
2. Run the pipeline: dvc repro
3. Commit changes: git add . && git commit -m "Add DVC pipeline"
""")

def create_example_workflows():
    """Create example workflow scripts"""
    
    # Create Makefile for common DVC operations
    makefile_content = """
# DVC Makefile for Debt Collection ML

.PHONY: data preprocess features train evaluate pipeline clean status

# Generate data
data:
	dvc repro generate_data

# Preprocess data
preprocess:
	dvc repro preprocess_data

# Engineer features
features:
	dvc repro feature_engineering

# Train models
train:
	dvc repro train_models

# Evaluate models
evaluate:
	dvc repro evaluate_models

# Run full pipeline
pipeline:
	dvc repro

# Check status
status:
	dvc status
	dvc metrics show
	dvc params diff

# Clean cache
clean:
	dvc cache dir
	dvc gc -w

# Push to remote
push:
	dvc push

# Pull from remote
pull:
	dvc pull

# Show DAG
dag:
	dvc dag

# Compare experiments
compare:
	dvc params diff
	dvc metrics diff
	dvc plots diff

# Setup remote (example)
setup-remote-s3:
	dvc remote add -d myremote s3://my-bucket/dvc-storage
	dvc remote modify myremote region us-east-1

setup-remote-local:
	dvc remote add -d myremote /tmp/dvc-storage
"""
    
    with open('Makefile', 'w') as f:
        f.write(makefile_content.strip())
    
    # Create experiment tracking script
    experiment_script = """#!/usr/bin/env python3
\"\"\"
Experiment tracking with DVC
\"\"\"

import os
import json
import yaml
import subprocess
from datetime import datetime

def run_experiment(experiment_name, params_override=None):
    \"\"\"Run a DVC experiment with parameter overrides\"\"\"
    
    print(f"Running experiment: {experiment_name}")
    
    # Create experiment directory
    exp_dir = f"experiments/{experiment_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Override parameters if provided
    if params_override:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        # Update parameters
        for key, value in params_override.items():
            if '.' in key:
                section, param = key.split('.', 1)
                if section in params:
                    params[section][param] = value
            else:
                params[key] = value
        
        # Save experiment parameters
        exp_params_file = f"{exp_dir}/params.yaml"
        with open(exp_params_file, 'w') as f:
            yaml.dump(params, f)
        
        # Copy to main params file
        with open('params.yaml', 'w') as f:
            yaml.dump(params, f)
    
    # Run pipeline
    result = subprocess.run(['dvc', 'repro'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Experiment completed successfully!")
        
        # Save experiment metadata
        metadata = {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'params_override': params_override or {},
            'status': 'success'
        }
        
        with open(f"{exp_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Copy results
        os.system(f"cp -r reports/ {exp_dir}/")
        os.system(f"cp -r models/trained/ {exp_dir}/models/")
        
    else:
        print(f"Experiment failed: {result.stderr}")

def compare_experiments():
    \"\"\"Compare multiple experiments\"\"\"
    
    subprocess.run(['dvc', 'metrics', 'show'])
    subprocess.run(['dvc', 'params', 'diff'])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--param', action='append', help='Parameter override (key=value)')
    
    args = parser.parse_args()
    
    # Parse parameter overrides
    params_override = {}
    if args.param:
        for param in args.param:
            key, value = param.split('=', 1)
            try:
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif '.' in value and value.replace('.', '').isdigit():
                    value = float(value)
            except:
                pass  # Keep as string
            
            params_override[key] = value
    
    run_experiment(args.name, params_override)
"""
    
    os.makedirs('scripts', exist_ok=True)
    with open('scripts/run_experiment.py', 'w') as f:
        f.write(experiment_script)
    
    # Make scripts executable
    os.chmod('scripts/run_experiment.py', 0o755)
    
    print("Example workflows created:")
    print("- Makefile (common DVC commands)")
    print("- scripts/run_experiment.py (experiment tracking)")

if __name__ == "__main__":
    main()