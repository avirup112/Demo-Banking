import os
import yaml
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class DVCManager:
    """DVC integration for data versioning and pipeline management"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dvc_dir = self.project_root / ".dvc"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def init_dvc(self, remote_url: Optional[str] = None):
        """Initialize DVC in the project"""
        
        try:
            # Initialize DVC
            result = subprocess.run(
                ["dvc", "init"], 
                cwd=self.project_root, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("DVC initialized successfully")
            else:
                self.logger.warning(f"DVC init warning: {result.stderr}")
            
            # Add remote if provided
            if remote_url:
                self.add_remote("origin", remote_url)
            
            # Create .dvcignore
            self._create_dvcignore()
            
        except FileNotFoundError:
            self.logger.error("DVC not installed. Install with: pip install dvc")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize DVC: {e}")
            raise
    
    def add_remote(self, name: str, url: str):
        """Add DVC remote storage"""
        
        try:
            result = subprocess.run(
                ["dvc", "remote", "add", "-d", name, url],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"DVC remote '{name}' added: {url}")
            else:
                self.logger.error(f"Failed to add remote: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error adding DVC remote: {e}")
    
    def add_data(self, data_path: str, commit_message: str = None):
        """Add data file to DVC tracking"""
        
        try:
            # Add to DVC
            result = subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Added {data_path} to DVC tracking")
                
                # Add .dvc file to git
                dvc_file = f"{data_path}.dvc"
                subprocess.run(["git", "add", dvc_file], cwd=self.project_root)
                
                # Add to .gitignore if not already there
                gitignore_path = self.project_root / ".gitignore"
                self._add_to_gitignore(data_path, gitignore_path)
                
                if commit_message:
                    subprocess.run(
                        ["git", "commit", "-m", commit_message],
                        cwd=self.project_root
                    )
                
            else:
                self.logger.error(f"Failed to add data to DVC: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error adding data to DVC: {e}")
    
    def create_pipeline(self, pipeline_config: Dict[str, Any]):
        """Create DVC pipeline from configuration"""
        
        pipeline_file = self.project_root / "dvc.yaml"
        
        try:
            with open(pipeline_file, 'w') as f:
                yaml.dump(pipeline_config, f, default_flow_style=False)
            
            self.logger.info(f"DVC pipeline created: {pipeline_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create DVC pipeline: {e}")
    
    def run_pipeline(self, stage: str = None):
        """Run DVC pipeline"""
        
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("DVC pipeline executed successfully")
                return True
            else:
                self.logger.error(f"Pipeline execution failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running DVC pipeline: {e}")
            return False
    
    def push_data(self):
        """Push data to remote storage"""
        
        try:
            result = subprocess.run(
                ["dvc", "push"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Data pushed to remote storage")
            else:
                self.logger.error(f"Failed to push data: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error pushing data: {e}")
    
    def pull_data(self):
        """Pull data from remote storage"""
        
        try:
            result = subprocess.run(
                ["dvc", "pull"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Data pulled from remote storage")
            else:
                self.logger.error(f"Failed to pull data: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error pulling data: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get DVC pipeline status"""
        
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"status": "up_to_date", "output": result.stdout}
            else:
                return {"status": "changes_detected", "output": result.stdout}
                
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            return {"status": "error", "output": str(e)}
    
    def _create_dvcignore(self):
        """Create .dvcignore file"""
        
        dvcignore_content = """
# DVC ignore file
*.pyc
__pycache__/
.git/
.pytest_cache/
.coverage
*.log
.DS_Store
Thumbs.db
"""
        
        dvcignore_path = self.project_root / ".dvcignore"
        with open(dvcignore_path, 'w') as f:
            f.write(dvcignore_content.strip())
    
    def _add_to_gitignore(self, path: str, gitignore_path: Path):
        """Add path to .gitignore if not already present"""
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            if path not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n{path}\n")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(f"{path}\n")

class DVCPipelineBuilder:
    """Builder for creating DVC pipelines"""
    
    def __init__(self):
        self.stages = {}
        self.logger = logging.getLogger(__name__)
    
    def add_stage(self, name: str, cmd: str, deps: List[str] = None, 
                  outs: List[str] = None, params: List[str] = None,
                  metrics: List[str] = None, plots: List[str] = None):
        """Add a stage to the pipeline"""
        
        stage = {
            "cmd": cmd
        }
        
        if deps:
            stage["deps"] = deps
        if outs:
            stage["outs"] = outs
        if params:
            stage["params"] = params
        if metrics:
            stage["metrics"] = metrics
        if plots:
            stage["plots"] = plots
        
        self.stages[name] = stage
        self.logger.info(f"Added stage '{name}' to pipeline")
    
    def build(self) -> Dict[str, Any]:
        """Build the complete pipeline configuration"""
        
        return {
            "stages": self.stages
        }

def create_debt_collection_pipeline() -> Dict[str, Any]:
    """Create DVC pipeline for debt collection ML project"""
    
    builder = DVCPipelineBuilder()
    
    # Data generation stage
    builder.add_stage(
        name="generate_data",
        cmd="python scripts/generate_data.py",
        outs=["data/raw/debt_collection_data.csv"],
        params=["params.yaml:generate_data"]
    )
    
    # Data preprocessing stage
    builder.add_stage(
        name="preprocess_data",
        cmd="python scripts/preprocess_data.py",
        deps=["data/raw/debt_collection_data.csv", "src/data/data_preprocessor.py"],
        outs=["data/processed/X_processed.npy", "data/processed/y_encoded.npy"],
        params=["params.yaml:preprocess"]
    )
    
    # Feature engineering stage
    builder.add_stage(
        name="feature_engineering",
        cmd="python scripts/feature_engineering.py",
        deps=["data/processed/X_processed.npy", "data/processed/y_encoded.npy"],
        outs=["data/processed/X_engineered.npy"],
        params=["params.yaml:feature_engineering"]
    )
    
    # Model training stage
    builder.add_stage(
        name="train_models",
        cmd="python scripts/train_model_pipeline.py --optimize",
        deps=[
            "data/processed/X_engineered.npy",
            "data/processed/y_encoded.npy",
            "src/models/ml_model.py"
        ],
        outs=["models/trained/"],
        metrics=["reports/model_metrics.json"],
        plots=["reports/model_comparison.json"],
        params=["params.yaml:training"]
    )
    
    # Model evaluation stage
    builder.add_stage(
        name="evaluate_models",
        cmd="python scripts/evaluate_models.py",
        deps=["models/trained/", "data/processed/X_engineered.npy"],
        metrics=["reports/evaluation_metrics.json"],
        plots=["reports/evaluation_plots/"]
    )
    
    return builder.build()

def setup_dvc_project(project_root: str = ".", remote_url: str = None):
    """Complete DVC project setup"""
    
    dvc_manager = DVCManager(project_root)
    
    # Initialize DVC
    dvc_manager.init_dvc(remote_url)
    
    # Create pipeline
    pipeline_config = create_debt_collection_pipeline()
    dvc_manager.create_pipeline(pipeline_config)
    
    # Create params.yaml
    create_params_file(project_root)
    
    # Create DVC-specific scripts
    create_dvc_scripts(project_root)
    
    print(f"""
DVC Setup Complete!
==================

Files created:
- .dvc/ (DVC configuration)
- dvc.yaml (Pipeline definition)
- params.yaml (Parameters file)
- .dvcignore (DVC ignore file)

Next steps:
1. Add data to DVC tracking:
   dvc add data/raw/debt_collection_data.csv

2. Run the pipeline:
   dvc repro

3. Push data to remote (if configured):
   dvc push

4. View pipeline:
   dvc dag

5. Compare experiments:
   dvc plots show
   dvc metrics show
""")

def create_params_file(project_root: str):
    """Create parameters file for DVC pipeline"""
    
    params = {
        "generate_data": {
            "n_samples": 10000,
            "random_state": 42
        },
        "preprocess": {
            "imputation_strategy": "knn",
            "scaling_method": "standard",
            "encoding_method": "onehot",
            "handle_outliers": True,
            "outlier_method": "iqr"
        },
        "feature_engineering": {
            "include_polynomial": True,
            "polynomial_degree": 2,
            "include_pca": False,
            "feature_selection": True,
            "selection_k": 50
        },
        "training": {
            "optimize_hyperparameters": True,
            "n_trials": 50,
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42
        }
    }
    
    params_file = Path(project_root) / "params.yaml"
    with open(params_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

def create_dvc_scripts(project_root: str):
    """Create DVC-specific scripts"""
    
    scripts_dir = Path(project_root) / "scripts"
    
    # Generate data script
    generate_data_script = """#!/usr/bin/env python3
import sys
import os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_generator import DebtCollectionDataGenerator

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    gen_params = params['generate_data']
    
    # Generate data
    generator = DebtCollectionDataGenerator(
        n_samples=gen_params['n_samples'],
        random_state=gen_params['random_state']
    )
    
    df = generator.generate_dataset()
    
    # Save data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/debt_collection_data.csv', index=False)
    
    print(f"Generated {len(df)} samples")

if __name__ == "__main__":
    main()
"""
    
    with open(scripts_dir / "generate_data.py", 'w') as f:
        f.write(generate_data_script)
    
    # Preprocess data script
    preprocess_script = """#!/usr/bin/env python3
import sys
import os
import yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_preprocessor import AdvancedDataPreprocessor

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    preprocess_params = params['preprocess']
    
    # Load data
    df = pd.read_csv('data/raw/debt_collection_data.csv')
    
    # Initialize preprocessor
    preprocessor = AdvancedDataPreprocessor(**preprocess_params)
    
    # Preprocess data
    X_processed, y_encoded = preprocessor.fit_transform(df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_processed.npy', X_processed)
    np.save('data/processed/y_encoded.npy', y_encoded)
    
    # Save preprocessor
    os.makedirs('models/artifacts', exist_ok=True)
    preprocessor.save_preprocessor('models/artifacts/preprocessor.joblib')
    
    print(f"Preprocessed data shape: {X_processed.shape}")

if __name__ == "__main__":
    main()
"""
    
    with open(scripts_dir / "preprocess_data.py", 'w') as f:
        f.write(preprocess_script)
    
    # Feature engineering script
    feature_engineering_script = """#!/usr/bin/env python3
import sys
import os
import yaml
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_engineering import AdvancedFeatureEngineer
from data.data_preprocessor import AdvancedDataPreprocessor

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    fe_params = params['feature_engineering']
    
    # Load processed data
    X_processed = np.load('data/processed/X_processed.npy')
    y_encoded = np.load('data/processed/y_encoded.npy')
    
    # Load preprocessor to get feature names
    preprocessor = AdvancedDataPreprocessor()
    preprocessor.load_preprocessor('models/artifacts/preprocessor.joblib')
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(X_processed, columns=preprocessor.feature_names)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer(**fe_params)
    
    # Engineer features
    X_engineered = feature_engineer.fit_transform(feature_df, y_encoded)
    
    # Save engineered features
    np.save('data/processed/X_engineered.npy', X_engineered)
    
    # Save feature engineer
    feature_engineer.save_feature_engineer('models/artifacts/feature_engineer.joblib')
    
    print(f"Engineered features shape: {X_engineered.shape}")

if __name__ == "__main__":
    main()
"""
    
    with open(scripts_dir / "feature_engineering.py", 'w') as f:
        f.write(feature_engineering_script)
    
    print("DVC scripts created successfully!")

# Example usage
if __name__ == "__main__":
    # Setup DVC project
    setup_dvc_project()