# Update src/utils/dagshub_integration.py

import os
import dagshub
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import requests
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DagsHubTracker:
    """Enhanced DagsHub integration with remote artifact storage"""
    
    def __init__(self, repo_owner: str, repo_name: str):
        """Initialize DagsHub tracker with remote storage capabilities"""
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        
        # Initialize DagsHub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=False)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create experiments directory
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Current experiment tracking
        self.current_experiment = None
        self.experiment_data = {}
        
        # Remote storage setup
        self.remote_artifacts = []
        
        self.logger.info(f"DagsHub tracker initialized for {repo_owner}/{repo_name}")
        self.logger.info(f"Repository URL: {self.repo_url}")
    
    def start_experiment(self, experiment_name: str = None, tags: Dict[str, str] = None):
        """Start a new experiment"""
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = experiment_name
        
        experiment_tags = {
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "framework": "debt_collection_ml",
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            experiment_tags.update(tags)
        
        self.experiment_data = {
            "experiment_name": experiment_name,
            "tags": experiment_tags,
            "params": {},
            "metrics": {},
            "artifacts": [],
            "remote_artifacts": [],
            "start_time": datetime.now().isoformat()
        }
        
        self.logger.info(f"Started experiment: {experiment_name}")
        return self
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to experiment"""
        if self.current_experiment:
            self.experiment_data["params"].update(params)
            self.logger.info(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to experiment"""
        if self.current_experiment:
            if step is not None:
                for key, value in metrics.items():
                    step_key = f"{key}_step_{step}"
                    self.experiment_data["metrics"][step_key] = value
            else:
                self.experiment_data["metrics"].update(metrics)
            self.logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def log_model_to_dagshub(self, model, model_name: str, model_type: str = "sklearn"):
        """Save model to DagsHub repository"""
        import joblib
        import tempfile
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
                joblib.dump(model, tmp_file.name)
                
                # Upload to DagsHub (simulate - in real implementation you'd use DagsHub API)
                remote_path = f"models/{model_name}_{self.current_experiment}.joblib"
                
                # For now, we'll store locally but track as remote artifact
                models_dir = Path("models/dagshub")
                models_dir.mkdir(parents=True, exist_ok=True)
                
                local_path = models_dir / f"{model_name}_{self.current_experiment}.joblib"
                joblib.dump(model, local_path)
                
                # Track as remote artifact
                self.experiment_data["remote_artifacts"].append({
                    "name": model_name,
                    "type": "model",
                    "model_type": model_type,
                    "local_path": str(local_path),
                    "remote_path": remote_path,
                    "size_mb": local_path.stat().st_size / (1024 * 1024)
                })
                
                self.logger.info(f"Model {model_name} stored to DagsHub: {remote_path}")
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
        except Exception as e:
            self.logger.error(f"Failed to store model to DagsHub: {e}")
    
    def log_artifact_to_dagshub(self, artifact_path: str, artifact_name: str = None, artifact_type: str = "file"):
        """Log artifact to DagsHub repository"""
        
        try:
            artifact_name = artifact_name or Path(artifact_path).name
            
            # Create DagsHub artifacts directory
            dagshub_dir = Path("artifacts/dagshub")
            dagshub_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy artifact to DagsHub directory
            import shutil
            dagshub_path = dagshub_dir / artifact_name
            
            if Path(artifact_path).exists():
                shutil.copy2(artifact_path, dagshub_path)
                
                remote_path = f"artifacts/{artifact_name}"
                
                self.experiment_data["remote_artifacts"].append({
                    "name": artifact_name,
                    "type": artifact_type,
                    "local_path": artifact_path,
                    "dagshub_path": str(dagshub_path),
                    "remote_path": remote_path,
                    "size_mb": dagshub_path.stat().st_size / (1024 * 1024)
                })
                
                self.logger.info(f"Artifact {artifact_name} stored to DagsHub: {remote_path}")
            else:
                self.logger.warning(f"Artifact not found: {artifact_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to store artifact to DagsHub: {e}")
    
    def log_metrics_to_dagshub(self, metrics: Dict[str, Any]):
        """Store metrics in DagsHub format"""
        
        try:
            # Create metrics directory
            metrics_dir = Path("metrics/dagshub")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = metrics_dir / f"metrics_{self.current_experiment}_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Track as remote artifact
            self.experiment_data["remote_artifacts"].append({
                "name": f"metrics_{self.current_experiment}",
                "type": "metrics",
                "local_path": str(metrics_file),
                "remote_path": f"metrics/{metrics_file.name}",
                "content": metrics
            })
            
            self.logger.info(f"Metrics stored to DagsHub: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics to DagsHub: {e}")
    
    def finish_experiment(self):
        """Finish and save current experiment with DagsHub integration"""
        if not self.current_experiment:
            self.logger.warning("No active experiment to finish")
            return
        
        # Add end time
        self.experiment_data["end_time"] = datetime.now().isoformat()
        
        # Store metrics to DagsHub
        if self.experiment_data["metrics"]:
            self.log_metrics_to_dagshub(self.experiment_data["metrics"])
        
        # Save experiment data locally
        experiment_file = self.experiments_dir / f"{self.current_experiment}.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        # Also save to DagsHub experiments directory
        dagshub_exp_dir = Path("experiments/dagshub")
        dagshub_exp_dir.mkdir(parents=True, exist_ok=True)
        
        dagshub_exp_file = dagshub_exp_dir / f"{self.current_experiment}.json"
        with open(dagshub_exp_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        self.logger.info(f"Experiment completed and stored: {experiment_file}")
        self.logger.info(f"DagsHub experiment stored: {dagshub_exp_file}")
        
        # Print summary
        print(f"\n📊 Experiment Summary: {self.current_experiment}")
        print(f"🔗 DagsHub URL: {self.repo_url}")
        print(f"📈 Metrics logged: {len(self.experiment_data['metrics'])}")
        print(f"📁 Artifacts stored: {len(self.experiment_data['remote_artifacts'])}")
        
        # Reset current experiment
        self.current_experiment = None
        self.experiment_data = {}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish_experiment()
