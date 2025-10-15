#!/usr/bin/env python3
"""
Enhanced DagsHub Integration with MLflow-compatible Model Registry
Comprehensive experiment tracking, model versioning, and deployment workflows
"""

import os
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import requests
import base64
import joblib
import tempfile
import shutil
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyContextManager:
    """Dummy context manager for when MLflow runs fail"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class EnhancedDagsHubTracker:
    """Enhanced DagsHub integration with MLflow-compatible model registry and comprehensive tracking"""
    
    def __init__(self, repo_owner: str, repo_name: str, experiment_name: str = "debt_collection_ml"):
        """
        Initialize enhanced DagsHub tracker with MLflow integration
        
        Args:
            repo_owner: DagsHub repository owner
            repo_name: DagsHub repository name  
            experiment_name: MLflow experiment name
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        self.experiment_name = experiment_name
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Check for DagsHub configuration
        config_file = Path(".dagshub/config.json")
        self.dagshub_configured = False
        
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                if config.get('setup_complete', False):
                    # Use configured DagsHub settings
                    mlflow_uri = config.get('mlflow_uri', f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow/")
                    
                    # Initialize DagsHub
                    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
                    mlflow.set_tracking_uri(mlflow_uri)
                    
                    self.dagshub_configured = True
                    self.logger.info(f"DagsHub integration loaded from config")
                else:
                    raise Exception("Configuration incomplete")
                    
            except Exception as e:
                self.logger.warning(f"DagsHub config invalid: {e}")
                self.dagshub_configured = False
        
        if not self.dagshub_configured:
            self.logger.warning("DagsHub not configured. Run: python setup_dagshub_simple.py")
            # Fallback to local MLflow
            mlflow_uri = "file:./mlruns"
            mlflow.set_tracking_uri(mlflow_uri)
        else:
            # Test DagsHub connection and fallback if needed
            try:
                # Quick test of DagsHub MLflow connection
                client = MlflowClient()
                experiments = client.search_experiments(max_results=1)
                self.logger.info("✅ DagsHub MLflow connection verified")
            except Exception as e:
                self.logger.warning(f"DagsHub MLflow connection failed: {e}")
                self.logger.info("🔄 Falling back to local MLflow tracking")
                mlflow_uri = "file:./mlruns"
                mlflow.set_tracking_uri(mlflow_uri)
                self.dagshub_configured = False
        
        # Initialize MLflow client
        self.mlflow_client = MlflowClient()
        
        # Setup experiment
        self._setup_experiment()
        
        # Create directory structure
        self._create_directory_structure()
        
        # Current run tracking
        self.current_run = None
        self.active_runs = {}
        
        # Model registry tracking
        self.registered_models = {}
        self.model_versions = {}
        
        self.logger.info(f"Enhanced DagsHub tracker initialized for {repo_owner}/{repo_name}")
        self.logger.info(f"MLflow tracking URI: {mlflow_uri}")
        self.logger.info(f"Repository URL: {self.repo_url}")
    
    def _setup_experiment(self):
        """Setup MLflow experiment with fallback for offline mode"""
        try:
            # Set local MLflow tracking for testing
            if not mlflow.get_tracking_uri() or "dagshub" not in mlflow.get_tracking_uri():
                mlflow.set_tracking_uri("file:./mlruns")
                self.logger.info("Using local MLflow tracking for testing")
            
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags={
                        "project": "debt_collection_ml",
                        "framework": "scikit-learn",
                        "repo_owner": self.repo_owner,
                        "repo_name": self.repo_name
                    }
                )
                self.experiment_id = experiment_id
                self.logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                self.logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
            # Set active experiment
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            self.logger.warning(f"MLflow experiment setup failed: {e}")
            self.experiment_id = None
    
    def _create_directory_structure(self):
        """Create comprehensive directory structure for DagsHub integration"""
        directories = [
            "experiments/dagshub",
            "models/dagshub/registry", 
            "models/dagshub/versions",
            "models/dagshub/staging",
            "models/dagshub/production",
            "artifacts/dagshub/experiments",
            "artifacts/dagshub/models",
            "artifacts/dagshub/reports",
            "metrics/dagshub/experiments",
            "metrics/dagshub/models",
            "logs/dagshub"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DagsHub directory structure created")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None, nested: bool = False):
        """
        Start a new MLflow run with comprehensive tracking
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            nested: Whether this is a nested run
            
        Returns:
            MLflow run context manager
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare tags
        run_tags = {
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "framework": "debt_collection_ml",
            "dagshub_url": self.repo_url,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            run_tags.update(tags)
        
        try:
            # End any existing run first if not nested
            if not nested and mlflow.active_run():
                self.logger.info("Ending existing MLflow run before starting new one")
                mlflow.end_run()
            
            # Start MLflow run
            run = mlflow.start_run(
                run_name=run_name,
                tags=run_tags,
                nested=nested
            )
            
            self.current_run = run
            self.active_runs[run_name] = run
            
            self.logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
            
            return run
            
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            # Try to end any existing run and return dummy context manager
            try:
                if mlflow.active_run():
                    mlflow.end_run()
            except:
                pass
            return DummyContextManager()
    
    def start_experiment(self, experiment_name: str = None, tags: Dict[str, str] = None):
        """Legacy method - redirects to start_run for backward compatibility"""
        try:
            # End any existing run first
            if mlflow.active_run():
                mlflow.end_run()
            
            run = self.start_run(run_name=experiment_name, tags=tags)
            if run is None:
                # Return a dummy context manager if run creation fails
                return DummyContextManager()
            return run
        except Exception as e:
            self.logger.error(f"Failed to start experiment: {e}")
            return DummyContextManager()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow run"""
        try:
            # Convert complex objects to strings
            clean_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    clean_params[key] = json.dumps(value)
                elif isinstance(value, (np.ndarray, pd.DataFrame)):
                    clean_params[key] = str(type(value).__name__)
                else:
                    clean_params[key] = str(value)
            
            mlflow.log_params(clean_params)
            self.logger.info(f"Logged parameters: {list(clean_params.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow run"""
        try:
            # Clean metrics - ensure all values are numeric
            clean_metrics = {}
            for key, value in metrics.items():
                try:
                    clean_metrics[key] = float(value)
                except (ValueError, TypeError):
                    self.logger.warning(f"Skipping non-numeric metric: {key} = {value}")
            
            if step is not None:
                for key, value in clean_metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(clean_metrics)
                
            self.logger.info(f"Logged metrics: {list(clean_metrics.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact to MLflow run"""
        try:
            if Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path, artifact_name)
                self.logger.info(f"Logged artifact: {artifact_path}")
            else:
                self.logger.warning(f"Artifact not found: {artifact_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to log artifact: {e}")
    
    def log_model_mlflow(self, model: Any, model_name: str, 
                        signature: Any = None, input_example: Any = None,
                        registered_model_name: str = None) -> str:
        """
        Log model to MLflow with comprehensive metadata
        
        Args:
            model: Trained model object
            model_name: Name for the model
            signature: MLflow model signature
            input_example: Example input for the model
            registered_model_name: Name for model registry
            
        Returns:
            Model URI
        """
        try:
            # Log model to MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
            
            # Store model metadata
            model_metadata = {
                "model_name": model_name,
                "model_uri": model_info.model_uri,
                "run_id": mlflow.active_run().info.run_id,
                "timestamp": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "registered_name": registered_model_name
            }
            
            # Save metadata locally
            metadata_file = Path(f"models/dagshub/registry/{model_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            self.logger.info(f"Model logged to MLflow: {model_name}")
            self.logger.info(f"Model URI: {model_info.model_uri}")
            
            return model_info.model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to log model to MLflow: {e}")
            return None
    
    def register_model(self, model_name: str, model_uri: str, 
                      description: str = None, tags: Dict[str, str] = None) -> str:
        """
        Register model in MLflow model registry
        
        Args:
            model_name: Name for registered model
            model_uri: URI of the model to register
            description: Model description
            tags: Model tags
            
        Returns:
            Model version
        """
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update model description if provided
            if description:
                self.mlflow_client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            # Store registration info
            self.registered_models[model_name] = {
                "version": model_version.version,
                "uri": model_uri,
                "description": description,
                "tags": tags or {},
                "registered_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Model registered: {model_name} v{model_version.version}")
            
            return model_version.version
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return None
    
    def transition_model_stage(self, model_name: str, version: str, stage: str) -> bool:
        """
        Transition model to different stage (Staging, Production, Archived)
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
            
        Returns:
            Success status
        """
        try:
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            # Create stage-specific directory and copy model
            stage_dir = Path(f"models/dagshub/{stage.lower()}")
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            # Log stage transition
            stage_log = {
                "model_name": model_name,
                "version": version,
                "stage": stage,
                "transitioned_at": datetime.now().isoformat(),
                "transitioned_by": "automated_pipeline"
            }
            
            stage_file = stage_dir / f"{model_name}_v{version}_stage.json"
            with open(stage_file, 'w') as f:
                json.dump(stage_log, f, indent=2)
            
            self.logger.info(f"Model {model_name} v{version} transitioned to {stage}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model
        
        Args:
            model_name: Registered model name
            
        Returns:
            List of model version information
        """
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            version_info = []
            for version in versions:
                version_info.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description,
                    "tags": version.tags
                })
            
            return version_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model versions: {e}")
            return []
    
    def compare_model_versions(self, model_name: str, 
                              version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model
        
        Args:
            model_name: Registered model name
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        try:
            # Get model version details
            v1_details = self.mlflow_client.get_model_version(model_name, version1)
            v2_details = self.mlflow_client.get_model_version(model_name, version2)
            
            # Get run metrics for comparison
            v1_run = self.mlflow_client.get_run(v1_details.run_id)
            v2_run = self.mlflow_client.get_run(v2_details.run_id)
            
            comparison = {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "stage": v1_details.current_stage,
                    "metrics": v1_run.data.metrics,
                    "params": v1_run.data.params,
                    "creation_time": v1_details.creation_timestamp
                },
                "version2": {
                    "version": version2,
                    "stage": v2_details.current_stage,
                    "metrics": v2_run.data.metrics,
                    "params": v2_run.data.params,
                    "creation_time": v2_details.creation_timestamp
                },
                "comparison_timestamp": datetime.now().isoformat()
            }
            
            # Calculate metric differences
            metric_diffs = {}
            for metric in v1_run.data.metrics:
                if metric in v2_run.data.metrics:
                    diff = v2_run.data.metrics[metric] - v1_run.data.metrics[metric]
                    metric_diffs[metric] = {
                        "v1_value": v1_run.data.metrics[metric],
                        "v2_value": v2_run.data.metrics[metric],
                        "difference": diff,
                        "improvement": diff > 0
                    }
            
            comparison["metric_differences"] = metric_diffs
            
            # Save comparison
            comparison_file = Path(f"models/dagshub/registry/{model_name}_comparison_v{version1}_v{version2}.json")
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            self.logger.info(f"Model comparison completed: {model_name} v{version1} vs v{version2}")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare model versions: {e}")
            return {}
    
    def log_model_to_dagshub(self, model, model_name: str, model_type: str = "sklearn"):
        """Enhanced model logging with registry integration"""
        try:
            # Log model to MLflow first
            model_uri = self.log_model_mlflow(
                model=model,
                model_name=model_name,
                registered_model_name=f"{model_name}_registered"
            )
            
            if model_uri:
                # Register model in registry
                version = self.register_model(
                    model_name=f"{model_name}_registered",
                    model_uri=model_uri,
                    description=f"Debt collection ML model - {model_type}",
                    tags={"model_type": model_type, "framework": "sklearn"}
                )
                
                # Save local copy for backup
                models_dir = Path("models/dagshub/versions")
                models_dir.mkdir(parents=True, exist_ok=True)
                
                local_path = models_dir / f"{model_name}_v{version}.joblib"
                joblib.dump(model, local_path)
                
                self.logger.info(f"Model {model_name} logged and registered successfully")
                
                return model_uri, version
            
        except Exception as e:
            self.logger.error(f"Failed to log model to DagsHub: {e}")
            return None, None
    
    def log_artifact_to_dagshub(self, artifact_path: str, artifact_name: str = None, artifact_type: str = "file"):
        """Enhanced artifact logging with MLflow integration"""
        try:
            artifact_name = artifact_name or Path(artifact_path).name
            
            # Log to MLflow
            self.log_artifact(artifact_path, artifact_name)
            
            # Create DagsHub artifacts directory
            dagshub_dir = Path(f"artifacts/dagshub/{artifact_type}")
            dagshub_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy artifact to DagsHub directory
            dagshub_path = dagshub_dir / artifact_name
            
            if Path(artifact_path).exists():
                shutil.copy2(artifact_path, dagshub_path)
                
                # Create artifact metadata
                artifact_metadata = {
                    "name": artifact_name,
                    "type": artifact_type,
                    "original_path": artifact_path,
                    "dagshub_path": str(dagshub_path),
                    "size_mb": dagshub_path.stat().st_size / (1024 * 1024),
                    "created_at": datetime.now().isoformat(),
                    "run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None
                }
                
                # Save metadata
                metadata_file = dagshub_path.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(artifact_metadata, f, indent=2)
                
                self.logger.info(f"Artifact {artifact_name} stored to DagsHub: {dagshub_path}")
            else:
                self.logger.warning(f"Artifact not found: {artifact_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to store artifact to DagsHub: {e}")
    
    def log_metrics_to_dagshub(self, metrics: Dict[str, Any]):
        """Enhanced metrics logging with MLflow integration"""
        try:
            # Log to MLflow first
            self.log_metrics(metrics)
            
            # Create metrics directory
            metrics_dir = Path("metrics/dagshub/experiments")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "no_run"
            metrics_file = metrics_dir / f"metrics_{run_id}_{timestamp}.json"
            
            # Enhanced metrics with metadata
            enhanced_metrics = {
                "metrics": metrics,
                "metadata": {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "repo_url": self.repo_url,
                    "experiment_name": self.experiment_name
                }
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(enhanced_metrics, f, indent=2)
            
            self.logger.info(f"Enhanced metrics stored to DagsHub: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics to DagsHub: {e}")
    
    def create_model_deployment_config(self, model_name: str, version: str, 
                                     deployment_config: Dict[str, Any]) -> str:
        """
        Create deployment configuration for model
        
        Args:
            model_name: Registered model name
            version: Model version
            deployment_config: Deployment configuration
            
        Returns:
            Configuration file path
        """
        try:
            config_dir = Path("models/dagshub/deployment")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            deployment_info = {
                "model_name": model_name,
                "version": version,
                "deployment_config": deployment_config,
                "created_at": datetime.now().isoformat(),
                "dagshub_url": self.repo_url,
                "mlflow_uri": mlflow.get_tracking_uri()
            }
            
            config_file = config_dir / f"{model_name}_v{version}_deployment.json"
            with open(config_file, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            self.logger.info(f"Deployment config created: {config_file}")
            
            return str(config_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create deployment config: {e}")
            return None
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        try:
            if not mlflow.active_run():
                return {"error": "No active run"}
            
            run = mlflow.active_run()
            run_data = self.mlflow_client.get_run(run.info.run_id)
            
            summary = {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "experiment_name": self.experiment_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run_data.data.metrics,
                "params": run_data.data.params,
                "tags": run_data.data.tags,
                "artifacts": [artifact.path for artifact in self.mlflow_client.list_artifacts(run.info.run_id)],
                "dagshub_url": self.repo_url,
                "mlflow_uri": mlflow.get_tracking_uri()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment summary: {e}")
            return {"error": str(e)}
    
    def finish_run(self):
        """Finish current MLflow run"""
        try:
            if mlflow.active_run():
                # Get run summary before ending
                summary = self.get_experiment_summary()
                
                # Save run summary
                summary_dir = Path("experiments/dagshub")
                summary_dir.mkdir(parents=True, exist_ok=True)
                
                run_id = mlflow.active_run().info.run_id
                summary_file = summary_dir / f"run_summary_{run_id}.json"
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # End MLflow run
                mlflow.end_run()
                
                self.logger.info(f"MLflow run finished: {run_id}")
                self.logger.info(f"Run summary saved: {summary_file}")
                
                # Print summary
                print(f"\n📊 Run Summary: {run_id}")
                print(f"🔗 DagsHub URL: {self.repo_url}")
                print(f"📈 Metrics logged: {len(summary.get('metrics', {}))}")
                print(f"📁 Artifacts logged: {len(summary.get('artifacts', []))}")
                
                return summary
            else:
                self.logger.warning("No active run to finish")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to finish run: {e}")
            return None
    
    def finish_experiment(self):
        """Legacy method - redirects to finish_run"""
        return self.finish_run()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish_run()


# Legacy class alias for backward compatibility
DagsHubTracker = EnhancedDagsHubTracker


def main():
    """Example usage of enhanced DagsHub integration"""
    logger.info("Enhanced DagsHub integration module loaded successfully")
    logger.info("Features: MLflow tracking, model registry, automated versioning, deployment workflows")


if __name__ == "__main__":
    main()
    def get_dagshub_status(self) -> Dict[str, Any]:
        """Get DagsHub integration status for dashboard"""
        
        current_uri = mlflow.get_tracking_uri()
        is_local = current_uri.startswith("file:")
        
        status = {
            "configured": self.dagshub_configured,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "mlflow_uri": current_uri,
            "tracking_mode": "Local MLflow" if is_local else "DagsHub MLflow",
            "is_local": is_local,
            "models_stored": 0,
            "metrics_files": 0,
            "experiments": 0,
            "artifacts": 0
        }
        
        try:
            # Count stored artifacts
            models_dir = Path("models/dagshub")
            if models_dir.exists():
                status["models_stored"] = len(list(models_dir.glob("*.joblib")))
            
            metrics_dir = Path("metrics/dagshub")
            if metrics_dir.exists():
                status["metrics_files"] = len(list(metrics_dir.glob("*.json")))
            
            experiments_dir = Path("experiments/dagshub")
            if experiments_dir.exists():
                status["experiments"] = len(list(experiments_dir.glob("*.json")))
            
            artifacts_dir = Path("artifacts/dagshub")
            if artifacts_dir.exists():
                status["artifacts"] = len(list(artifacts_dir.glob("*")))
            
            # Get MLflow experiments
            try:
                experiments = self.mlflow_client.search_experiments()
                status["mlflow_experiments"] = len(experiments)
            except:
                status["mlflow_experiments"] = 0
            
        except Exception as e:
            self.logger.warning(f"Failed to get status details: {e}")
        
        return status

def main():
    """Example usage and testing"""
    
    # Test DagsHub integration
    tracker = EnhancedDagsHubTracker("avirup112", "Demo-Banking")
    
    # Get status
    status = tracker.get_dagshub_status()
    print("DagsHub Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()