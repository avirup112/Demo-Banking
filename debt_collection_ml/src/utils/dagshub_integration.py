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

class DagsHubTracker:
    """Pure DagsHub integration for experiment tracking and data versioning"""
    
    def __init__(self, repo_owner: str, repo_name: str):
        """
        Initialize DagsHub tracker
        
        Args:
            repo_owner: Your DagsHub username
            repo_name: Repository name on DagsHub
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        
        # Initialize DagsHub without MLflow
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=False)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create experiments directory
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Current experiment tracking
        self.current_experiment = None
        self.experiment_data = {}
        
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
                # Handle step-based metrics
                for key, value in metrics.items():
                    step_key = f"{key}_step_{step}"
                    self.experiment_data["metrics"][step_key] = value
            else:
                self.experiment_data["metrics"].update(metrics)
            self.logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """Save model as artifact"""
        import joblib
        
        # Create models directory
        models_dir = Path("models/experiments")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{model_name}_{self.current_experiment}.joblib"
        joblib.dump(model, model_path)
        
        # Log as artifact
        self.experiment_data["artifacts"].append({
            "name": model_name,
            "path": str(model_path),
            "type": "model",
            "model_type": model_type
        })
        
        self.logger.info(f"Saved model: {model_name} to {model_path}")
    
    def log_artifacts(self, artifact_path: str, artifact_name: str = None):
        """Log artifacts to experiment"""
        if self.current_experiment:
            self.experiment_data["artifacts"].append({
                "name": artifact_name or Path(artifact_path).name,
                "path": artifact_path,
                "type": "artifact"
            })
            self.logger.info(f"Logged artifact: {artifact_path}")
    
    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str = "training_data"):
        """Log dataset information"""
        
        dataset_info = {
            f"{dataset_name}_shape_rows": df.shape[0],
            f"{dataset_name}_shape_cols": df.shape[1],
            f"{dataset_name}_memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            f"{dataset_name}_missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        # Log basic dataset metrics
        self.log_metrics(dataset_info)
        
        # Log column information
        column_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_counts": df.isnull().sum().to_dict()
        }
        
        # Save and log column info as artifact
        column_info_path = f"temp_{dataset_name}_info.json"
        with open(column_info_path, 'w') as f:
            json.dump(column_info, f, indent=2)
        
        mlflow.log_artifact(column_info_path, f"{dataset_name}_info")
        
        # Clean up temp file
        os.remove(column_info_path)
    
    def log_model_comparison(self, comparison_df: pd.DataFrame):
        """Log model comparison results"""
        
        # Save comparison as CSV
        comparison_path = "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Log as artifact
        mlflow.log_artifact(comparison_path, "model_comparison")
        
        # Log best model metrics
        best_model_idx = comparison_df['Business F1'].idxmax()
        best_model_metrics = {
            "best_model_name": comparison_df.loc[best_model_idx, 'Model'],
            "best_accuracy": comparison_df.loc[best_model_idx, 'Accuracy'],
            "best_f1_score": comparison_df.loc[best_model_idx, 'F1-Score'],
            "best_roc_auc": comparison_df.loc[best_model_idx, 'ROC-AUC'],
            "best_business_f1": comparison_df.loc[best_model_idx, 'Business F1'],
            "best_recovery_precision": comparison_df.loc[best_model_idx, 'Recovery Precision']
        }
        
        self.log_metrics(best_model_metrics)
        
        # Clean up temp file
        os.remove(comparison_path)
    
    def log_feature_importance(self, feature_names: list, importance_values: np.ndarray, 
                              importance_type: str = "feature_importance"):
        """Log feature importance"""
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        # Save as CSV
        importance_path = f"{importance_type}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Log as artifact
        mlflow.log_artifact(importance_path, "feature_importance")
        
        # Log top 10 features as metrics
        top_features = importance_df.head(10)
        for idx, row in top_features.iterrows():
            mlflow.log_metric(f"top_feature_{idx+1}_{row['feature']}", row['importance'])
        
        # Clean up temp file
        os.remove(importance_path)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: list = None):
        """Log confusion matrix"""
        
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log as artifact
        mlflow.log_artifact(cm_path, "evaluation_plots")
        
        # Clean up temp file
        os.remove(cm_path)
    
    def log_training_history(self, history: Dict[str, list]):
        """Log training history (for iterative models)"""
        
        for metric_name, values in history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(f"training_{metric_name}", value, step=step)
    
    def register_model(self, model_name: str, model_version: str = None):
        """Register model in MLflow Model Registry"""
        
        try:
            # Register the model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "repo": f"{self.repo_owner}/{self.repo_name}",
                    "framework": "debt_collection_ml"
                }
            )
            
            self.logger.info(f"Model registered: {model_name} v{registered_model.version}")
            return registered_model
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return None
    
    def create_dagshub_yaml(self, output_path: str = ".dagshub.yaml"):
        """Create DagsHub configuration file"""
        
        config = {
            "repo": f"{self.repo_owner}/{self.repo_name}",
            "mlflow": {
                "tracking_uri": f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow"
            },
            "dvc": {
                "remote": "origin"
            },
            "experiments": {
                "default_branch": "main",
                "auto_stage": True
            }
        }
        
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"DagsHub configuration saved to {output_path}")
    
    def finish_experiment(self):
        """Finish and save current experiment"""
        if not self.current_experiment:
            self.logger.warning("No active experiment to finish")
            return
        
        # Add end time
        self.experiment_data["end_time"] = datetime.now().isoformat()
        
        # Save experiment data
        experiment_file = self.experiments_dir / f"{self.current_experiment}.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        self.logger.info(f"Experiment saved: {experiment_file}")
        
        # Reset current experiment
        self.current_experiment = None
        self.experiment_data = {}
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments"""
        experiments = []
        
        for exp_file in self.experiments_dir.glob("*.json"):
            try:
                with open(exp_file, 'r') as f:
                    exp_data = json.load(f)
                
                experiments.append({
                    "name": exp_data["experiment_name"],
                    "start_time": exp_data["start_time"],
                    "end_time": exp_data.get("end_time", "Running"),
                    "params_count": len(exp_data.get("params", {})),
                    "metrics_count": len(exp_data.get("metrics", {})),
                    "artifacts_count": len(exp_data.get("artifacts", []))
                })
            except Exception as e:
                self.logger.error(f"Error reading experiment {exp_file}: {e}")
        
        return pd.DataFrame(experiments)
    
    def get_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Get experiment data"""
        exp_file = self.experiments_dir / f"{experiment_name}.json"
        
        if not exp_file.exists():
            self.logger.error(f"Experiment not found: {experiment_name}")
            return {}
        
        try:
            with open(exp_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading experiment {experiment_name}: {e}")
            return {}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish_experiment()

class DagsHubModelRegistry:
    """Simple model registry with DagsHub integration"""
    
    def __init__(self, dagshub_tracker: DagsHubTracker):
        self.tracker = dagshub_tracker
        self.models_dir = Path("models/registry")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model, model_name: str, version: str, 
                      metrics: Dict[str, float], params: Dict[str, Any],
                      description: str = ""):
        """Register a model in the registry"""
        
        import joblib
        
        # Create model version directory
        model_version_dir = self.models_dir / model_name / version
        model_version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_version_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "name": model_name,
            "version": version,
            "description": description,
            "metrics": metrics,
            "params": params,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "repo_owner": self.tracker.repo_owner,
            "repo_name": self.tracker.repo_name
        }
        
        metadata_path = model_version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model registered: {model_name} v{version}")
        return metadata
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models"""
        
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        metadata_file = version_dir / "metadata.json"
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                models.append(metadata)
                            except Exception as e:
                                self.logger.error(f"Error reading {metadata_file}: {e}")
        
        return pd.DataFrame(models)
    
    def get_model(self, model_name: str, version: str = "latest"):
        """Get a model from the registry"""
        
        import joblib
        
        if version == "latest":
            # Find latest version
            model_dir = self.models_dir / model_name
            if not model_dir.exists():
                self.logger.error(f"Model not found: {model_name}")
                return None, None
            
            versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
            if not versions:
                self.logger.error(f"No versions found for model: {model_name}")
                return None, None
            
            # Sort versions and get latest
            versions.sort(reverse=True)
            version = versions[0]
        
        # Load model and metadata
        model_version_dir = self.models_dir / model_name / version
        model_path = model_version_dir / "model.joblib"
        metadata_path = model_version_dir / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            self.logger.error(f"Model files not found: {model_name} v{version}")
            return None, None
        
        try:
            model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, None

def setup_dagshub_integration(repo_owner: str, repo_name: str) -> DagsHubTracker:
    """Setup DagsHub integration for the project"""
    
    # Create DagsHub tracker
    tracker = DagsHubTracker(repo_owner, repo_name)
    
    # Create configuration file
    tracker.create_dagshub_yaml()
    
    # Create directory structure
    dirs = ["experiments", "models/registry", "models/experiments", "data/raw", "data/processed"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"""
DagsHub Integration Setup Complete!
=====================================

Repository: https://dagshub.com/{repo_owner}/{repo_name}
Data: https://dagshub.com/{repo_owner}/{repo_name}/data
Experiments: https://dagshub.com/{repo_owner}/{repo_name}/experiments

Next Steps:
1. Create a repository on DagsHub: https://dagshub.com/repo/create
2. Push your code to the repository
3. Run experiments and track them with DagsHub
4. Use DVC for data versioning

Example Usage:
--------------
tracker = DagsHubTracker("{repo_owner}", "{repo_name}")
with tracker.start_experiment("experiment_1"):
    tracker.log_params({{"learning_rate": 0.1}})
    tracker.log_metrics({{"accuracy": 0.95}})
    tracker.log_model(model, "debt_collection_model")
""")
    
    return tracker

# Example usage and testing
if __name__ == "__main__":
    # Example setup - replace with your DagsHub credentials
    REPO_OWNER = "your_username"  # Replace with your DagsHub username
    REPO_NAME = "debt-collection-ml"  # Replace with your repository name
    
    # Setup DagsHub integration
    tracker = setup_dagshub_integration(REPO_OWNER, REPO_NAME)
    
    # Example experiment
    with tracker.start_experiment("test_experiment"):
        # Log parameters
        tracker.log_params({
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 6
        })
        
        # Log metrics
        tracker.log_metrics({
            "accuracy": 0.85,
            "f1_score": 0.83,
            "roc_auc": 0.89
        })
        
        print("Test experiment logged successfully!")