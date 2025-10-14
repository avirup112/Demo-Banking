import os
import dagshub
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class DagsHubTracker:
    """DagsHub integration for experiment tracking and model registry"""
    
    def __init__(self, repo_owner: str, repo_name: str, mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize DagsHub tracker
        
        Args:
            repo_owner: Your DagsHub username
            repo_name: Repository name on DagsHub
            mlflow_tracking_uri: Optional custom MLflow URI
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        # Initialize DagsHub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        
        # Set MLflow tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            # Use DagsHub's MLflow tracking URI
            mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        # Set experiment
        experiment_name = f"{repo_name}_debt_collection"
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists
        
        mlflow.set_experiment(experiment_name)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"DagsHub tracker initialized for {repo_owner}/{repo_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        
        run_tags = {
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "framework": "debt_collection_ml",
            "timestamp": datetime.now().isoformat()
        }
        
        if tags:
            run_tags.update(tags)
        
        return mlflow.start_run(run_name=run_name, tags=run_tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """Log model to MLflow"""
        
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)
    
    def log_artifacts(self, artifact_path: str, local_path: str = None):
        """Log artifacts to MLflow"""
        if local_path:
            mlflow.log_artifacts(local_path, artifact_path)
        else:
            mlflow.log_artifact(artifact_path)
    
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

class DagsHubModelRegistry:
    """Enhanced model registry with DagsHub integration"""
    
    def __init__(self, dagshub_tracker: DagsHubTracker):
        self.tracker = dagshub_tracker
        self.client = mlflow.tracking.MlflowClient()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def list_models(self) -> pd.DataFrame:
        """List all registered models"""
        
        try:
            registered_models = self.client.search_registered_models()
            
            model_data = []
            for model in registered_models:
                latest_version = self.client.get_latest_versions(
                    model.name, stages=["None", "Staging", "Production"]
                )[0]
                
                model_data.append({
                    "name": model.name,
                    "version": latest_version.version,
                    "stage": latest_version.current_stage,
                    "created_at": latest_version.creation_timestamp,
                    "description": model.description or "No description"
                })
            
            return pd.DataFrame(model_data)
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return pd.DataFrame()
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to a specific stage"""
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            self.logger.info(f"Model {model_name} v{version} promoted to {stage}")
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """Get detailed model information"""
        
        try:
            if version is None:
                # Get latest version
                latest_versions = self.client.get_latest_versions(model_name)
                if not latest_versions:
                    return {}
                model_version = latest_versions[0]
            else:
                model_version = self.client.get_model_version(model_name, version)
            
            # Get run information
            run = self.client.get_run(model_version.run_id)
            
            return {
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "created_at": model_version.creation_timestamp,
                "description": model_version.description
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {}

def setup_dagshub_integration(repo_owner: str, repo_name: str) -> DagsHubTracker:
    """Setup DagsHub integration for the project"""
    
    # Create DagsHub tracker
    tracker = DagsHubTracker(repo_owner, repo_name)
    
    # Create configuration file
    tracker.create_dagshub_yaml()
    
    # Create .dvc directory structure
    dvc_dirs = [".dvc", "data/.dvc", "models/.dvc"]
    for dvc_dir in dvc_dirs:
        Path(dvc_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"""
DagsHub Integration Setup Complete!
=====================================

Repository: https://dagshub.com/{repo_owner}/{repo_name}
MLflow UI: https://dagshub.com/{repo_owner}/{repo_name}.mlflow

Next Steps:
1. Create a repository on DagsHub: https://dagshub.com/repo/create
2. Push your code to the repository
3. Run experiments and view them in the MLflow UI
4. Use DVC for data versioning (optional)

Example Usage:
--------------
tracker = DagsHubTracker("{repo_owner}", "{repo_name}")
with tracker.start_run("experiment_1"):
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
    with tracker.start_run("test_experiment"):
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