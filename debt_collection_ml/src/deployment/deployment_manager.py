#!/usr/bin/env python3
"""
Automated Model Deployment Pipeline with DagsHub Integration
Blue-green deployment, health checks, and automated rollback capabilities
"""

import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import requests
import yaml
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# ML Libraries
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    deployment_type: str  # 'staging', 'production', 'canary'
    environment: str  # 'staging', 'production'
    health_check_endpoint: str
    rollback_threshold: float = 0.95  # Success rate threshold
    max_rollback_attempts: int = 3
    deployment_timeout: int = 300  # seconds
    health_check_interval: int = 30  # seconds
    blue_green_enabled: bool = True
    auto_rollback_enabled: bool = True

@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    deployment_id: str
    model_name: str
    model_version: str
    environment: str
    status: str  # 'pending', 'deploying', 'healthy', 'unhealthy', 'failed', 'rolled_back'
    start_time: datetime
    end_time: Optional[datetime] = None
    health_score: float = 0.0
    error_message: Optional[str] = None
    rollback_count: int = 0

class ModelDeploymentManager:
    """Comprehensive model deployment manager with DagsHub integration"""
    
    def __init__(self, dagshub_tracker=None, base_deployment_dir: str = "deployments"):
        """
        Initialize deployment manager
        
        Args:
            dagshub_tracker: Enhanced DagsHub tracker instance
            base_deployment_dir: Base directory for deployments
        """
        self.dagshub_tracker = dagshub_tracker
        self.base_deployment_dir = Path(base_deployment_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize MLflow client
        self.mlflow_client = MlflowClient()
        
        # Create deployment directory structure
        self._create_deployment_structure()
        
        # Track active deployments
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[DeploymentStatus] = []
        
        # Load existing deployment state
        self._load_deployment_state()
        
        self.logger.info("Model deployment manager initialized")
    
    def _create_deployment_structure(self):
        """Create comprehensive deployment directory structure"""
        directories = [
            "deployments/staging/blue",
            "deployments/staging/green", 
            "deployments/staging/active",
            "deployments/production/blue",
            "deployments/production/green",
            "deployments/production/active",
            "deployments/configs",
            "deployments/logs",
            "deployments/health_checks",
            "deployments/rollbacks",
            "deployments/scripts"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Deployment directory structure created")
    
    def _load_deployment_state(self):
        """Load existing deployment state from disk"""
        state_file = self.base_deployment_dir / "deployment_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Reconstruct deployment status objects
                for deployment_data in state_data.get('active_deployments', []):
                    deployment_status = DeploymentStatus(**deployment_data)
                    self.active_deployments[deployment_status.deployment_id] = deployment_status
                
                for deployment_data in state_data.get('deployment_history', []):
                    deployment_status = DeploymentStatus(**deployment_data)
                    self.deployment_history.append(deployment_status)
                
                self.logger.info(f"Loaded deployment state: {len(self.active_deployments)} active deployments")
                
            except Exception as e:
                self.logger.error(f"Failed to load deployment state: {e}")
    
    def _save_deployment_state(self):
        """Save deployment state to disk"""
        state_file = self.base_deployment_dir / "deployment_state.json"
        
        try:
            # Convert to serializable format
            state_data = {
                'active_deployments': [asdict(status) for status in self.active_deployments.values()],
                'deployment_history': [asdict(status) for status in self.deployment_history[-100:]],  # Keep last 100
                'last_updated': datetime.now().isoformat()
            }
            
            # Handle datetime serialization
            def datetime_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=datetime_serializer)
            
        except Exception as e:
            self.logger.error(f"Failed to save deployment state: {e}")
    
    def create_deployment_config(self, model_name: str, model_version: str, 
                               environment: str = "staging", **kwargs) -> DeploymentConfig:
        """
        Create deployment configuration
        
        Args:
            model_name: Name of the model to deploy
            model_version: Version of the model
            environment: Target environment
            **kwargs: Additional configuration options
            
        Returns:
            DeploymentConfig object
        """
        config = DeploymentConfig(
            model_name=model_name,
            model_version=model_version,
            deployment_type=environment,
            environment=environment,
            health_check_endpoint=f"/health/{model_name}",
            **kwargs
        )
        
        # Save configuration
        config_file = self.base_deployment_dir / "configs" / f"{model_name}_v{model_version}_{environment}.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        self.logger.info(f"Deployment config created: {config_file}")
        
        return config
    
    def get_model_from_registry(self, model_name: str, version: str) -> Tuple[Any, str]:
        """
        Retrieve model from MLflow registry
        
        Args:
            model_name: Registered model name
            version: Model version
            
        Returns:
            Tuple of (model_object, model_uri)
        """
        try:
            # Get model version details
            model_version = self.mlflow_client.get_model_version(model_name, version)
            model_uri = model_version.source
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            
            self.logger.info(f"Retrieved model {model_name} v{version} from registry")
            
            return model, model_uri
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve model from registry: {e}")
            return None, None
    
    def prepare_deployment_environment(self, config: DeploymentConfig, 
                                     slot: str = "blue") -> str:
        """
        Prepare deployment environment (blue or green slot)
        
        Args:
            config: Deployment configuration
            slot: Deployment slot ('blue' or 'green')
            
        Returns:
            Deployment directory path
        """
        deployment_dir = self.base_deployment_dir / config.environment / slot
        
        # Clean existing deployment
        if deployment_dir.exists():
            shutil.rmtree(deployment_dir)
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model from registry
        model, model_uri = self.get_model_from_registry(
            config.model_name, config.model_version
        )
        
        if model is None:
            raise ValueError(f"Failed to retrieve model {config.model_name} v{config.model_version}")
        
        # Save model to deployment directory
        model_file = deployment_dir / "model.joblib"
        joblib.dump(model, model_file)
        
        # Create model metadata
        metadata = {
            "model_name": config.model_name,
            "model_version": config.model_version,
            "model_uri": model_uri,
            "deployment_time": datetime.now().isoformat(),
            "environment": config.environment,
            "slot": slot
        }
        
        metadata_file = deployment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create deployment script
        self._create_deployment_script(deployment_dir, config)
        
        # Create health check script
        self._create_health_check_script(deployment_dir, config)
        
        self.logger.info(f"Deployment environment prepared: {deployment_dir}")
        
        return str(deployment_dir)
    
    def _create_deployment_script(self, deployment_dir: Path, config: DeploymentConfig):
        """Create deployment script for the model"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated deployment script for {config.model_name} v{config.model_version}
"""

import joblib
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata
model_dir = Path(__file__).parent
model = joblib.load(model_dir / "model.joblib")

with open(model_dir / "metadata.json", 'r') as f:
    metadata = json.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert to numpy array
        if isinstance(data, dict) and 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        else:
            features = np.array(data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            confidence = float(np.max(probabilities))
        else:
            confidence = 1.0
        
        return jsonify({{
            'prediction': int(prediction[0]),
            'confidence': confidence,
            'model_name': metadata['model_name'],
            'model_version': metadata['model_version'],
            'timestamp': metadata['deployment_time']
        }})
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return jsonify({{'error': str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{
        'status': 'healthy',
        'model_name': metadata['model_name'],
        'model_version': metadata['model_version'],
        'environment': metadata['environment'],
        'deployment_time': metadata['deployment_time']
    }})

@app.route('/metadata', methods=['GET'])
def get_metadata():
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        script_file = deployment_dir / "app.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_file.chmod(0o755)
        
        self.logger.info(f"Deployment script created: {script_file}")
    
    def _create_health_check_script(self, deployment_dir: Path, config: DeploymentConfig):
        """Create health check script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Health check script for {config.model_name} v{config.model_version}
"""

import requests
import json
import sys
import time
from datetime import datetime

def check_health(endpoint_url, timeout=10):
    """Check if the deployment is healthy"""
    try:
        response = requests.get(f"{{endpoint_url}}/health", timeout=timeout)
        
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        else:
            return False, {{"error": f"HTTP {{response.status_code}}"}}
            
    except Exception as e:
        return False, {{"error": str(e)}}

def test_prediction(endpoint_url, test_data=None, timeout=10):
    """Test prediction endpoint"""
    try:
        if test_data is None:
            # Default test data (adjust based on your model)
            test_data = {{"features": [0.5] * 10}}  # Adjust feature count
        
        response = requests.post(
            f"{{endpoint_url}}/predict", 
            json=test_data,
            timeout=timeout
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            return True, prediction_data
        else:
            return False, {{"error": f"HTTP {{response.status_code}}"}}
            
    except Exception as e:
        return False, {{"error": str(e)}}

def main():
    endpoint_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    print(f"Checking health of {{endpoint_url}}")
    
    # Health check
    health_ok, health_data = check_health(endpoint_url)
    print(f"Health check: {{'OK' if health_ok else 'FAILED'}}")
    print(f"Health data: {{json.dumps(health_data, indent=2)}}")
    
    if not health_ok:
        sys.exit(1)
    
    # Prediction test
    pred_ok, pred_data = test_prediction(endpoint_url)
    print(f"Prediction test: {{'OK' if pred_ok else 'FAILED'}}")
    print(f"Prediction data: {{json.dumps(pred_data, indent=2)}}")
    
    if not pred_ok:
        sys.exit(1)
    
    print("All health checks passed!")
    sys.exit(0)

if __name__ == "__main__":
    main()
'''
        
        script_file = deployment_dir / "health_check.py"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_file.chmod(0o755)
        
        self.logger.info(f"Health check script created: {script_file}")
    
    def deploy_model(self, config: DeploymentConfig) -> str:
        """
        Deploy model using blue-green deployment strategy
        
        Args:
            config: Deployment configuration
            
        Returns:
            Deployment ID
        """
        deployment_id = f"{config.model_name}_v{config.model_version}_{config.environment}_{int(time.time())}"
        
        # Create deployment status
        deployment_status = DeploymentStatus(
            deployment_id=deployment_id,
            model_name=config.model_name,
            model_version=config.model_version,
            environment=config.environment,
            status="pending",
            start_time=datetime.now()
        )
        
        self.active_deployments[deployment_id] = deployment_status
        
        try:
            self.logger.info(f"Starting deployment: {deployment_id}")
            
            # Update status
            deployment_status.status = "deploying"
            
            # Determine deployment slot (blue/green)
            active_slot = self._get_active_slot(config.environment)
            target_slot = "green" if active_slot == "blue" else "blue"
            
            self.logger.info(f"Deploying to {target_slot} slot (active: {active_slot})")
            
            # Prepare deployment environment
            deployment_dir = self.prepare_deployment_environment(config, target_slot)
            
            # Log deployment to DagsHub
            if self.dagshub_tracker:
                self.dagshub_tracker.log_params({
                    f"deployment_{deployment_id}_model": config.model_name,
                    f"deployment_{deployment_id}_version": config.model_version,
                    f"deployment_{deployment_id}_environment": config.environment,
                    f"deployment_{deployment_id}_slot": target_slot
                })
            
            # Simulate deployment process (in real scenario, this would deploy to actual infrastructure)
            self._simulate_deployment(deployment_dir, config)
            
            # Perform health checks
            health_passed = self._perform_health_checks(deployment_dir, config)
            
            if health_passed:
                # Switch traffic to new deployment
                self._switch_traffic(config.environment, target_slot)
                
                deployment_status.status = "healthy"
                deployment_status.health_score = 1.0
                deployment_status.end_time = datetime.now()
                
                self.logger.info(f"Deployment successful: {deployment_id}")
                
                # Log success to DagsHub
                if self.dagshub_tracker:
                    self.dagshub_tracker.log_metrics({
                        f"deployment_{deployment_id}_success": 1.0,
                        f"deployment_{deployment_id}_health_score": 1.0
                    })
                
            else:
                deployment_status.status = "unhealthy"
                deployment_status.health_score = 0.0
                deployment_status.error_message = "Health checks failed"
                
                self.logger.error(f"Deployment failed health checks: {deployment_id}")
                
                # Trigger rollback if enabled
                if config.auto_rollback_enabled:
                    self._rollback_deployment(deployment_id, "Health checks failed")
            
        except Exception as e:
            deployment_status.status = "failed"
            deployment_status.error_message = str(e)
            deployment_status.end_time = datetime.now()
            
            self.logger.error(f"Deployment failed: {deployment_id} - {e}")
            
            # Log failure to DagsHub
            if self.dagshub_tracker:
                self.dagshub_tracker.log_metrics({
                    f"deployment_{deployment_id}_success": 0.0,
                    f"deployment_{deployment_id}_error": 1.0
                })
        
        # Save deployment state
        self._save_deployment_state()
        
        return deployment_id
    
    def _get_active_slot(self, environment: str) -> str:
        """Get currently active deployment slot"""
        active_link = self.base_deployment_dir / environment / "active"
        
        if active_link.exists() and active_link.is_symlink():
            target = active_link.readlink()
            return target.name
        
        return "blue"  # Default to blue
    
    def _simulate_deployment(self, deployment_dir: str, config: DeploymentConfig):
        """Simulate deployment process"""
        self.logger.info(f"Simulating deployment to {deployment_dir}")
        
        # In a real scenario, this would:
        # 1. Build Docker container
        # 2. Push to container registry
        # 3. Deploy to Kubernetes/cloud platform
        # 4. Configure load balancer
        
        # For simulation, just wait a bit
        time.sleep(2)
        
        self.logger.info("Deployment simulation completed")
    
    def _perform_health_checks(self, deployment_dir: str, config: DeploymentConfig) -> bool:
        """Perform comprehensive health checks"""
        self.logger.info("Performing health checks...")
        
        try:
            # In a real scenario, this would check:
            # 1. Service is running
            # 2. Health endpoint responds
            # 3. Prediction endpoint works
            # 4. Performance metrics are acceptable
            
            # For simulation, perform basic checks
            deployment_path = Path(deployment_dir)
            
            # Check if required files exist
            required_files = ["model.joblib", "metadata.json", "app.py", "health_check.py"]
            for file_name in required_files:
                if not (deployment_path / file_name).exists():
                    self.logger.error(f"Required file missing: {file_name}")
                    return False
            
            # Simulate health check delay
            time.sleep(1)
            
            self.logger.info("Health checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Health checks failed: {e}")
            return False
    
    def _switch_traffic(self, environment: str, target_slot: str):
        """Switch traffic to new deployment slot"""
        active_link = self.base_deployment_dir / environment / "active"
        target_path = self.base_deployment_dir / environment / target_slot
        
        # Remove existing symlink
        if active_link.exists():
            active_link.unlink()
        
        # Create new symlink
        active_link.symlink_to(target_path, target_is_directory=True)
        
        self.logger.info(f"Traffic switched to {target_slot} slot in {environment}")
    
    def _rollback_deployment(self, deployment_id: str, reason: str):
        """Rollback failed deployment"""
        if deployment_id not in self.active_deployments:
            self.logger.error(f"Deployment not found for rollback: {deployment_id}")
            return
        
        deployment_status = self.active_deployments[deployment_id]
        
        self.logger.info(f"Rolling back deployment: {deployment_id} - Reason: {reason}")
        
        try:
            # Switch back to previous slot
            environment = deployment_status.environment
            current_slot = self._get_active_slot(environment)
            previous_slot = "blue" if current_slot == "green" else "green"
            
            # Check if previous slot exists and is healthy
            previous_path = self.base_deployment_dir / environment / previous_slot
            if previous_path.exists():
                self._switch_traffic(environment, previous_slot)
                
                deployment_status.status = "rolled_back"
                deployment_status.rollback_count += 1
                deployment_status.error_message = f"Rolled back: {reason}"
                deployment_status.end_time = datetime.now()
                
                self.logger.info(f"Rollback successful: {deployment_id}")
                
                # Log rollback to DagsHub
                if self.dagshub_tracker:
                    self.dagshub_tracker.log_metrics({
                        f"deployment_{deployment_id}_rollback": 1.0,
                        f"deployment_{deployment_id}_rollback_count": deployment_status.rollback_count
                    })
            else:
                self.logger.error(f"No previous deployment found for rollback: {deployment_id}")
                deployment_status.status = "failed"
                deployment_status.error_message = f"Rollback failed: No previous deployment"
        
        except Exception as e:
            self.logger.error(f"Rollback failed: {deployment_id} - {e}")
            deployment_status.status = "failed"
            deployment_status.error_message = f"Rollback failed: {e}"
        
        # Save state
        self._save_deployment_state()
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of a specific deployment"""
        return self.active_deployments.get(deployment_id)
    
    def list_deployments(self, environment: str = None) -> List[DeploymentStatus]:
        """List all deployments, optionally filtered by environment"""
        deployments = list(self.active_deployments.values())
        
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        
        return sorted(deployments, key=lambda x: x.start_time, reverse=True)
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get comprehensive deployment summary"""
        summary = {
            "total_deployments": len(self.active_deployments),
            "environments": {},
            "status_counts": {},
            "recent_deployments": []
        }
        
        # Count by environment and status
        for deployment in self.active_deployments.values():
            env = deployment.environment
            status = deployment.status
            
            if env not in summary["environments"]:
                summary["environments"][env] = {"total": 0, "healthy": 0, "unhealthy": 0}
            
            summary["environments"][env]["total"] += 1
            
            if status == "healthy":
                summary["environments"][env]["healthy"] += 1
            elif status in ["unhealthy", "failed"]:
                summary["environments"][env]["unhealthy"] += 1
            
            if status not in summary["status_counts"]:
                summary["status_counts"][status] = 0
            summary["status_counts"][status] += 1
        
        # Recent deployments
        recent = sorted(self.active_deployments.values(), key=lambda x: x.start_time, reverse=True)[:5]
        summary["recent_deployments"] = [asdict(d) for d in recent]
        
        return summary


def main():
    """Example usage of deployment manager"""
    logger.info("Model deployment manager module loaded successfully")
    logger.info("Features: blue-green deployment, health checks, automated rollback")


if __name__ == "__main__":
    main()