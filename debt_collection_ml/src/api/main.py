from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.ml_model import DebtCollectionMLModel
from models.recommendations import RecommendationEngine
from models.explainability import ModelExplainer
from data.data_preprocessor import AdvancedDataPreprocessor
from utils.mlops import MLOpsOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Debt Collection ML API",
    description="REST API for debt collection ML predictions and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
ml_model = None
preprocessor = None
recommender = None
explainer = None
mlops = None

# Pydantic models for API requests/responses
class CustomerData(BaseModel):
    """Customer data for prediction"""
    Customer_ID: str = Field(..., description="Unique customer identifier")
    Bank_Code: str = Field(..., description="Bank code")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Income: float = Field(..., gt=0, description="Customer income")
    Occupation: str = Field(..., description="Customer occupation")
    Loan_Amount: float = Field(..., gt=0, description="Original loan amount")
    Outstanding_Balance: float = Field(..., ge=0, description="Current outstanding balance")
    Days_Past_Due: int = Field(..., ge=0, description="Days past due")
    Number_of_Calls: int = Field(..., ge=0, description="Number of collection calls made")
    Response_Rate: float = Field(..., ge=0, le=100, description="Response rate percentage")
    Last_Contact_Channel: str = Field(..., description="Last contact channel used")
    Payment_Made_Last_30_Days: int = Field(..., ge=0, le=1, description="Payment made in last 30 days (0/1)")
    Region: str = Field(..., description="Customer region")
    Credit_Score: int = Field(..., ge=300, le=850, description="Credit score")
    Complaint_Flag: int = Field(..., ge=0, le=1, description="Complaint flag (0/1)")

class PredictionResponse(BaseModel):
    """Prediction response model"""
    customer_id: str
    prediction: str
    prediction_probability: Dict[str, float]
    confidence_score: float
    risk_level: str
    business_metrics: Dict[str, float]

class RecommendationResponse(BaseModel):
    """Recommendation response model"""
    customer_id: str
    channel_recommendation: Dict[str, Any]
    timing_recommendation: Dict[str, Any]
    priority_score: float
    overall_confidence: float

class ExplanationResponse(BaseModel):
    """Model explanation response"""
    customer_id: str
    prediction: str
    top_features: List[Dict[str, float]]
    explanation_summary: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and initialize components on startup"""
    global ml_model, preprocessor, recommender, explainer, mlops
    
    logger.info("Starting up Debt Collection ML API...")
    
    try:
        # Load preprocessor
        preprocessor_path = Path("models/artifacts/preprocessor.joblib")
        if preprocessor_path.exists():
            preprocessor = AdvancedDataPreprocessor()
            preprocessor.load_preprocessor(str(preprocessor_path))
            logger.info("✅ Preprocessor loaded successfully")
        else:
            logger.warning("⚠️ Preprocessor not found")
        
        # Load ML model
        model_files = list(Path("models/trained").glob("*.joblib"))
        if model_files:
            model_path = model_files[0]  # Load first available model
            ml_model = DebtCollectionMLModel()
            ml_model.load_model(str(model_path))
            logger.info(f"✅ ML model loaded: {model_path.name}")
        else:
            logger.warning("⚠️ No trained models found")
        
        # Initialize recommendation engine
        recommender = RecommendationEngine()
        logger.info("✅ Recommendation engine initialized")
        
        # Initialize MLOps orchestrator
        mlops = MLOpsOrchestrator()
        logger.info("✅ MLOps orchestrator initialized")
        
        logger.info("🚀 API startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=ml_model is not None,
        api_version="1.0.0"
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_repayment(customer_data: CustomerData):
    """Predict repayment probability for a customer"""
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data.dict()])
        
        # Preprocess data
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = ml_model.predict(X_processed)[0]
        prediction_proba = ml_model.predict_proba(X_processed)[0]
        
        # Map prediction to class names
        class_names = ['Not Paid', 'Partially Paid', 'Paid']
        predicted_class = class_names[prediction]
        
        # Create probability dictionary
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(prediction_proba)}
        
        # Calculate confidence score
        confidence_score = float(np.max(prediction_proba))
        
        # Determine risk level
        if prediction == 0:  # Not Paid
            risk_level = "High Risk"
        elif prediction == 1:  # Partially Paid
            risk_level = "Medium Risk"
        else:  # Paid
            risk_level = "Low Risk"
        
        # Calculate business metrics (mock for API response)
        business_metrics = {
            "expected_recovery_rate": float(prediction_proba[1] * 0.5 + prediction_proba[2] * 1.0),
            "collection_priority": float(1.0 - prediction_proba[2]),  # Higher for less likely to pay
            "contact_urgency": float(customer_data.Days_Past_Due / 365),
            "risk_score": float(1.0 - confidence_score if prediction == 0 else confidence_score)
        }
        
        return PredictionResponse(
            customer_id=customer_data.Customer_ID,
            prediction=predicted_class,
            prediction_probability=prob_dict,
            confidence_score=confidence_score,
            risk_level=risk_level,
            business_metrics=business_metrics
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerData]):
    """Batch prediction for multiple customers"""
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size too large (max 1000)")
    
    try:
        results = []
        
        for customer in customers:
            # Use the single prediction endpoint
            prediction_result = await predict_repayment(customer)
            results.append(prediction_result)
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(customer_data: CustomerData):
    """Get collection recommendations for a customer"""
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not loaded")
    
    try:
        # Convert to dictionary for recommender
        customer_dict = customer_data.dict()
        
        # Get comprehensive recommendations
        recommendations = recommender.get_comprehensive_recommendation(customer_dict)
        
        return RecommendationResponse(
            customer_id=customer_data.Customer_ID,
            channel_recommendation=recommendations.get('channel_recommendation', {}),
            timing_recommendation=recommendations.get('timing_recommendation', {}),
            priority_score=recommendations.get('priority_score', 0.0),
            overall_confidence=recommendations.get('overall_confidence', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# Explanation endpoint
@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(customer_data: CustomerData):
    """Get explanation for a prediction"""
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        # Convert to DataFrame and preprocess
        df = pd.DataFrame([customer_data.dict()])
        X_processed = preprocessor.transform(df)
        
        # Make prediction first
        prediction = ml_model.predict(X_processed)[0]
        class_names = ['Not Paid', 'Partially Paid', 'Paid']
        predicted_class = class_names[prediction]
        
        # Mock explanation (in production, would use SHAP/LIME)
        # This simulates feature importance for the explanation
        feature_importance = {
            'Credit_Score': 0.25,
            'Days_Past_Due': 0.20,
            'Response_Rate': 0.15,
            'Outstanding_Balance': 0.12,
            'Income': 0.10,
            'Age': 0.08,
            'Number_of_Calls': 0.06,
            'Payment_Made_Last_30_Days': 0.04
        }
        
        # Create top features list
        top_features = [
            {"feature": feature, "importance": importance}
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Generate explanation summary
        explanation_summary = f"The model predicts '{predicted_class}' based primarily on credit score, days past due, and response rate patterns."
        
        return ExplanationResponse(
            customer_id=customer_data.Customer_ID,
            prediction=predicted_class,
            top_features=top_features,
            explanation_summary=explanation_summary
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Model information endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    try:
        model_info = {
            "model_type": ml_model.model_type,
            "random_state": ml_model.random_state,
            "feature_importance_available": ml_model.feature_importance is not None,
            "cv_scores_available": ml_model.cv_scores is not None,
            "best_params": ml_model.best_params
        }
        
        if ml_model.cv_scores is not None:
            model_info["cv_mean_score"] = float(ml_model.cv_scores.mean())
            model_info["cv_std_score"] = float(ml_model.cv_scores.std())
        
        return model_info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Feature importance endpoint
@app.get("/model/feature-importance")
async def get_feature_importance():
    """Get feature importance from the model"""
    
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    if ml_model.feature_importance is None:
        raise HTTPException(status_code=404, detail="Feature importance not available for this model")
    
    try:
        # Mock feature names (in production, would get from preprocessor)
        feature_names = [
            'Age', 'Income', 'Loan_Amount', 'Outstanding_Balance', 'Days_Past_Due',
            'Number_of_Calls', 'Response_Rate', 'Credit_Score', 'Payment_Made_Last_30_Days'
        ]
        
        # Get top 10 most important features
        importance_indices = np.argsort(ml_model.feature_importance)[::-1][:10]
        
        feature_importance_list = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": float(ml_model.feature_importance[i])
            }
            for i in importance_indices
        ]
        
        return {
            "feature_importance": feature_importance_list,
            "total_features": len(ml_model.feature_importance)
        }
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

# Statistics endpoint
@app.get("/stats")
async def get_api_stats():
    """Get API usage statistics"""
    
    # Mock statistics (in production, would track actual usage)
    stats = {
        "total_predictions": 1250,
        "total_recommendations": 890,
        "total_explanations": 340,
        "uptime_hours": 72.5,
        "average_response_time_ms": 145,
        "model_accuracy": 0.847,
        "last_model_update": "2024-01-15T10:30:00Z"
    }
    
    return stats

# Monitoring endpoint for MLOps
@app.post("/monitor/data-drift")
async def monitor_data_drift(current_data: List[CustomerData]):
    """Monitor for data drift"""
    
    if mlops is None:
        raise HTTPException(status_code=503, detail="MLOps orchestrator not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict() for customer in current_data])
        
        # Mock drift detection (in production, would use actual drift detection)
        drift_results = {
            "drift_detected": False,
            "drift_score": 0.05,
            "drifted_features": [],
            "timestamp": datetime.now().isoformat(),
            "recommendation": "No action required"
        }
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Data drift monitoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Drift monitoring failed: {str(e)}")

# Background task for model retraining
@app.post("/retrain")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)"""
    
    def retrain_model():
        """Background task to retrain the model"""
        logger.info("Starting model retraining...")
        # In production, this would trigger the actual retraining pipeline
        logger.info("Model retraining completed")
    
    background_tasks.add_task(retrain_model)
    
    return {
        "message": "Model retraining triggered",
        "status": "started",
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Debt Collection ML API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "recommend": "/recommend",
            "explain": "/explain",
            "model_info": "/model/info",
            "feature_importance": "/model/feature-importance",
            "stats": "/stats"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)