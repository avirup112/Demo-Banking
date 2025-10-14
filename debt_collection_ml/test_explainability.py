#!/usr/bin/env python3
"""
Simple SHAP Explainability Test
"""

import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('src')

def test_explainability():
    """Test SHAP explainability on trained model"""
    
    print("🔍 Testing SHAP Explainability...")
    
    try:
        # Load model
        model_files = list(Path("models/optimized").glob("*.joblib"))
        if not model_files:
            model_files = list(Path("models/trained").glob("*.joblib"))
        
        if not model_files:
            print("❌ No trained models found")
            return
        
        model = joblib.load(model_files[0])
        model_name = model_files[0].stem
        print(f"📊 Loaded model: {model_name}")
        
        # Load test data
        try:
            X_test = np.load('data/processed/X_engineered.npy')
        except FileNotFoundError:
            X_test = np.load('data/processed/X_processed.npy')
        
        print(f"📈 Data shape: {X_test.shape}")
        
        # Use small sample for speed
        sample_size = min(50, len(X_test))
        X_sample = X_test[:sample_size]
        
        # Create SHAP explainer
        print("🔧 Creating SHAP explainer...")
        
        if 'forest' in model_name.lower() or 'tree' in model_name.lower():
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_sample[:20])
        
        # Get SHAP values
        print("📊 Computing SHAP values...")
        shap_values = explainer.shap_values(X_sample[:10])
        
        # Create feature names
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Calculate feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            if len(shap_values.shape) == 3:
                # Multi-class in single array
                importance = np.abs(shap_values).mean(axis=(0, 2))
            else:
                importance = np.abs(shap_values).mean(axis=0)
        
        # Ensure importance is 1D array
        if len(importance.shape) > 1:
            importance = importance.flatten()
        
        # Get top features
        top_indices = np.argsort(importance)[-10:][::-1]
        
        print(f"\n🎯 TOP 10 FEATURES DRIVING REPAYMENT DECISIONS:")
        print("-" * 50)
        
        for i, idx in enumerate(top_indices, 1):
            print(f"{i:2d}. {feature_names[idx]:<20} (Importance: {importance[idx]:.4f})")
        
        # Create summary plot
        print(f"\n📊 Creating SHAP summary plot...")
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            # For multi-class, show first class
            shap.summary_plot(shap_values[0], X_sample[:10], 
                            feature_names=feature_names, show=False, max_display=10)
        else:
            shap.summary_plot(shap_values, X_sample[:10], 
                            feature_names=feature_names, show=False, max_display=10)
        
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        
        # Save plot
        Path("explanations").mkdir(exist_ok=True)
        plt.savefig('explanations/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Summary plot saved to: explanations/shap_summary.png")
        
        # Individual explanation example
        if len(X_sample) > 0:
            print(f"\n🔍 Individual Prediction Explanation (Sample 1):")
            
            instance = X_sample[0]
            prediction = model.predict([instance])[0]
            prediction_proba = model.predict_proba([instance])[0]
            
            class_names = ['Not Paid', 'Paid', 'Partially Paid']
            
            print(f"   Predicted Class: {class_names[prediction]}")
            print(f"   Probabilities: {dict(zip(class_names, prediction_proba))}")
            
            # Get SHAP values for this instance
            if isinstance(shap_values, list):
                instance_shap = shap_values[prediction][0]
            else:
                if len(shap_values.shape) == 3:
                    instance_shap = shap_values[0, :, prediction]
                else:
                    instance_shap = shap_values[0]
            
            # Top contributing features
            feature_contributions = list(zip(feature_names, instance_shap.flatten(), instance.flatten()))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"   Top Contributing Features:")
            for i, (feat, shap_val, feat_val) in enumerate(feature_contributions[:5], 1):
                impact = "↑" if shap_val > 0 else "↓"
                print(f"     {i}. {feat:<20} {impact} {shap_val:+.4f} (value: {feat_val:.3f})")
        
        print(f"\n✅ SHAP explainability analysis completed!")
        print(f"📁 Check explanations/ directory for visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Explainability test failed: {e}")
        return False

if __name__ == "__main__":
    test_explainability()