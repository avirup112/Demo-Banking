import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
import logging

class RecommendationEngine:
    """Recommendation engine for optimal contact strategy"""
    
    def __init__(self):
        self.channel_model = None
        self.timing_model = None
        self.customer_segments = None
        self.scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def prepare_features_for_recommendations(self, df):
        """Prepare features for recommendation models"""
        
        features = df.copy()
        
        # Channel preference features
        features['High_Response_Rate'] = (features['Response_Rate'] > 50).astype(int)
        features['Frequent_Caller'] = (features['Number_of_Calls'] > 10).astype(int)
        features['Recent_Payment'] = features['Payment_Made_Last_30_Days']
        
        # Timing features
        features['Urgency_Score'] = features['Days_Past_Due'] / 365
        features['Risk_Score'] = (features['Outstanding_Balance'] / features['Income']).fillna(0)
        
        # Demographic features for segmentation
        features['Age_Group'] = pd.cut(features['Age'], bins=[0, 30, 45, 60, 100], 
                                     labels=['Young', 'Middle', 'Senior', 'Elderly'])
        features['Income_Level'] = pd.cut(features['Income'], 
                                        bins=[0, 30000, 60000, 100000, float('inf')],
                                        labels=['Low', 'Medium', 'High', 'Very High'])
        
        return features
    
    def create_customer_segments(self, df, n_clusters=5):
        """Create customer segments for personalized recommendations"""
        
        # Select features for clustering
        clustering_features = [
            'Age', 'Income', 'Credit_Score', 'Days_Past_Due',
            'Response_Rate', 'Number_of_Calls', 'Outstanding_Balance'
        ]
        
        # Prepare data
        X_cluster = df[clustering_features].fillna(df[clustering_features].median())
        X_scaled = self.scaler.fit_transform(X_cluster)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Analyze segments
        df_with_clusters = df.copy()
        df_with_clusters['Segment'] = clusters
        
        segment_profiles = {}
        for segment in range(n_clusters):
            segment_data = df_with_clusters[df_with_clusters['Segment'] == segment]
            
            profile = {
                'size': len(segment_data),
                'avg_age': segment_data['Age'].mean(),
                'avg_income': segment_data['Income'].mean(),
                'avg_credit_score': segment_data['Credit_Score'].mean(),
                'avg_dpd': segment_data['Days_Past_Due'].mean(),
                'avg_response_rate': segment_data['Response_Rate'].mean(),
                'preferred_channel': segment_data['Last_Contact_Channel'].mode().iloc[0] if len(segment_data) > 0 else 'Call',
                'payment_rate': (segment_data['Outcome'] == 'Paid').mean() if 'Outcome' in segment_data.columns else 0.5
            }
            
            segment_profiles[f'Segment_{segment}'] = profile
        
        self.customer_segments = {
            'model': kmeans,
            'scaler': self.scaler,
            'profiles': segment_profiles,
            'features': clustering_features
        }
        
        return segment_profiles
    
    def train_channel_recommendation_model(self, df):
        """Train model to recommend optimal contact channel"""
        
        # Prepare features
        features_df = self.prepare_features_for_recommendations(df)
        
        # Create channel effectiveness scores based on historical data
        channel_effectiveness = {}
        
        for channel in df['Last_Contact_Channel'].unique():
            channel_data = df[df['Last_Contact_Channel'] == channel]
            
            # Calculate effectiveness metrics
            response_rate = channel_data['Response_Rate'].mean()
            payment_rate = (channel_data['Outcome'] == 'Paid').mean() if 'Outcome' in channel_data.columns else 0.5
            
            # Combined effectiveness score
            effectiveness = 0.6 * (response_rate / 100) + 0.4 * payment_rate
            channel_effectiveness[channel] = effectiveness
        
        # Rule-based channel recommendation
        def recommend_channel(row):
            # High-income, high credit score -> Email/WhatsApp
            if row.get('Income', 0) > 80000 and row.get('Credit_Score', 0) > 700:
                return 'Email' if np.random.random() > 0.5 else 'WhatsApp'
            
            # Young customers -> WhatsApp/SMS
            elif row.get('Age', 0) < 35:
                return 'WhatsApp' if np.random.random() > 0.4 else 'SMS'
            
            # High response rate customers -> preferred channel
            elif row.get('Response_Rate', 0) > 60:
                return row.get('Last_Contact_Channel', 'Call')
            
            # Default to most effective channel
            else:
                return max(channel_effectiveness, key=channel_effectiveness.get)
        
        self.channel_model = {
            'type': 'rule_based',
            'effectiveness': channel_effectiveness,
            'recommend_function': recommend_channel
        }
        
        return channel_effectiveness
    
    def train_timing_recommendation_model(self, df):
        """Train model to recommend optimal contact timing"""
        
        # Create timing recommendations based on customer characteristics
        timing_rules = {
            'morning': {'start': 9, 'end': 12, 'effectiveness': 0.7},
            'afternoon': {'start': 14, 'end': 17, 'effectiveness': 0.8},
            'evening': {'start': 18, 'end': 20, 'effectiveness': 0.6}
        }
        
        def recommend_timing(row):
            # Working professionals - evening
            if row.get('Occupation') in ['Salaried', 'Professional']:
                return 'evening'
            
            # Business owners - afternoon
            elif row.get('Occupation') == 'Business':
                return 'afternoon'
            
            # Self-employed - flexible, prefer afternoon
            elif row.get('Occupation') == 'Self-Employed':
                return 'afternoon' if np.random.random() > 0.3 else 'morning'
            
            # Retired - morning
            elif row.get('Occupation') == 'Retired':
                return 'morning'
            
            # Default to afternoon
            else:
                return 'afternoon'
        
        self.timing_model = {
            'type': 'rule_based',
            'rules': timing_rules,
            'recommend_function': recommend_timing
        }
        
        return timing_rules
    
    def get_channel_recommendation(self, customer_data):
        """Get channel recommendation for a customer"""
        
        if self.channel_model is None:
            return {'channel': 'Call', 'confidence': 0.5, 'reason': 'Default recommendation'}
        
        try:
            # Prepare customer features
            if isinstance(customer_data, pd.Series):
                customer_features = customer_data
            elif isinstance(customer_data, dict):
                customer_features = pd.Series(customer_data)
            else:
                customer_features = pd.Series(customer_data)
            
            # Get recommendation
            recommended_channel = self.channel_model['recommend_function'](customer_features)
            
            # Calculate confidence based on customer characteristics
            confidence = 0.7  # Base confidence
            
            # Adjust confidence based on data quality
            if pd.isna(customer_features.get('Response_Rate')):
                confidence -= 0.2
            elif customer_features.get('Response_Rate', 0) > 70:
                confidence += 0.2
            
            if customer_features.get('Number_of_Calls', 0) > 5:
                confidence += 0.1
            
            confidence = min(confidence, 0.95)
            
            # Generate reason
            reason = self._generate_channel_reason(customer_features, recommended_channel)
            
            return {
                'channel': recommended_channel,
                'confidence': confidence,
                'reason': reason,
                'alternatives': self._get_alternative_channels(recommended_channel)
            }
            
        except Exception as e:
            return {
                'channel': 'Call',
                'confidence': 0.5,
                'reason': f'Error in recommendation: {str(e)}',
                'alternatives': ['SMS', 'Email']
            }
    
    def get_timing_recommendation(self, customer_data):
        """Get timing recommendation for a customer"""
        
        if self.timing_model is None:
            return {'timing': 'afternoon', 'confidence': 0.5, 'reason': 'Default recommendation'}
        
        try:
            # Prepare customer features
            if isinstance(customer_data, pd.Series):
                customer_features = customer_data
            elif isinstance(customer_data, dict):
                customer_features = pd.Series(customer_data)
            else:
                customer_features = pd.Series(customer_data)
            
            # Get recommendation
            recommended_timing = self.timing_model['recommend_function'](customer_features)
            
            # Calculate confidence
            confidence = 0.8
            
            # Adjust based on occupation certainty
            if customer_features.get('Occupation') in ['Salaried', 'Professional', 'Retired']:
                confidence += 0.1
            
            confidence = min(confidence, 0.95)
            
            # Generate reason
            reason = self._generate_timing_reason(customer_features, recommended_timing)
            
            return {
                'timing': recommended_timing,
                'confidence': confidence,
                'reason': reason,
                'time_window': self.timing_model['rules'][recommended_timing]
            }
            
        except Exception as e:
            return {
                'timing': 'afternoon',
                'confidence': 0.5,
                'reason': f'Error in recommendation: {str(e)}',
                'time_window': {'start': 14, 'end': 17}
            }
    
    def get_comprehensive_recommendation(self, customer_data):
        """Get comprehensive contact recommendation"""
        
        channel_rec = self.get_channel_recommendation(customer_data)
        timing_rec = self.get_timing_recommendation(customer_data)
        
        # Get customer segment if available
        segment_info = None
        if self.customer_segments is not None:
            segment_info = self._get_customer_segment(customer_data)
        
        return {
            'customer_id': customer_data.get('Customer_ID', 'Unknown'),
            'channel_recommendation': channel_rec,
            'timing_recommendation': timing_rec,
            'segment_info': segment_info,
            'overall_confidence': (channel_rec['confidence'] + timing_rec['confidence']) / 2,
            'priority_score': self._calculate_priority_score(customer_data)
        }
    
    def _generate_channel_reason(self, customer_features, channel):
        """Generate explanation for channel recommendation"""
        
        reasons = []
        
        if channel == 'Email':
            if customer_features.get('Income', 0) > 80000:
                reasons.append("High income suggests email preference")
            if customer_features.get('Credit_Score', 0) > 700:
                reasons.append("Good credit score indicates digital comfort")
        
        elif channel == 'WhatsApp':
            if customer_features.get('Age', 0) < 35:
                reasons.append("Young demographic prefers WhatsApp")
            reasons.append("Modern communication preference")
        
        elif channel == 'SMS':
            if customer_features.get('Age', 0) < 40:
                reasons.append("Age group responds well to SMS")
            reasons.append("Quick and direct communication")
        
        elif channel == 'Call':
            if customer_features.get('Age', 0) > 50:
                reasons.append("Older demographic prefers voice calls")
            reasons.append("Personal touch for better engagement")
        
        return "; ".join(reasons) if reasons else "Based on customer profile analysis"
    
    def _generate_timing_reason(self, customer_features, timing):
        """Generate explanation for timing recommendation"""
        
        occupation = customer_features.get('Occupation', 'Unknown')
        
        timing_reasons = {
            'morning': f"Best for {occupation} - available before work hours",
            'afternoon': f"Optimal for {occupation} - flexible schedule",
            'evening': f"Suitable for {occupation} - after work hours"
        }
        
        return timing_reasons.get(timing, "Based on general effectiveness patterns")
    
    def _get_alternative_channels(self, primary_channel):
        """Get alternative channel recommendations"""
        
        all_channels = ['Call', 'SMS', 'WhatsApp', 'Email']
        alternatives = [ch for ch in all_channels if ch != primary_channel]
        
        return alternatives[:2]  # Return top 2 alternatives
    
    def _get_customer_segment(self, customer_data):
        """Get customer segment information"""
        
        if self.customer_segments is None:
            return None
        
        try:
            # Prepare features for clustering
            features = [customer_data.get(f, 0) for f in self.customer_segments['features']]
            features_scaled = self.customer_segments['scaler'].transform([features])
            
            # Predict segment
            segment = self.customer_segments['model'].predict(features_scaled)[0]
            segment_profile = self.customer_segments['profiles'][f'Segment_{segment}']
            
            return {
                'segment_id': segment,
                'segment_profile': segment_profile
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_priority_score(self, customer_data):
        """Calculate priority score for contact"""
        
        score = 0
        
        # Days past due (higher = more urgent)
        dpd = customer_data.get('Days_Past_Due', 0)
        score += min(dpd / 365, 1) * 40  # Max 40 points
        
        # Outstanding balance (higher = more important)
        balance = customer_data.get('Outstanding_Balance', 0)
        income = customer_data.get('Income', 1)
        debt_ratio = balance / income if income > 0 else 0
        score += min(debt_ratio, 2) * 20  # Max 20 points
        
        # Response rate (higher = more likely to engage)
        response_rate = customer_data.get('Response_Rate', 0)
        score += (response_rate / 100) * 20  # Max 20 points
        
        # Recent payment behavior
        if customer_data.get('Payment_Made_Last_30_Days', 0):
            score += 10
        
        # Credit score (lower = higher risk)
        credit_score = customer_data.get('Credit_Score', 600)
        score += max(0, (750 - credit_score) / 450) * 10  # Max 10 points
        
        return min(score, 100)  # Cap at 100
    
    def save_models(self, filepath):
        """Save recommendation models"""
        
        models_data = {
            'channel_model': self.channel_model,
            'timing_model': self.timing_model,
            'customer_segments': self.customer_segments
        }
        
        joblib.dump(models_data, filepath)
        self.logger.info(f"Recommendation models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load recommendation models"""
        
        models_data = joblib.load(filepath)
        
        self.channel_model = models_data['channel_model']
        self.timing_model = models_data['timing_model']
        self.customer_segments = models_data['customer_segments']
        
        self.logger.info(f"Recommendation models loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Test the recommendation engine
    recommender = RecommendationEngine()
    
    # Sample customer data
    sample_customer = {
        'Customer_ID': 'CUST_001234',
        'Age': 32,
        'Income': 75000,
        'Occupation': 'Salaried',
        'Outstanding_Balance': 25000,
        'Days_Past_Due': 45,
        'Response_Rate': 65.0,
        'Last_Contact_Channel': 'Call',
        'Credit_Score': 720,
        'Number_of_Calls': 8,
        'Payment_Made_Last_30_Days': 0
    }
    
    # Get recommendations
    recommendations = recommender.get_comprehensive_recommendation(sample_customer)
    
    print("=== CUSTOMER RECOMMENDATIONS ===")
    print(f"Customer ID: {recommendations['customer_id']}")
    print(f"Recommended Channel: {recommendations['channel_recommendation']['channel']}")
    print(f"Channel Confidence: {recommendations['channel_recommendation']['confidence']:.2f}")
    print(f"Channel Reason: {recommendations['channel_recommendation']['reason']}")
    print(f"Recommended Timing: {recommendations['timing_recommendation']['timing']}")
    print(f"Timing Confidence: {recommendations['timing_recommendation']['confidence']:.2f}")
    print(f"Priority Score: {recommendations['priority_score']:.1f}/100")
    print(f"Overall Confidence: {recommendations['overall_confidence']:.2f}")