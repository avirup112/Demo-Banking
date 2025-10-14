#!/usr/bin/env python3
"""
Contact Recommendations Engine for Debt Collection
Rule-based and ML-driven recommendations for optimal contact strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContactRecommendationEngine:
    """Intelligent contact recommendations for debt collection"""
    
    def __init__(self):
        """Initialize the recommendation engine"""
        
        # Contact channels with effectiveness scores
        self.contact_channels = {
            'phone': {'effectiveness': 0.75, 'cost': 5.0, 'response_time': 'immediate'},
            'email': {'effectiveness': 0.45, 'cost': 1.0, 'response_time': '24-48h'},
            'sms': {'effectiveness': 0.65, 'cost': 2.0, 'response_time': '1-4h'},
            'letter': {'effectiveness': 0.35, 'cost': 8.0, 'response_time': '5-7 days'},
            'legal_notice': {'effectiveness': 0.85, 'cost': 50.0, 'response_time': '7-14 days'}
        }
        
        # Time preferences based on customer segments
        self.time_preferences = {
            'employed': {'best_hours': [9, 10, 17, 18, 19], 'avoid_hours': [11, 12, 13, 14, 15, 16]},
            'unemployed': {'best_hours': [10, 11, 14, 15, 16], 'avoid_hours': [8, 9, 17, 18, 19]},
            'retired': {'best_hours': [9, 10, 11, 14, 15], 'avoid_hours': [12, 13, 17, 18, 19, 20]},
            'self_employed': {'best_hours': [9, 10, 17, 18, 19, 20], 'avoid_hours': [11, 12, 13, 14, 15, 16]}
        }
        
        logger.info("Contact Recommendation Engine initialized")
    
    def get_channel_recommendation(self, customer_profile: Dict) -> Dict[str, any]:
        """Recommend best contact channel based on customer profile"""
        
        # Extract customer characteristics
        payment_status = customer_profile.get('payment_status', 'unknown')
        days_overdue = customer_profile.get('days_overdue', 0)
        previous_contact_success = customer_profile.get('contact_success_rate', 0.5)
        debt_amount = customer_profile.get('debt_amount', 0)
        customer_age = customer_profile.get('customer_age', 45)
        employment_status = customer_profile.get('employment_status', 'employed')
        
        # Rule-based channel selection
        recommendations = []
        
        # High urgency cases (long overdue or high amount)
        if days_overdue > 90 or debt_amount > 10000:
            if debt_amount > 25000:
                recommendations.append({
                    'channel': 'legal_notice',
                    'priority': 1,
                    'reason': 'High debt amount requires legal intervention',
                    'expected_effectiveness': self.contact_channels['legal_notice']['effectiveness'],
                    'cost': self.contact_channels['legal_notice']['cost']
                })
            
            recommendations.append({
                'channel': 'phone',
                'priority': 2,
                'reason': 'Urgent case requires immediate contact',
                'expected_effectiveness': self.contact_channels['phone']['effectiveness'],
                'cost': self.contact_channels['phone']['cost']
            })
        
        # Medium urgency cases
        elif days_overdue > 30:
            if previous_contact_success > 0.6:
                recommendations.append({
                    'channel': 'phone',
                    'priority': 1,
                    'reason': 'Good response history suggests phone contact',
                    'expected_effectiveness': self.contact_channels['phone']['effectiveness'] * 1.2,
                    'cost': self.contact_channels['phone']['cost']
                })
            else:
                recommendations.append({
                    'channel': 'sms',
                    'priority': 1,
                    'reason': 'SMS for better response with poor phone history',
                    'expected_effectiveness': self.contact_channels['sms']['effectiveness'],
                    'cost': self.contact_channels['sms']['cost']
                })
        
        # Low urgency cases
        else:
            if customer_age < 40:
                recommendations.append({
                    'channel': 'sms',
                    'priority': 1,
                    'reason': 'Younger customers prefer SMS communication',
                    'expected_effectiveness': self.contact_channels['sms']['effectiveness'] * 1.1,
                    'cost': self.contact_channels['sms']['cost']
                })
                
                recommendations.append({
                    'channel': 'email',
                    'priority': 2,
                    'reason': 'Digital-native demographic',
                    'expected_effectiveness': self.contact_channels['email']['effectiveness'] * 1.1,
                    'cost': self.contact_channels['email']['cost']
                })
            else:
                recommendations.append({
                    'channel': 'phone',
                    'priority': 1,
                    'reason': 'Traditional contact method for older customers',
                    'expected_effectiveness': self.contact_channels['phone']['effectiveness'],
                    'cost': self.contact_channels['phone']['cost']
                })
        
        # Always add email as backup
        if not any(r['channel'] == 'email' for r in recommendations):
            recommendations.append({
                'channel': 'email',
                'priority': 3,
                'reason': 'Cost-effective backup option',
                'expected_effectiveness': self.contact_channels['email']['effectiveness'],
                'cost': self.contact_channels['email']['cost']
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return {
            'customer_id': customer_profile.get('customer_id', 'unknown'),
            'recommendations': recommendations,
            'primary_channel': recommendations[0]['channel'] if recommendations else 'phone',
            'urgency_level': self._calculate_urgency(days_overdue, debt_amount),
            'generated_at': datetime.now().isoformat()
        }
    
    def get_timing_recommendation(self, customer_profile: Dict) -> Dict[str, any]:
        """Recommend best contact timing based on customer profile"""
        
        employment_status = customer_profile.get('employment_status', 'employed')
        customer_age = customer_profile.get('customer_age', 45)
        days_overdue = customer_profile.get('days_overdue', 0)
        
        # Get base time preferences
        time_prefs = self.time_preferences.get(employment_status, self.time_preferences['employed'])
        
        # Adjust for urgency
        if days_overdue > 60:
            # Urgent cases - expand contact hours
            best_hours = list(range(9, 20))  # 9 AM to 8 PM
            avoid_hours = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            best_hours = time_prefs['best_hours']
            avoid_hours = time_prefs['avoid_hours']
        
        # Day of week recommendations
        if employment_status == 'employed':
            best_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            avoid_days = ['Saturday', 'Sunday']
        else:
            best_days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            avoid_days = ['Monday', 'Sunday']
        
        # Calculate next best contact time
        now = datetime.now()
        next_contact_times = []
        
        for day_offset in range(7):  # Next 7 days
            target_date = now + timedelta(days=day_offset)
            day_name = target_date.strftime('%A')
            
            if day_name not in avoid_days:
                for hour in best_hours:
                    contact_time = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    if contact_time > now:
                        next_contact_times.append({
                            'datetime': contact_time.isoformat(),
                            'day': day_name,
                            'hour': hour,
                            'score': self._calculate_time_score(day_name, hour, employment_status)
                        })
        
        # Sort by score (best times first)
        next_contact_times.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'customer_id': customer_profile.get('customer_id', 'unknown'),
            'best_hours': best_hours,
            'avoid_hours': avoid_hours,
            'best_days': best_days,
            'avoid_days': avoid_days,
            'next_best_times': next_contact_times[:5],  # Top 5 recommendations
            'employment_status': employment_status,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_comprehensive_recommendation(self, customer_profile: Dict) -> Dict[str, any]:
        """Get comprehensive contact strategy recommendation"""
        
        channel_rec = self.get_channel_recommendation(customer_profile)
        timing_rec = self.get_timing_recommendation(customer_profile)
        
        # Generate contact strategy
        strategy = self._generate_contact_strategy(customer_profile, channel_rec, timing_rec)
        
        return {
            'customer_id': customer_profile.get('customer_id', 'unknown'),
            'channel_recommendations': channel_rec,
            'timing_recommendations': timing_rec,
            'contact_strategy': strategy,
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_urgency(self, days_overdue: int, debt_amount: float) -> str:
        """Calculate urgency level"""
        
        urgency_score = 0
        
        # Days overdue factor
        if days_overdue > 90:
            urgency_score += 3
        elif days_overdue > 60:
            urgency_score += 2
        elif days_overdue > 30:
            urgency_score += 1
        
        # Debt amount factor
        if debt_amount > 25000:
            urgency_score += 3
        elif debt_amount > 10000:
            urgency_score += 2
        elif debt_amount > 5000:
            urgency_score += 1
        
        if urgency_score >= 4:
            return 'critical'
        elif urgency_score >= 2:
            return 'high'
        elif urgency_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_time_score(self, day_name: str, hour: int, employment_status: str) -> float:
        """Calculate score for specific day/time combination"""
        
        score = 1.0
        
        # Day preference
        if employment_status == 'employed':
            if day_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                score += 0.5
            else:
                score -= 0.3
        else:
            if day_name in ['Tuesday', 'Wednesday', 'Thursday']:
                score += 0.3
        
        # Hour preference
        time_prefs = self.time_preferences.get(employment_status, self.time_preferences['employed'])
        if hour in time_prefs['best_hours']:
            score += 0.5
        elif hour in time_prefs['avoid_hours']:
            score -= 0.5
        
        return max(0.1, score)  # Minimum score of 0.1
    
    def _generate_contact_strategy(self, customer_profile: Dict, 
                                 channel_rec: Dict, timing_rec: Dict) -> Dict[str, any]:
        """Generate comprehensive contact strategy"""
        
        days_overdue = customer_profile.get('days_overdue', 0)
        debt_amount = customer_profile.get('debt_amount', 0)
        payment_status = customer_profile.get('payment_status', 'unknown')
        
        strategy = {
            'approach': 'standard',
            'frequency': 'weekly',
            'escalation_timeline': '30_days',
            'special_instructions': []
        }
        
        # Adjust strategy based on profile
        if days_overdue > 90:
            strategy['approach'] = 'aggressive'
            strategy['frequency'] = 'daily'
            strategy['escalation_timeline'] = '7_days'
            strategy['special_instructions'].append('Consider legal action')
            strategy['special_instructions'].append('Offer payment plan immediately')
        
        elif days_overdue > 60:
            strategy['approach'] = 'firm'
            strategy['frequency'] = 'every_3_days'
            strategy['escalation_timeline'] = '14_days'
            strategy['special_instructions'].append('Emphasize consequences')
            strategy['special_instructions'].append('Negotiate payment terms')
        
        elif days_overdue > 30:
            strategy['approach'] = 'persuasive'
            strategy['frequency'] = 'twice_weekly'
            strategy['escalation_timeline'] = '21_days'
            strategy['special_instructions'].append('Focus on relationship')
            strategy['special_instructions'].append('Offer incentives')
        
        # High-value accounts get special treatment
        if debt_amount > 15000:
            strategy['special_instructions'].append('Assign senior collector')
            strategy['special_instructions'].append('Consider settlement options')
        
        # Payment status specific instructions
        if payment_status == 'partial_payment':
            strategy['special_instructions'].append('Acknowledge partial payments')
            strategy['special_instructions'].append('Encourage completion')
        elif payment_status == 'no_payment':
            strategy['special_instructions'].append('Investigate payment ability')
            strategy['special_instructions'].append('Consider hardship programs')
        
        return strategy
    
    def batch_recommendations(self, customer_profiles: List[Dict]) -> List[Dict]:
        """Generate recommendations for multiple customers"""
        
        recommendations = []
        
        for profile in customer_profiles:
            try:
                rec = self.get_comprehensive_recommendation(profile)
                recommendations.append(rec)
            except Exception as e:
                logger.error(f"Failed to generate recommendation for customer {profile.get('customer_id', 'unknown')}: {e}")
                recommendations.append({
                    'customer_id': profile.get('customer_id', 'unknown'),
                    'error': str(e),
                    'generated_at': datetime.now().isoformat()
                })
        
        return recommendations
    
    def save_recommendations(self, recommendations: List[Dict], filename: str = None):
        """Save recommendations to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"contact_recommendations_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"Recommendations saved to {filename}")

def main():
    """Example usage of the recommendation engine"""
    
    # Example customer profile
    customer_profile = {
        'customer_id': 'CUST_001234',
        'customer_age': 35,
        'employment_status': 'employed',
        'payment_status': 'partial_payment',
        'days_overdue': 45,
        'debt_amount': 8500,
        'contact_success_rate': 0.7,
        'previous_promises': 2
    }
    
    # Initialize engine
    engine = ContactRecommendationEngine()
    
    # Get recommendations
    recommendations = engine.get_comprehensive_recommendation(customer_profile)
    
    print("📞 CONTACT RECOMMENDATIONS")
    print("=" * 50)
    print(f"Customer: {recommendations['customer_id']}")
    print(f"Primary Channel: {recommendations['channel_recommendations']['primary_channel']}")
    print(f"Urgency: {recommendations['channel_recommendations']['urgency_level']}")
    print(f"Best Contact Times: {[t['datetime'][:16] for t in recommendations['timing_recommendations']['next_best_times'][:3]]}")
    print(f"Strategy: {recommendations['contact_strategy']['approach']} approach")
    print(f"Frequency: {recommendations['contact_strategy']['frequency']}")

if __name__ == "__main__":
    main()