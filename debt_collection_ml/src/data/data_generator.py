#!/usr/bin/env python3
"""
Synthetic Debt Collection Data Generator with DVC Support
Generates realistic synthetic data for debt collection ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebtDataGenerator:
    """Generate synthetic debt collection data with realistic patterns"""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator"""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define realistic value ranges and distributions
        self.debt_amount_ranges = {
            'small': (100, 1000),
            'medium': (1000, 10000),
            'large': (10000, 50000),
            'very_large': (50000, 200000)
        }
        
        self.age_ranges = {
            'young': (18, 30),
            'middle': (30, 50),
            'senior': (50, 70),
            'elderly': (70, 85)
        }
        
        self.income_ranges = {
            'low': (20000, 40000),
            'medium': (40000, 80000),
            'high': (80000, 150000),
            'very_high': (150000, 300000)
        }
    
    def generate_customer_demographics(self, n_samples: int) -> pd.DataFrame:
        """Generate customer demographic data"""
        
        # Age distribution (slightly skewed toward middle age)
        age_weights = [0.2, 0.4, 0.3, 0.1]  # young, middle, senior, elderly
        age_categories = np.random.choice(['young', 'middle', 'senior', 'elderly'], 
                                        size=n_samples, p=age_weights)
        
        ages = []
        for category in age_categories:
            min_age, max_age = self.age_ranges[category]
            ages.append(np.random.randint(min_age, max_age + 1))
        
        # Income correlated with age (middle-aged tend to have higher income)
        incomes = []
        for age in ages:
            if age < 30:
                income_range = self.income_ranges['low']
            elif age < 45:
                income_range = self.income_ranges['medium']
            elif age < 60:
                income_range = self.income_ranges['high']
            else:
                income_range = self.income_ranges['medium']  # Retirement income
            
            # Add some randomness
            base_income = np.random.uniform(income_range[0], income_range[1])
            # Add noise
            income = base_income * np.random.normal(1.0, 0.2)
            incomes.append(max(15000, income))  # Minimum income floor
        
        # Employment status correlated with age and income
        employment_status = []
        for age, income in zip(ages, incomes):
            if age < 25:
                status = np.random.choice(['employed', 'student', 'unemployed'], p=[0.6, 0.3, 0.1])
            elif age < 65:
                status = np.random.choice(['employed', 'self_employed', 'unemployed'], p=[0.8, 0.15, 0.05])
            else:
                status = np.random.choice(['retired', 'employed', 'unemployed'], p=[0.7, 0.2, 0.1])
            employment_status.append(status)
        
        # Credit score correlated with income and age
        credit_scores = []
        for age, income in zip(ages, incomes):
            # Base score from income
            base_score = 300 + (income / 300000) * 550  # Scale to 300-850 range
            
            # Age factor (older people tend to have better credit)
            age_factor = min(50, age - 18) * 2  # Up to 100 points for age
            
            # Add randomness
            score = base_score + age_factor + np.random.normal(0, 50)
            credit_scores.append(max(300, min(850, score)))
        
        return pd.DataFrame({
            'customer_age': ages,
            'annual_income': incomes,
            'employment_status': employment_status,
            'credit_score': credit_scores
        })
    
    def generate_debt_characteristics(self, n_samples: int, demographics: pd.DataFrame) -> pd.DataFrame:
        """Generate debt-related characteristics"""
        
        # Debt amount correlated with income
        debt_amounts = []
        debt_types = []
        
        for income in demographics['annual_income']:
            # Higher income people tend to have larger debts
            if income < 40000:
                debt_category = np.random.choice(['small', 'medium'], p=[0.7, 0.3])
            elif income < 80000:
                debt_category = np.random.choice(['small', 'medium', 'large'], p=[0.3, 0.5, 0.2])
            else:
                debt_category = np.random.choice(['medium', 'large', 'very_large'], p=[0.3, 0.5, 0.2])
            
            min_debt, max_debt = self.debt_amount_ranges[debt_category]
            debt_amount = np.random.uniform(min_debt, max_debt)
            debt_amounts.append(debt_amount)
            
            # Debt type based on amount
            if debt_amount < 5000:
                debt_type = np.random.choice(['credit_card', 'personal_loan', 'medical'], p=[0.5, 0.3, 0.2])
            elif debt_amount < 20000:
                debt_type = np.random.choice(['credit_card', 'personal_loan', 'auto_loan'], p=[0.4, 0.4, 0.2])
            else:
                debt_type = np.random.choice(['personal_loan', 'auto_loan', 'mortgage'], p=[0.4, 0.3, 0.3])
            
            debt_types.append(debt_type)
        
        # Days overdue - correlated with credit score and debt amount
        days_overdue = []
        for credit_score, debt_amount in zip(demographics['credit_score'], debt_amounts):
            # Lower credit score = more likely to be overdue longer
            base_overdue = max(0, (750 - credit_score) / 10)  # 0-45 days base
            
            # Larger debts tend to be overdue longer
            debt_factor = min(30, debt_amount / 1000)  # Up to 30 extra days
            
            # Add randomness
            overdue = base_overdue + debt_factor + np.random.exponential(10)
            days_overdue.append(max(1, min(365, overdue)))  # 1-365 days range
        
        # Previous payment history
        payment_history_scores = []
        for credit_score in demographics['credit_score']:
            # Correlated with credit score but with noise
            base_score = (credit_score - 300) / 550  # Normalize to 0-1
            history_score = base_score + np.random.normal(0, 0.2)
            payment_history_scores.append(max(0, min(1, history_score)))
        
        return pd.DataFrame({
            'debt_amount': debt_amounts,
            'debt_type': debt_types,
            'days_overdue': days_overdue,
            'payment_history_score': payment_history_scores
        })
    
    def generate_behavioral_features(self, n_samples: int, demographics: pd.DataFrame, 
                                   debt_chars: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral and interaction features"""
        
        # Contact attempts correlated with days overdue
        contact_attempts = []
        for days in debt_chars['days_overdue']:
            # More attempts for longer overdue periods
            base_attempts = min(20, days / 10)
            attempts = max(1, base_attempts + np.random.poisson(2))
            contact_attempts.append(attempts)
        
        # Response rate correlated with credit score and age
        response_rates = []
        for credit_score, age in zip(demographics['credit_score'], demographics['customer_age']):
            # Higher credit score = more responsive
            base_rate = (credit_score - 300) / 550
            
            # Older people tend to be more responsive
            age_factor = min(0.2, (age - 18) / 200)
            
            rate = base_rate + age_factor + np.random.normal(0, 0.15)
            response_rates.append(max(0, min(1, rate)))
        
        # Promise to pay indicator
        promise_to_pay = []
        for response_rate in response_rates:
            # More responsive people more likely to make promises
            prob = response_rate * 0.7  # 70% of responsive people make promises
            promise = np.random.random() < prob
            promise_to_pay.append(promise)
        
        # Number of previous promises
        previous_promises = []
        for promise, days_overdue in zip(promise_to_pay, debt_chars['days_overdue']):
            if promise:
                # More promises for longer overdue periods
                promises = max(0, int(days_overdue / 30) + np.random.poisson(1))
            else:
                promises = 0
            previous_promises.append(promises)
        
        return pd.DataFrame({
            'contact_attempts': contact_attempts,
            'response_rate': response_rates,
            'promise_to_pay': promise_to_pay,
            'previous_promises': previous_promises
        })
    
    def generate_target_variable(self, demographics: pd.DataFrame, debt_chars: pd.DataFrame, 
                               behavioral: pd.DataFrame) -> pd.Series:
        """Generate the target variable (payment status)"""
        
        payment_statuses = []
        
        for i in range(len(demographics)):
            # Factors influencing payment probability
            credit_score = demographics.iloc[i]['credit_score']
            income = demographics.iloc[i]['annual_income']
            debt_amount = debt_chars.iloc[i]['debt_amount']
            days_overdue = debt_chars.iloc[i]['days_overdue']
            payment_history = debt_chars.iloc[i]['payment_history_score']
            response_rate = behavioral.iloc[i]['response_rate']
            
            # Calculate payment probability
            # Higher credit score = higher payment probability
            credit_factor = (credit_score - 300) / 550
            
            # Higher income relative to debt = higher payment probability
            debt_to_income = debt_amount / income
            income_factor = max(0, 1 - debt_to_income)
            
            # Fewer days overdue = higher payment probability
            overdue_factor = max(0, 1 - days_overdue / 365)
            
            # Combine factors
            payment_prob = (
                0.3 * credit_factor +
                0.2 * income_factor +
                0.2 * overdue_factor +
                0.15 * payment_history +
                0.15 * response_rate
            )
            
            # Add some randomness
            payment_prob += np.random.normal(0, 0.1)
            payment_prob = max(0, min(1, payment_prob))
            
            # Convert to categorical outcome
            if payment_prob > 0.7:
                status = 'full_payment'
            elif payment_prob > 0.4:
                status = 'partial_payment'
            else:
                status = 'no_payment'
            
            payment_statuses.append(status)
        
        return pd.Series(payment_statuses, name='payment_status')
    
    def generate_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        
        logger.info(f"Generating {n_samples} synthetic debt collection samples...")
        
        # Generate each component
        demographics = self.generate_customer_demographics(n_samples)
        debt_chars = self.generate_debt_characteristics(n_samples, demographics)
        behavioral = self.generate_behavioral_features(n_samples, demographics, debt_chars)
        target = self.generate_target_variable(demographics, debt_chars, behavioral)
        
        # Combine all features
        df = pd.concat([demographics, debt_chars, behavioral, target], axis=1)
        
        # Add customer ID
        df.insert(0, 'customer_id', [f'CUST_{i:06d}' for i in range(n_samples)])
        
        # Add some derived features
        df['debt_to_income_ratio'] = df['debt_amount'] / df['annual_income']
        df['contact_success_rate'] = df['response_rate'] * np.random.uniform(0.8, 1.2, n_samples)
        df['contact_success_rate'] = df['contact_success_rate'].clip(0, 1)
        
        logger.info(f"Generated dataset with {len(df)} samples and {len(df.columns)} features")
        logger.info(f"Target distribution:\n{df['payment_status'].value_counts()}")
        
        return df

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Generate synthetic debt collection data')
    parser.add_argument('--output', type=str, default='data/raw/synthetic_debt_data.csv',
                       help='Output file path for the generated data')
    parser.add_argument('--size', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Generate synthetic data
    logger.info(f"Starting data generation with {args.size} samples...")
    
    generator = DebtDataGenerator(random_state=args.random_state)
    df = generator.generate_data(n_samples=args.size)
    
    # Save the data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Generated {len(df)} samples")
    logger.info(f"📁 Saved to: {output_path}")
    logger.info(f"📊 Columns: {list(df.columns)}")
    
    # Display basic statistics
    print(f"\n📈 Dataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nTarget distribution:")
    print(df['payment_status'].value_counts())
    print(f"\nBasic statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()