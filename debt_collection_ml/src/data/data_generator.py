import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DebtCollectionDataGenerator:
    """
    Generate synthetic debt collection dataset
    """
    
    def __init__(self, n_samples=1000, random_state=42):
        self.n_samples = n_samples
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_dataset(self):
        """
        Generate complete synthetic datatset
        """
        
        # Basic customer info
        customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, self.n_samples + 1)]
        bank_codes = np.random.choice(['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK', 'BANDHAN', 'UCO'], self.n_samples)
        
        # Demographics
        ages = np.random.normal(35,12, self.n_samples).astype(int)
        ages = np.clip(ages,18,80)
        
        incomes = np.random.lognormal(10.5, 0.8, self.n_samples)
        incomes = np.clip(incomes, 15000, 500000)
        
        occupations = np.random.choice([
            'Salaried', 'Self-Employed', 'Business', 'Professional', 'Retired'
        ], self.n_samples, p=[0.4, 0.25, 0.15, 0.15, 0.05])
        
        # Financial attributes
        loan_amounts = np.random.lognormal(11, 0.6, self.n_samples)
        loan_amounts = np.clip(loan_amounts, 50000, 2000000)
        
        # Outstanding balance (correlated with loan amount)
        outstanding_ratios = np.random.beta(2, 3, self.n_samples)
        outstanding_balances = loan_amounts * outstanding_ratios
        
          # Days past due
        days_past_due = np.random.exponential(45, self.n_samples).astype(int)
        days_past_due = np.clip(days_past_due, 1, 365)
        
        # Communication history
        number_of_calls = np.random.poisson(8, self.n_samples)
        response_rates = np.random.beta(2, 5, self.n_samples) * 100
        
        last_contact_channels = np.random.choice([
            'Call', 'SMS', 'WhatsApp', 'Email'
        ], self.n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Recent payment behavior
        payment_made_last_30_days = np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3])
        
        # Geographic
        regions = np.random.choice(['East', 'West', 'North', 'South'], self.n_samples)
        
        # Credit scores (correlated with other factors)
        base_scores = 300 + (incomes / 1000) + (ages * 2) - (days_past_due * 0.5)
        credit_scores = base_scores + np.random.normal(0, 50, self.n_samples)
        credit_scores = np.clip(credit_scores, 300, 850).astype(int)
        
        # Complaint flags
        complaint_flags = np.random.choice([0, 1], self.n_samples, p=[0.85, 0.15])
        
        # Generate outcome based on features (realistic correlations)
        outcome_probs = self._calculate_outcome_probabilities(
            ages, incomes, credit_scores, days_past_due, 
            response_rates, payment_made_last_30_days, complaint_flags
        )
        
        outcomes = []
        for prob in outcome_probs:
            rand = np.random.random()
            if rand < prob[0]:
                outcomes.append('Paid')
            elif rand < prob[0] + prob[1]:
                outcomes.append('Partially Paid')
            else:
                outcomes.append('Not Paid')
        
        # Create DataFrame
        df = pd.DataFrame({
            'Customer_ID': customer_ids,
            'Bank_Code': bank_codes,
            'Age': ages,
            'Income': incomes.astype(int),
            'Occupation': occupations,
            'Loan_Amount': loan_amounts.astype(int),
            'Outstanding_Balance': outstanding_balances.astype(int),
            'Days_Past_Due': days_past_due,
            'Number_of_Calls': number_of_calls,
            'Response_Rate': response_rates.round(2),
            'Last_Contact_Channel': last_contact_channels,
            'Payment_Made_Last_30_Days': payment_made_last_30_days,
            'Region': regions,
            'Credit_Score': credit_scores,
            'Complaint_Flag': complaint_flags,
            'Outcome': outcomes
        })
        
        # Add some missing values to make it realistic
        df = self._add_missing_values(df)
        
        return df
    
    def _calculate_outcome_probabilities(self, ages, incomes, credit_scores, 
                                       days_past_due, response_rates, 
                                       payment_made_last_30_days, complaint_flags):
        """Calculate realistic outcome probabilities based on features"""
        
        # Normalize features
        age_norm = (ages - 18) / (80 - 18)
        income_norm = (incomes - 15000) / (500000 - 15000)
        credit_norm = (credit_scores - 300) / (850 - 300)
        dpd_norm = days_past_due / 365
        response_norm = response_rates / 100
        
        # Calculate base probability for 'Paid'
        paid_prob = (
            0.3 * credit_norm +
            0.2 * income_norm +
            0.15 * (1 - dpd_norm) +
            0.15 * response_norm +
            0.1 * payment_made_last_30_days +
            0.05 * (1 - complaint_flags) +
            0.05 * age_norm
        )
        
        # Adjust probabilities
        paid_prob = np.clip(paid_prob, 0.05, 0.8)
        partial_prob = np.random.uniform(0.1, 0.3, len(paid_prob))
        not_paid_prob = 1 - paid_prob - partial_prob
        
        # Ensure probabilities sum to 1
        total_prob = paid_prob + partial_prob + not_paid_prob
        paid_prob = paid_prob / total_prob
        partial_prob = partial_prob / total_prob
        
        return list(zip(paid_prob, partial_prob))
    
    def _add_missing_values(self, df):
        """Add realistic missing values"""
        
        # Add missing values to some columns
        missing_cols = ['Income', 'Credit_Score', 'Response_Rate']
        
        for col in missing_cols:
            if col in df.columns:
                missing_mask = np.random.random(len(df)) < 0.05  # 5% missing
                df.loc[missing_mask, col] = np.nan
        
        return df