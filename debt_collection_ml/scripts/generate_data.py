#!/usr/bin/env python3
import sys
import os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_generator import DebtCollectionDataGenerator

def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    gen_params = params['generate_data']
    
    # Generate data
    generator = DebtCollectionDataGenerator(
        n_samples=gen_params['n_samples'],
        random_state=gen_params['random_state']
    )
    
    df = generator.generate_dataset()
    
    # Save data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/debt_collection_data.csv', index=False)
    
    print(f"Generated {len(df)} samples")

if __name__ == "__main__":
    main()