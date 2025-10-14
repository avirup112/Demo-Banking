#!/usr/bin/env python3
"""
Web Scraping Script for DVC Pipeline
"""

import sys
import os
import yaml
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.web_scraper import WebScraper

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['web_scraping']

def main():
    """Perform web scraping based on DVC parameters"""
    
    # Load parameters
    params = load_params()
    
    print("Starting web scraping...")
    
    # Create output directories
    Path('data/external').mkdir(parents=True, exist_ok=True)
    Path('reports').mkdir(parents=True, exist_ok=True)
    
    # Check if scraping is enabled
    if not params.get('enable_scraping', False):
        print("Web scraping is disabled in parameters")
        
        # Create empty output file and metrics
        df = pd.read_csv('data/raw/debt_collection_data.csv')
        df.to_csv('data/external/enriched_data.csv', index=False)
        
        metrics = {
            'scraping_enabled': False,
            'records_processed': len(df),
            'external_features_added': 0,
            'scraping_errors': 0
        }
        
        with open('reports/web_scraping_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("Web scraping skipped - original data copied to external folder")
        return
    
    # Load original data
    df = pd.read_csv('data/raw/debt_collection_data.csv')
    print(f"Loaded dataset: {df.shape}")
    
    # Initialize web scraper
    scraper = WebScraper(delay_range=tuple(params['delay_range']))
    
    try:
        # Enrich data with external information
        enriched_df = scraper.enrich_customer_data(df)
        
        # Save enriched data
        enriched_df.to_csv('data/external/enriched_data.csv', index=False)
        
        # Calculate metrics
        original_features = len(df.columns)
        enriched_features = len(enriched_df.columns)
        features_added = enriched_features - original_features
        
        metrics = {
            'scraping_enabled': True,
            'records_processed': len(df),
            'original_features': original_features,
            'enriched_features': enriched_features,
            'external_features_added': features_added,
            'scraping_errors': 0,
            'scraping_config': params
        }
        
        print(f"Web scraping completed successfully")
        print(f"Original features: {original_features}")
        print(f"Enriched features: {enriched_features}")
        print(f"Features added: {features_added}")
        
    except Exception as e:
        print(f"Web scraping failed: {e}")
        
        # Fallback: copy original data
        df.to_csv('data/external/enriched_data.csv', index=False)
        
        metrics = {
            'scraping_enabled': True,
            'records_processed': len(df),
            'original_features': len(df.columns),
            'enriched_features': len(df.columns),
            'external_features_added': 0,
            'scraping_errors': 1,
            'error_message': str(e)
        }
    
    finally:
        # Clean up scraper resources
        scraper.close_session()
    
    # Save metrics
    with open('reports/web_scraping_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()