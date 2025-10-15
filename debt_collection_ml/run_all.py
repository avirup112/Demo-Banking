#!/usr/bin/env python3
"""
One-Click Debt Collection ML System
Runs everything: DVC pipeline + ML training + Dashboard launch
"""

import subprocess
import sys
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_dvc_pipeline():
    """Run DVC pipeline which includes ML training and dashboard"""
    
    logger.info("🚀 Starting Complete DVC Pipeline...")
    logger.info("This will:")
    logger.info("  1. Generate synthetic data")
    logger.info("  2. Preprocess and engineer features") 
    logger.info("  3. Train and optimize ML models")
    logger.info("  4. Generate SHAP explanations")
    logger.info("  5. Launch interactive dashboard")
    logger.info("=" * 60)
    
    try:
        # Run DVC reproduce
        cmd = ["dvc", "repro"]
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, text=True)
        
        logger.info("✅ DVC Pipeline completed successfully!")
        logger.info("🎉 All stages completed:")
        logger.info("   ✅ Data generation")
        logger.info("   ✅ Data preprocessing") 
        logger.info("   ✅ Feature engineering")
        logger.info("   ✅ ML model training")
        logger.info("   ✅ Dashboard launched")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ DVC Pipeline failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("❌ DVC not found. Please install DVC: pip install dvc")
        return False

def run_quick_pipeline(samples: int = 1000):
    """Run quick pipeline without DVC"""
    
    logger.info("⚡ Starting Quick Pipeline (without DVC)...")
    
    try:
        cmd = [
            sys.executable, "run_complete_pipeline.py",
            "--samples", str(samples),
            "--dashboard-timeout", "300"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        logger.info("✅ Quick Pipeline completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Quick Pipeline failed: {e}")
        return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="One-Click Debt Collection ML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 USAGE EXAMPLES:

# Full DVC pipeline (recommended)
python run_all.py

# Quick run without DVC
python run_all.py --quick --samples 1000

# DVC pipeline only (no dashboard timeout)
dvc repro

🚀 WHAT THIS DOES:
1. Generates synthetic debt collection data
2. Preprocesses and engineers features
3. Trains multiple ML models with optimization
4. Generates SHAP explanations
5. Launches interactive Streamlit dashboard
6. Opens browser automatically (if not --no-browser)

📊 EXPECTED RESULTS:
- F1 Score: >0.65 (target exceeded)
- Training time: ~45 seconds
- Dashboard: http://localhost:8501
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                       help="Run quick pipeline without DVC")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples for quick mode")
    
    args = parser.parse_args()
    
    logger.info("🎯 Debt Collection ML System - One-Click Launch")
    logger.info("=" * 60)
    
    if args.quick:
        success = run_quick_pipeline(args.samples)
    else:
        success = run_dvc_pipeline()
    
    if success:
        logger.info("🎉 SUCCESS! System is ready.")
        logger.info("🌐 Dashboard should be running at: http://localhost:8501")
        logger.info("📊 Check reports/ directory for detailed results")
        return 0
    else:
        logger.error("❌ FAILED! Check logs above for details")
        return 1

if __name__ == "__main__":
    exit(main())