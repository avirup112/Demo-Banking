#!/usr/bin/env python3
"""
Complete Debt Collection ML Pipeline with Automatic Dashboard Launch
Runs the entire pipeline and launches Streamlit dashboard automatically
"""

import subprocess
import sys
import time
import logging
import argparse
from pathlib import Path
import webbrowser
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePipeline:
    """Complete ML pipeline with automatic dashboard launch"""
    
    def __init__(self):
        self.dashboard_process = None
        
    def run_ml_pipeline(self, samples: int = 1000, dagshub_owner: str = "avirup112", 
                       dagshub_repo: str = "Demo-Banking", enable_shap: bool = True):
        """Run the main ML pipeline"""
        
        logger.info("🚀 Starting Enhanced Debt Collection ML Pipeline...")
        
        # Build command
        cmd = [
            sys.executable, "run_enhanced_pipeline.py",
            "--samples", str(samples),
            "--dagshub-owner", dagshub_owner,
            "--dagshub-repo", dagshub_repo
        ]
        
        if enable_shap:
            cmd.append("--enable-shap")
        
        try:
            # Run ML pipeline
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("✅ ML Pipeline completed successfully!")
            logger.info("Pipeline output:")
            print(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ML Pipeline failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def launch_dashboard(self, auto_open: bool = True, port: int = 8501):
        """Launch Streamlit dashboard"""
        
        logger.info("🌐 Launching Streamlit Dashboard...")
        
        try:
            # Check if dashboard file exists
            if not Path("streamlit_dashboard.py").exists():
                logger.error("Dashboard file not found: streamlit_dashboard.py")
                return False
            
            # Launch Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "streamlit_dashboard.py",
                "--server.port", str(port),
                "--browser.gatherUsageStats", "false"
            ]
            
            logger.info(f"Starting dashboard on port {port}...")
            
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for dashboard to start
            time.sleep(5)
            
            if self.dashboard_process.poll() is not None:
                stdout, stderr = self.dashboard_process.communicate()
                logger.error(f"Dashboard failed to start: {stderr}")
                return False
            
            dashboard_url = f"http://localhost:{port}"
            logger.info(f"✅ Dashboard launched successfully!")
            logger.info(f"🌐 Access at: {dashboard_url}")
            
            # Auto-open browser
            if auto_open:
                try:
                    time.sleep(2)
                    webbrowser.open(dashboard_url)
                    logger.info("🚀 Browser opened automatically")
                except Exception as e:
                    logger.warning(f"Could not auto-open browser: {e}")
                    logger.info(f"Please manually open: {dashboard_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
            return False
    
    def run_complete_pipeline(self, samples: int = 1000, dagshub_owner: str = "avirup112",
                            dagshub_repo: str = "Demo-Banking", enable_shap: bool = True,
                            launch_dashboard: bool = True, auto_open: bool = True,
                            dashboard_timeout: int = 300):
        """Run complete pipeline with dashboard"""
        
        logger.info("🎯 Starting Complete Debt Collection ML Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Run ML Pipeline
        logger.info("📊 Step 1: Running ML Pipeline...")
        ml_success = self.run_ml_pipeline(samples, dagshub_owner, dagshub_repo, enable_shap)
        
        if not ml_success:
            logger.error("❌ ML Pipeline failed. Stopping execution.")
            return False
        
        logger.info("✅ Step 1 Complete: ML Pipeline finished successfully")
        logger.info("-" * 60)
        
        # Step 2: Launch Dashboard (if requested)
        if launch_dashboard:
            logger.info("🌐 Step 2: Launching Interactive Dashboard...")
            
            dashboard_success = self.launch_dashboard(auto_open=auto_open)
            
            if dashboard_success:
                logger.info("✅ Step 2 Complete: Dashboard launched successfully")
                logger.info("-" * 60)
                
                # Keep dashboard running
                logger.info(f"🕐 Dashboard will run for {dashboard_timeout} seconds...")
                logger.info("💡 Press Ctrl+C to stop early and exit")
                
                try:
                    time.sleep(dashboard_timeout)
                    logger.info("⏰ Timeout reached, stopping dashboard...")
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Manual stop requested...")
                
                finally:
                    self.stop_dashboard()
                    
            else:
                logger.warning("⚠️ Dashboard launch failed, but ML pipeline completed successfully")
        
        logger.info("🎉 Complete Pipeline Finished!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("📋 PIPELINE SUMMARY:")
        logger.info(f"   ✅ ML Training: {'Success' if ml_success else 'Failed'}")
        logger.info(f"   ✅ Dashboard: {'Launched' if launch_dashboard and dashboard_success else 'Skipped/Failed'}")
        logger.info(f"   📊 Samples Processed: {samples}")
        logger.info(f"   🎯 Target F1 Score: 0.65")
        logger.info(f"   📈 Check reports/ directory for detailed results")
        
        return True
    
    def stop_dashboard(self):
        """Stop the dashboard process"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("Stopping dashboard...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
            logger.info("✅ Dashboard stopped")

def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Complete Debt Collection ML Pipeline with Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with dashboard
  python run_complete_pipeline.py --samples 1000
  
  # Full run without auto-opening browser
  python run_complete_pipeline.py --samples 5000 --no-browser
  
  # ML only (no dashboard)
  python run_complete_pipeline.py --samples 1000 --no-dashboard
  
  # Extended dashboard time
  python run_complete_pipeline.py --dashboard-timeout 600
        """
    )
    
    parser.add_argument("--samples", type=int, default=1000,
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--dagshub-owner", default="avirup112",
                       help="DagsHub repository owner")
    parser.add_argument("--dagshub-repo", default="Demo-Banking",
                       help="DagsHub repository name")
    parser.add_argument("--no-shap", action="store_true",
                       help="Disable SHAP explanations")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Skip dashboard launch")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open browser")
    parser.add_argument("--dashboard-timeout", type=int, default=300,
                       help="Dashboard timeout in seconds (default: 300)")
    parser.add_argument("--port", type=int, default=8501,
                       help="Dashboard port (default: 8501)")
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = CompletePipeline()
    
    try:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            samples=args.samples,
            dagshub_owner=args.dagshub_owner,
            dagshub_repo=args.dagshub_repo,
            enable_shap=not args.no_shap,
            launch_dashboard=not args.no_dashboard,
            auto_open=not args.no_browser,
            dashboard_timeout=args.dashboard_timeout
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        pipeline.stop_dashboard()
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        pipeline.stop_dashboard()
        return 1

if __name__ == "__main__":
    exit(main())