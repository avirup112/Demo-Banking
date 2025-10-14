#!/usr/bin/env python3
"""
DVC Pipeline Management Script
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def initialize_dvc():
    """Initialize DVC in the project"""
    
    print("Initializing DVC...")
    
    # Initialize DVC
    if not Path('.dvc').exists():
        if not run_command("dvc init", "Initialize DVC"):
            return False
    
    # Add data to DVC tracking
    data_files = [
        'data/raw/debt_collection_data.csv',
        'data/processed/',
        'models/trained/',
        'models/artifacts/'
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            run_command(f"dvc add {data_file}", f"Add {data_file} to DVC")
    
    return True

def run_pipeline(stages=None, force=False):
    """Run DVC pipeline"""
    
    print("Running DVC pipeline...")
    
    # Build command
    cmd = "dvc repro"
    
    if stages:
        cmd += f" {' '.join(stages)}"
    
    if force:
        cmd += " --force"
    
    return run_command(cmd, "Run DVC pipeline")

def show_pipeline_status():
    """Show pipeline status"""
    
    print("Pipeline Status:")
    run_command("dvc status", "Show pipeline status")
    
    print("\nPipeline DAG:")
    run_command("dvc dag", "Show pipeline DAG")

def show_metrics():
    """Show pipeline metrics"""
    
    print("Pipeline Metrics:")
    run_command("dvc metrics show", "Show metrics")
    
    print("\nMetrics Diff:")
    run_command("dvc metrics diff", "Show metrics diff")

def show_plots():
    """Show pipeline plots"""
    
    print("Pipeline Plots:")
    run_command("dvc plots show", "Show plots")

def setup_remote(remote_url=None):
    """Setup DVC remote storage"""
    
    if not remote_url:
        # Use DagsHub as default remote
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        dagshub_params = params.get('dagshub', {})
        if dagshub_params.get('repo_owner'):
            remote_url = f"https://dagshub.com/{dagshub_params['repo_owner']}/{dagshub_params['repo_name']}.dvc"
        else:
            print("No DagsHub configuration found. Please set up remote manually.")
            return False
    
    print(f"Setting up DVC remote: {remote_url}")
    
    # Add remote
    run_command(f"dvc remote add -d origin {remote_url}", "Add DVC remote")
    
    # Configure remote
    run_command("dvc remote modify origin --local auth basic", "Configure remote auth")
    
    return True

def push_data():
    """Push data to remote storage"""
    
    print("Pushing data to remote storage...")
    return run_command("dvc push", "Push data to remote")

def pull_data():
    """Pull data from remote storage"""
    
    print("Pulling data from remote storage...")
    return run_command("dvc pull", "Pull data from remote")

def create_experiment(name, params_override=None):
    """Create a new experiment"""
    
    print(f"Creating experiment: {name}")
    
    cmd = f"dvc exp run --name {name}"
    
    if params_override:
        for param, value in params_override.items():
            cmd += f" --set-param {param}={value}"
    
    return run_command(cmd, f"Run experiment: {name}")

def list_experiments():
    """List all experiments"""
    
    print("Listing experiments:")
    run_command("dvc exp show", "Show experiments")

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description='DVC Pipeline Management')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize DVC')
    
    # Run pipeline command
    run_parser = subparsers.add_parser('run', help='Run DVC pipeline')
    run_parser.add_argument('--stages', nargs='+', help='Specific stages to run')
    run_parser.add_argument('--force', action='store_true', help='Force run all stages')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show metrics')
    
    # Plots command
    plots_parser = subparsers.add_parser('plots', help='Show plots')
    
    # Remote setup command
    remote_parser = subparsers.add_parser('remote', help='Setup remote storage')
    remote_parser.add_argument('--url', help='Remote URL')
    
    # Push command
    push_parser = subparsers.add_parser('push', help='Push data to remote')
    
    # Pull command
    pull_parser = subparsers.add_parser('pull', help='Pull data from remote')
    
    # Experiment commands
    exp_parser = subparsers.add_parser('experiment', help='Experiment management')
    exp_subparsers = exp_parser.add_subparsers(dest='exp_command')
    
    exp_run_parser = exp_subparsers.add_parser('run', help='Run experiment')
    exp_run_parser.add_argument('--name', required=True, help='Experiment name')
    exp_run_parser.add_argument('--param', action='append', help='Override parameters (key=value)')
    
    exp_list_parser = exp_subparsers.add_parser('list', help='List experiments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Execute commands
    if args.command == 'init':
        initialize_dvc()
    
    elif args.command == 'run':
        run_pipeline(args.stages, args.force)
    
    elif args.command == 'status':
        show_pipeline_status()
    
    elif args.command == 'metrics':
        show_metrics()
    
    elif args.command == 'plots':
        show_plots()
    
    elif args.command == 'remote':
        setup_remote(args.url)
    
    elif args.command == 'push':
        push_data()
    
    elif args.command == 'pull':
        pull_data()
    
    elif args.command == 'experiment':
        if args.exp_command == 'run':
            params_override = {}
            if args.param:
                for param in args.param:
                    key, value = param.split('=', 1)
                    params_override[key] = value
            
            create_experiment(args.name, params_override)
        
        elif args.exp_command == 'list':
            list_experiments()
    
    print("\nDVC pipeline management completed!")

if __name__ == "__main__":
    main()