# Debt Collection ML System Makefile

.PHONY: help install setup-dagshub init-dvc run-pipeline clean

help:
	@echo "Available commands:"
	@echo "  install        - Install Python dependencies"
	@echo "  setup-dagshub  - Setup DagsHub integration"
	@echo "  init-dvc       - Initialize DVC pipeline"
	@echo "  run-pipeline   - Run complete DVC pipeline"
	@echo "  show-metrics   - Show pipeline metrics"
	@echo "  clean          - Clean generated files"

install:
	pip install -r requirements.txt

setup-dagshub:
	python scripts/setup_dagshub.py --repo-owner YOUR_USERNAME --create-config

init-dvc:
	python scripts/dvc_pipeline.py init

run-pipeline:
	python scripts/dvc_pipeline.py run

show-metrics:
	python scripts/dvc_pipeline.py metrics

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/processed/*.npy
	rm -rf models/trained/*.joblib