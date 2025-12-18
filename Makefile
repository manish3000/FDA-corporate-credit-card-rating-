# Corporate Credit Rating Pipeline Makefile

.PHONY: help install setup clean pipeline app all test

help:
	@echo "Corporate Credit Rating Pipeline Commands"
	@echo "=========================================="
	@echo "make install       - Install all dependencies"
	@echo "make setup         - Create necessary directories"
	@echo "make pipeline      - Run the complete pipeline"
	@echo "make app           - Launch Streamlit application"
	@echo "make all           - Run pipeline and launch app"
	@echo "make clean         - Clean temporary files"
	@echo "make test          - Run tests"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

setup:
	@echo "Setting up project directories..."
	python -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data/raw', 'data/processed', 'models', 'logs']]"
	@echo "✓ Directories created"

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	@echo "✓ Cleaned"

pipeline:
	@echo "Running pipeline..."
	python pipeline.py

pipeline-step-%:
	@echo "Running pipeline step $*..."
	python pipeline.py --step=$*

notebooks:
	@echo "Executing all notebooks..."
	python run_pipeline.py --mode=pipeline

app:
	@echo "Launching Streamlit app..."
	streamlit run streamlit_app.py

app-scraper:
	@echo "Launching scraper app..."
	streamlit run app_with_scraping.py

all: pipeline app

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

format:
	@echo "Formatting code..."
	black *.py
	@echo "✓ Code formatted"

lint:
	@echo "Linting code..."
	flake8 *.py --max-line-length=100

requirements:
	@echo "Generating requirements.txt..."
	pip freeze > requirements.txt
	@echo "✓ requirements.txt updated"
