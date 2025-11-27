.PHONY: help install dev test test-cov lint format clean docker-build docker-up docker-down docker-logs

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Run development server
	uvicorn server:app --reload --port 8000

test: ## Run tests
	pytest test_server.py -v

test-cov: ## Run tests with coverage
	pytest test_server.py -v --cov=server --cov-report=html --cov-report=term

lint: ## Run linting
	ruff check server.py test_server.py
	mypy server.py --ignore-missing-imports

format: ## Format code
	black server.py test_server.py
	ruff check --fix server.py test_server.py

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf ipld_store ipld_index.json
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-restart: ## Restart Docker containers
	docker-compose restart

docker-clean: ## Clean Docker resources
	docker-compose down -v
	docker system prune -f

setup-schemas: ## Create example schemas directory
	mkdir -p schemas
	@echo 'Example schema directory created. Add your JSON schemas to ./schemas/'

init: install setup-schemas ## Initialize project
	@echo 'Project initialized! Run "make dev" to start the server.'