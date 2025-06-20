make run_pipeline:
	@echo "Starting pipeline..."
	@python start_pipeline.py

make build_application:
	@echo "Building application..."
	@docker build ./ -t strealit_churn_predictor:latest

make run_application:
	@echo "Running application..."
	@docker run -p 8501:8501 strealit_churn_predictor

make build_and_run:
	@echo "Building and running application..."
	@make build_application
	@make run_application