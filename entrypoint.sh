#!/bin/bash

# Start the FastAPI app
uvicorn backend.churn_predictor:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit app
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0