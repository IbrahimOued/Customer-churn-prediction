# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY ./app /app
COPY ./models /models


RUN pip install --upgrade pip \
 && pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "churn_predictor:app", "--host", "0.0.0.0", "--port", "8000"]
