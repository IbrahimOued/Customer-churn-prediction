# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY ./app /app
COPY ./models /models
COPY requirements.txt /app/requirements.txt


RUN pip install --upgrade pip \
 && pip install -r requirements.txt

EXPOSE 8501 8000

# Run backend app using FastAPI and frontend app using streamlit
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
