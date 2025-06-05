# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

EXPOSE 8000

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8000", "--server.address", "

