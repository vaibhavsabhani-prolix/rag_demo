# Patent search web app: React frontend (built here) served by the FastAPI API.

# --- Stage 1: build the React frontend -------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: FastAPI server -------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY metadata_fields.py .
COPY app ./app
COPY --from=frontend /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
