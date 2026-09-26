# ==============================================================================
# SkyGuard AI - Production Multi-Stage Dockerfile
# ==============================================================================

# --- Stage 1: Build the React + Vite Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Production Runtime ---
FROM python:3.11-slim
WORKDIR /app

# Install minimal OS dependencies for building Python C-extensions (NumPy, Scikit-Learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend, data, and models
COPY backend/ ./backend/
COPY data/ ./data/
COPY weather_anomaly_detection/ ./weather_anomaly_detection/

# Copy compiled frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Configure runtime environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DATA_DIR=/app/data \
    CORS_ORIGINS=*

EXPOSE 8000

# Start FastAPI application with Uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
