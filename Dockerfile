# ==============================================================================
# SmartCrop AI — Multi-Modal Agricultural Intelligence Platform Dockerfile
# Production-ready multi-process container (FastAPI + Streamlit)
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    BACKEND_PORT=8000 \
    YOLO_AUTOINSTALL=False

WORKDIR /app

# Install system dependencies required for OpenCV, GDAL/Rasterio, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for efficient image caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and configurations
COPY app/ ./app/
COPY models/ ./models/
COPY .streamlit/ ./.streamlit/
COPY entrypoint.py .
COPY README.md .

# Expose both Frontend (8501) and Backend (8000)
EXPOSE 8501 8000

# Healthcheck on FastAPI backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/api/health || exit 1

# Launch the unified supervisor
ENTRYPOINT ["python", "entrypoint.py"]
