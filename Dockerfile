# =============================================================================
# Dockerfile: AI-Driven RF Spectrum Monitoring & Interference Detection System
# Multi-Target Deployment: Streamlit HMI Console or FastAPI REST Microservice
# Author: Akshay Kumar (CSE Collaborator)
# =============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project codebase
COPY . .

# Expose ports: 8501 (Streamlit HMI) & 8000 (FastAPI REST Service)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default launch command: Streamlit Dashboard
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
