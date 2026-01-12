FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default port (Railway overrides with PORT env var)
ENV PORT=8000

# Run FastAPI with uvicorn - use shell form so $PORT expands
CMD uvicorn api:app --host 0.0.0.0 --port $PORT
