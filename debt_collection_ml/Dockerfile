FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/trained logs reports monitoring/metrics

# Set Python path
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000 8501 8888

# Default command
CMD ["python", "scripts/train_model_pipeline.py"]