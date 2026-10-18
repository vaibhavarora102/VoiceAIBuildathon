# Use official lightweight Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables for clean logging and dynamic port binding
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install minimal system tools for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose default port
EXPOSE 8000

# Run Uvicorn listening on 0.0.0.0 with dynamic $PORT support
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
