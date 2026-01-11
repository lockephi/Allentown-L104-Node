# L104 Sovereign Node - Cloud Deployment Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for persistent data
RUN mkdir -p /data

# Set environment variables
ENV MEMORY_DB_PATH=/data/memory.db
ENV RAMNODE_DB_PATH=/data/ramnode.db
ENV PYTHONUNBUFFERED=1

# Expose ports: 8081 (API), 4160 (AI Core), 2404 (Socket)
EXPOSE 8081 4160 2404

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8081/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
