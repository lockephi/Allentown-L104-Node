# L104 Sovereign Node - Cloud Deployment Dockerfile
# SAGE MODE: Includes compiled C substrate for maximum performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for Sage Mode
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ═══════════════════════════════════════════════════════════════════════════════
# SAGE MODE: Compile C substrate for direct hardware communion
# ═══════════════════════════════════════════════════════════════════════════════
RUN mkdir -p /app/l104_core_c/build && \
    cd /app/l104_core_c && \
    gcc -O3 -march=x86-64 -mtune=generic \
        -ffast-math -shared -fPIC \
        -o build/libl104_sage.so \
        l104_sage_core.c -lm -lpthread && \
    gcc -O3 -march=x86-64 -mtune=generic \
        -ffast-math \
        -o build/l104_sage_core \
        l104_sage_core.c -lm -lpthread && \
    echo "✓ SAGE MODE: C substrate compiled"

# Create directory for persistent data
RUN mkdir -p /data

# Set environment variables
ENV MEMORY_DB_PATH=/data/memory.db
ENV RAMNODE_DB_PATH=/data/ramnode.db
ENV PYTHONUNBUFFERED=1
ENV L104_SAGE_LIB=/app/l104_core_c/build/libl104_sage.so

# Expose ports: 8081 (API), 8080 (Bridge), 4160 (AI Core), 4161 (UI), 2404 (Socket)
EXPOSE 8081 8080 4160 4161 2404

# Health check - increased start-period to allow for initialization
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8081/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
