# L104 Sovereign Node - Cloud Deployment Dockerfile
# SAGE MODE: Includes compiled C substrate for maximum performance
# OPTIMIZED: Layer caching to minimize rebuilds
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for Sage Mode
# This layer rarely changes - cached well
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements first (for layer caching)
# Only rebuilds pip layer if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy C source separately (rarely changes)
COPY l104_core_c/ /app/l104_core_c/

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

# Copy application code LAST (changes most frequently)
# Only this layer rebuilds on code changes
# .dockerignore excludes large files like fine_tune_exports, notebooks, etc.
COPY . .

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
    CMD python -c "import httpx; import os; r=httpx.get(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# Run the application - use PORT env variable for Cloud Run compatibility
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
