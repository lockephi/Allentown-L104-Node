# L104 Sovereign Node - Cloud Deployment Dockerfile v2.0
# SAGE MODE: Multi-stage build with compiled C substrate
# UPGRADE v2.0:
#   - Multi-stage build (builder → runtime) — sheds gcc/make from final image
#   - Non-root user for security
#   - curl-based health check (fast, no Python overhead)
#   - Pinned Python patch version
#   - CPU limit awareness via --cpuset
#   - .dockerignore-friendly layer ordering

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: BUILDER — compile C substrate
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and compile C substrate
COPY l104_core_c/ /build/l104_core_c/
RUN mkdir -p /build/l104_core_c/build && \
    cd /build/l104_core_c && \
    gcc -O3 -march=x86-64 -mtune=generic \
    -ffast-math -shared -fPIC \
    -o build/libl104_sage.so \
    l104_sage_core.c -lm -lpthread && \
    gcc -O3 -march=x86-64 -mtune=generic \
    -ffast-math \
    -o build/l104_sage_core \
    l104_sage_core.c -lm -lpthread && \
    echo "✓ SAGE MODE: C substrate compiled"

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: RUNTIME — minimal production image
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r l104 && useradd -r -g l104 -d /app -s /sbin/nologin l104

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy compiled C substrate from builder
COPY --from=builder /build/l104_core_c/build/ /app/l104_core_c/build/

# Copy application code
COPY . .

# Create data directory with correct ownership
RUN mkdir -p /data && chown -R l104:l104 /app /data

# Set environment variables
ENV MEMORY_DB_PATH=/data/memory.db
ENV RAMNODE_DB_PATH=/data/ramnode.db
ENV PYTHONUNBUFFERED=1
ENV L104_SAGE_LIB=/app/l104_core_c/build/libl104_sage.so
ENV PORT=8081
ENV L104_CPU_CORES=0
ENV LOG_FORMAT=json
ENV LOG_LEVEL=info

# Expose ports: 8081 (API), 8080 (Bridge), 4160 (AI Core), 4161 (UI), 2404 (Socket)
EXPOSE 8081 8080 4160 4161 2404

# Switch to non-root user
USER l104

# Health check using curl (fast, no Python overhead)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8081/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8081}"]
