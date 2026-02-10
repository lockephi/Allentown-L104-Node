# ═══════════════════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Production Dockerfile (Feb 2026)
# Multi-layer optimized · Python 3.13 · Non-root · Hardened
# SAGE MODE: Includes compiled C substrate for maximum performance
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.13-slim AS base

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ─── System dependencies (rarely changes — cached well) ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc6-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ─── Python dependencies (only rebuilds when requirements.txt changes) ───
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── SAGE MODE: Compile C substrate for direct hardware communion ───
COPY l104_core_c/ /app/l104_core_c/
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

# ─── Application code (changes most frequently — last layer) ───
COPY . .

# ─── Persistent data + non-root user ───
RUN mkdir -p /data && \
    addgroup --system l104 && \
    adduser --system --ingroup l104 l104 && \
    chown -R l104:l104 /app /data

# ─── Environment ───
ENV MEMORY_DB_PATH=/data/memory.db \
    RAMNODE_DB_PATH=/data/ramnode.db \
    L104_SAGE_LIB=/app/l104_core_c/build/libl104_sage.so \
    PORT=8081

# Expose ports: 8081 (API), 8080 (Bridge), 4160 (AI Core), 4161 (UI), 2404 (Socket)
EXPOSE 8081 8080 4160 4161 2404

# ─── Healthcheck using curl (lighter than importing httpx) ───
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8081/health || exit 1

USER l104

# ─── Entrypoint ───
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8081} --log-level info --no-access-log"]
