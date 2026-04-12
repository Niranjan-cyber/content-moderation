# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy source
COPY server/   ./server/

COPY openenv.yaml .
COPY inference.py .

# HF Spaces runs as non-root on port 7860
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_PATH=server/data/dataset_final.csv \
    PYTHONPATH=/app
        
EXPOSE 7860

# Use --workers 1 to keep state consistent (single-process, sequential inference)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
