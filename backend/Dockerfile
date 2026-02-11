FROM python:3.11-slim

# Set locale to UTF-8 to prevent encoding issues
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies for PDF processing, OCR, compilation, FAISS, and locales
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    gcc \
    g++ \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install Python dependencies
# 1. Install CUDA-enabled PyTorch + FAISS-GPU FIRST so that sentence-transformers
#    picks up GPU torch instead of installing CPU torch.
# 2. Pin numpy<2 to avoid binary incompatibility with pandas/pyarrow.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir faiss-gpu-cu12 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/outputs /app/data/faiss && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default: run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
