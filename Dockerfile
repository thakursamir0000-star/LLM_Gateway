FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000 \
    GATEWAY_DB=/app/var/gateway.db \
    HF_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# The trained (classifier) router embeds requests with sentence-transformers.
# It is optional: without it the gateway uses the rule-based router. CPU-only
# torch keeps the image smaller, and the embedder is downloaded at build time.
ARG WITH_EMBEDDINGS=false
ARG EMBEDDER=sentence-transformers/all-MiniLM-L6-v2
RUN if [ "$WITH_EMBEDDINGS" = "true" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu "sentence-transformers>=3.0" && \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDER')" && \
        chmod -R a+rX /opt/hf-cache; \
    fi

COPY gateway ./gateway
COPY dashboard ./dashboard
COPY config ./config
COPY models ./models

RUN useradd --create-home --uid 1000 app && mkdir -p /app/var && chown app /app/var
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "gateway"]
