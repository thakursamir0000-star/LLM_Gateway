# Base image pinned by digest (python:3.12-slim on 2026-11-22); Dependabot proposes digest updates.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000 \
    GATEWAY_DB=/app/var/gateway.db \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Exact versions with hashes from requirements.lock (generated from requirements.txt; see SECURITY.md).
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock

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
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini

RUN useradd --create-home --uid 1000 app && mkdir -p /app/var && chown app /app/var
USER app

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "gateway"]
