# =============================================================================
# Stage 1: deps
# Install all Python dependencies via uv into /app/.venv
# Only invalidates when pyproject.toml or uv.lock changes
# =============================================================================
FROM python:3.12-slim AS deps

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy only dependency files — not app code or weights
COPY pyproject.toml uv.lock ./

# Sync dependencies into a virtual environment at /app/.venv
# --frozen: use lockfile exactly, no updates
# --no-dev: skip dev dependencies
# --no-install-project: don't install the project itself, just deps
RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Stage 2: final
# Assemble the production image with local HF cache
# =============================================================================
FROM python:3.12-slim AS final

WORKDIR /app

# Copy venv from Stage 1
COPY --from=deps /app/.venv /app/.venv

# Copy pre-cached HuggingFace models from local hf-cache directory
COPY hf-cache /hf-cache

# Copy local model weights (~1.5 GB — this layer changes rarely)
COPY weights/ ./weights/

# Copy application code (changes frequently — keep last for cache efficiency)
COPY app.py ./

# Run as non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app /hf-cache
USER appuser

# Environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/hf-cache
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

EXPOSE 8501

ENTRYPOINT ["sh", "-c", "python -m streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
