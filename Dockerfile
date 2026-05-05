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
# Stage 2: hf-cache
# Pre-download HuggingFace model files into /hf-cache
# Only invalidates when Stage 1 changes (i.e., deps change)
# =============================================================================
FROM python:3.12-slim AS hf-cache

WORKDIR /app

# Bring in uv and the venv from Stage 1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=deps /app/.venv /app/.venv

# Tell HuggingFace where to cache models
ENV HF_HOME=/hf-cache
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy and run the preload script
COPY scripts/preload_hf_models.py ./scripts/preload_hf_models.py
RUN python scripts/preload_hf_models.py

# =============================================================================
# Stage 3: final
# Assemble the production image
# =============================================================================
FROM python:3.12-slim AS final

WORKDIR /app

# Bring in uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy venv from Stage 1
COPY --from=deps /app/.venv /app/.venv

# Copy HuggingFace model cache from Stage 2
COPY --from=hf-cache /hf-cache /hf-cache

# Copy local model weights (~1.5 GB — this layer changes rarely)
COPY weights/ ./weights/

# Copy application code (changes frequently — keep last for cache efficiency)
COPY app.py ./

# Environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/hf-cache
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
