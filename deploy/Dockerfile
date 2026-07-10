# syntax=docker/dockerfile:1
# Bundled platform image for the Simpro Cloud deploy of obelix.
#
# Unlike the monorepo's per-service images (core-api/Dockerfile,
# core-storage-api/Dockerfile), this ONE image runs BOTH services in a single
# container so obelix is a single Backstage service/deployment:
#   - core-api           — the public API, on :3000 (the platform ingress port)
#   - core-storage-api   — its Postgres CRUD backend, INTERNAL on :8002
# core-api reaches storage-api at http://localhost:8002 (CORE_STORAGE_API_URL).
#
# Each service keeps its OWN frozen venv (its own pyproject + uv.lock) so their
# independently-pinned deps can never clash. deploy/entrypoint.sh starts
# storage-api, waits for it to be healthy, then execs core-api.
#
# SRC_PREFIX selects where the code lives in the build context:
#   - platform / shared app-repo (default): src/backend/  (deploy.Jenkinsfile
#     builds containers/Dockerfile with no --build-arg)
#   - monorepo build (docker build -f deploy/Dockerfile .): pass SRC_PREFIX=.

FROM python:3.12-slim AS builder
WORKDIR /app
ARG SRC_PREFIX=src/backend

# System deps for asyncpg / hiredis
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

# --- core-storage-api venv (frozen from its own lock) -----------------------
COPY ${SRC_PREFIX}/core-storage-api/pyproject.toml ${SRC_PREFIX}/core-storage-api/uv.lock ${SRC_PREFIX}/core-storage-api/README.md* ./core-storage-api/
RUN mkdir -p core-storage-api/src/core_storage_api \
    && touch core-storage-api/src/core_storage_api/__init__.py \
    && cd core-storage-api && uv sync --frozen --no-install-project --no-dev

# --- core-api venv (frozen; [pubsub] extra as in core-api's own image) ------
COPY ${SRC_PREFIX}/core-api/pyproject.toml ${SRC_PREFIX}/core-api/uv.lock ${SRC_PREFIX}/core-api/README.md* ./core-api/
RUN mkdir -p core-api/src/core_api \
    && touch core-api/src/core_api/__init__.py \
    && cd core-api && uv sync --frozen --no-install-project --no-dev --extra pubsub

# --- actual source ----------------------------------------------------------
COPY ${SRC_PREFIX}/common/ /app/common/
COPY ${SRC_PREFIX}/core-api/src/ /app/core-api/src/
COPY ${SRC_PREFIX}/core-storage-api/src/ /app/core-storage-api/src/
COPY ${SRC_PREFIX}/plugin/ /app/plugin/
COPY ${SRC_PREFIX}/static/ /app/static/
# alembic.ini's script_location points at core-storage-api's migrations; the
# migration job runs `alembic upgrade head` from /app inside this image.
COPY ${SRC_PREFIX}/alembic.ini /app/alembic.ini


# ── Production image ──
FROM python:3.12-slim AS runtime

# Re-declared: build ARGs do not cross FROM stage boundaries.
ARG SRC_PREFIX=src/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Both venvs, both source trees, common/, plugin/, static/, alembic.ini.
COPY --from=builder --chown=appuser:appuser /app /app
COPY --chown=appuser:appuser ${SRC_PREFIX}/deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
# core-api talks to the in-container storage service.
ENV CORE_STORAGE_API_URL=http://localhost:8002

USER appuser

# Platform ingress is fixed at :3000 (core-api). storage-api stays internal.
EXPOSE 3000

ENTRYPOINT ["/app/entrypoint.sh"]
