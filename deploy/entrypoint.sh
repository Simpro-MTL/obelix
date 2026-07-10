#!/bin/sh
# Bundled obelix entrypoint — runs core-storage-api (internal :8002) and
# core-api (:3000) in one container. storage-api starts first; once it is
# healthy, core-api execs into the foreground as the container's main process.
set -eu

STORAGE_UVICORN=/app/core-storage-api/.venv/bin/uvicorn
API_UVICORN=/app/core-api/.venv/bin/uvicorn
PORT="${PORT:-3000}"

# If a command was passed (e.g. the platform migration job runs
# `docker run <image> sh -c '…alembic upgrade head'`), exec it directly instead
# of booting the bundled services. The K8s Deployment passes no command, so it
# still starts the full bundle below.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# Assemble DATABASE_URL for core-storage-api from the platform's DB env.
# Simpro Cloud injects the pieces — DATABASE_WRITER_HOST + DATABASE_PORT (ConfigMap)
# and DB_USERNAME/DB_PASSWORD/DB_NAME (ExternalSecret) — but not the single DSN
# core-storage-api reads. Without this it falls back to its localhost default and
# dies with ECONNREFUSED. Fires only when DATABASE_URL isn't already set (local
# docker-compose sets it), so it's platform-only.
if [ -z "${DATABASE_URL:-}" ] && [ -n "${DATABASE_WRITER_HOST:-}" ]; then
    _db_user="${DB_USERNAME:-${DB_USER:-}}"
    _db_pass="${DB_PASSWORD:-${DB_PASS:-}}"
    _db_name="${DB_NAME:-}"
    _db_port="${DATABASE_PORT:-5432}"
    _db_ssl=""
    if [ "${POSTGRES_REQUIRE_SSL:-false}" = "true" ]; then _db_ssl="?ssl=require"; fi
    export DATABASE_URL="postgresql+asyncpg://${_db_user}:${_db_pass}@${DATABASE_WRITER_HOST}:${_db_port}/${_db_name}${_db_ssl}"
    # Reader pool: use the RO endpoint when provided; otherwise core-storage-api
    # falls back to the writer on its own.
    if [ -n "${DATABASE_READER_HOST:-}" ]; then
        export READ_DATABASE_URL="postgresql+asyncpg://${_db_user}:${_db_pass}@${DATABASE_READER_HOST}:${_db_port}/${_db_name}${_db_ssl}"
    fi
    echo "[entrypoint] assembled DATABASE_URL (host=${DATABASE_WRITER_HOST} db=${_db_name} ssl=${POSTGRES_REQUIRE_SSL:-false})"
fi

# 1. core-storage-api in the background (internal — never exposed to the ALB).
echo "[entrypoint] starting core-storage-api on :8002"
PYTHONPATH=/app/core-storage-api/src:/app \
  "$STORAGE_UVICORN" core_storage_api.app:app \
  --host 0.0.0.0 --port 8002 --workers 1 --timeout-keep-alive 65 &
STORAGE_PID=$!

# Forward termination to the background service so the pod stops cleanly.
trap 'kill -TERM "$STORAGE_PID" 2>/dev/null || true' TERM INT

# 2. Wait for storage-api to answer its health check before starting core-api
#    (core-api's startup assumes the storage backend is reachable).
echo "[entrypoint] waiting for core-storage-api /healthz ..."
i=0
while [ "$i" -lt 60 ]; do
  if curl -fs http://localhost:8002/healthz >/dev/null 2>&1; then
    echo "[entrypoint] core-storage-api is healthy"
    break
  fi
  if ! kill -0 "$STORAGE_PID" 2>/dev/null; then
    echo "[entrypoint] FATAL: core-storage-api exited during startup" >&2
    exit 1
  fi
  i=$((i + 1))
  sleep 2
done
if [ "$i" -ge 60 ]; then
  echo "[entrypoint] FATAL: core-storage-api did not become healthy in time" >&2
  kill -TERM "$STORAGE_PID" 2>/dev/null || true
  exit 1
fi

# 3. core-api in the foreground (becomes the container's main process via exec).
echo "[entrypoint] starting core-api on :$PORT"
exec env \
  PYTHONPATH=/app/core-api/src:/app \
  CORE_STORAGE_API_URL=http://localhost:8002 \
  "$API_UVICORN" core_api.app:app \
  --host 0.0.0.0 --port "$PORT" --workers 2 --timeout-keep-alive 65
