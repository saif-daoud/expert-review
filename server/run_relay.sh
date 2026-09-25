#!/usr/bin/env bash
set -euo pipefail

server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$server_dir"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

exec python -m uvicorn relay_app:app \
  --host "${SIMULATOR_RELAY_HOST:-127.0.0.1}" \
  --port "${SIMULATOR_RELAY_PORT:-8001}"
