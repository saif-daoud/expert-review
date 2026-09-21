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

ngrok_bin="${NGROK_BIN:-$HOME/bin/ngrok}"
api_port="${STUDY_PORT:-8000}"
ngrok_domain="${NGROK_DOMAIN:-}"

if [[ ! -x "$ngrok_bin" ]]; then
  echo "ngrok is not executable at $ngrok_bin" >&2
  exit 1
fi

if ! curl --fail --silent --show-error "http://127.0.0.1:${api_port}/api/health" >/dev/null; then
  echo "The local API is not healthy on 127.0.0.1:${api_port}. Start it before ngrok." >&2
  exit 1
fi

arguments=(http "http://127.0.0.1:${api_port}" --log=stdout)
if [[ -n "$ngrok_domain" ]]; then
  arguments+=(--url "$ngrok_domain")
fi

exec "$ngrok_bin" "${arguments[@]}"
