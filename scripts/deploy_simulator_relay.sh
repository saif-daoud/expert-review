#!/usr/bin/env bash
set -euo pipefail

deploy_root="${HOME}/clean-env"
server_dir="${deploy_root}/server"
incoming_app="${deploy_root}/relay-app.py"
incoming_relay="${deploy_root}/relay-llm_relay.py"
incoming_example="${deploy_root}/relay-env-example"
incoming_secrets="${deploy_root}/relay-secrets.env"
backup_dir="$(mktemp -d "${deploy_root}/relay-backup.XXXXXX")"
restart_required=false

conda_script="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
api_conda_environment="${CBT_LIVE_API_CONDA_ENV:-cbt-live-api}"
if [[ ! -f "$conda_script" ]]; then
  echo "Conda initialization script not found at ${conda_script}." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$conda_script"
conda activate "$api_conda_environment"

cleanup_incoming() {
  rm -f "$incoming_app" "$incoming_relay" "$incoming_example" "$incoming_secrets"
}

restart_api() {
  mkdir -p "${server_dir}/logs"
  (
    cd "$server_dir"
    nohup bash run.sh > logs/api.log 2>&1 &
    echo "$!" > logs/api.pid
  )
  for _ in $(seq 1 30); do
    if curl --fail --silent "http://127.0.0.1:8000/api/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

rollback() {
  if [[ "$restart_required" != true ]]; then
    return
  fi
  echo "Deployment failed; restoring the previous API files." >&2
  cp "${backup_dir}/app.py" "${server_dir}/app.py"
  cp "${backup_dir}/.env" "${server_dir}/.env"
  if [[ -f "${backup_dir}/llm_relay.py" ]]; then
    cp "${backup_dir}/llm_relay.py" "${server_dir}/llm_relay.py"
  else
    rm -f "${server_dir}/llm_relay.py"
  fi
  restart_api || true
}

trap 'rollback; cleanup_incoming; rm -rf "$backup_dir"' ERR

if [[ ! -d "$server_dir" || ! -f "${server_dir}/run.sh" || ! -f "${server_dir}/.env" ]]; then
  echo "Expected deployed API at ${server_dir}; refusing to continue." >&2
  exit 1
fi
for file in "$incoming_app" "$incoming_relay" "$incoming_example" "$incoming_secrets"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing deployment input: ${file}" >&2
    exit 1
  fi
done

health="$(curl --fail --silent "http://127.0.0.1:8000/api/health")"
if [[ "$health" != *'"busy":false'* || "$health" != *'"queue_depth":0'* ]]; then
  echo "The live API is busy; relay deployment was not attempted." >&2
  exit 1
fi

api_pid="$(cat "${server_dir}/logs/api.pid")"
if [[ ! "$api_pid" =~ ^[0-9]+$ ]]; then
  echo "The API PID file is invalid; refusing to stop any process." >&2
  exit 1
fi
api_command="$(ps -p "$api_pid" -o args= || true)"
if [[ "$api_command" != *"uvicorn"* || "$api_command" != *"app:app"* ]]; then
  echo "PID ${api_pid} is not the expected uvicorn app process; refusing to stop it." >&2
  exit 1
fi

cp "${server_dir}/app.py" "${backup_dir}/app.py"
cp "${server_dir}/.env" "${backup_dir}/.env"
if [[ -f "${server_dir}/llm_relay.py" ]]; then
  cp "${server_dir}/llm_relay.py" "${backup_dir}/llm_relay.py"
fi
restart_required=true

set -a
# shellcheck disable=SC1090
source "$incoming_secrets"
set +a
for name in SIMULATOR_RELAY_TOKEN SIMULATOR_RELAY_UPSTREAM_API_KEY SIMULATOR_RELAY_UPSTREAM_BASE_URL SIMULATOR_RELAY_MODEL; do
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is missing from the private deployment input." >&2
    exit 1
  fi
done

update_env() {
  local key="$1"
  local value="$2"
  local temporary="${server_dir}/.env.relay.tmp"
  awk -v key="$key" -v value="$value" '
    BEGIN { replaced = 0 }
    index($0, key "=") == 1 { if (!replaced) print key "=" value; replaced = 1; next }
    { print }
    END { if (!replaced) print key "=" value }
  ' "${server_dir}/.env" > "$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "${server_dir}/.env"
}

update_env SIMULATOR_RELAY_TOKEN "$SIMULATOR_RELAY_TOKEN"
update_env SIMULATOR_RELAY_UPSTREAM_API_KEY "$SIMULATOR_RELAY_UPSTREAM_API_KEY"
update_env SIMULATOR_RELAY_UPSTREAM_BASE_URL "$SIMULATOR_RELAY_UPSTREAM_BASE_URL"
update_env SIMULATOR_RELAY_MODEL "$SIMULATOR_RELAY_MODEL"
update_env SIMULATOR_RELAY_MAX_REQUEST_BYTES "2097152"
update_env SIMULATOR_RELAY_MAX_OUTPUT_TOKENS "4000"
update_env SIMULATOR_RELAY_TIMEOUT_SECONDS "180"

cp "$incoming_app" "${server_dir}/app.py"
cp "$incoming_relay" "${server_dir}/llm_relay.py"
cp "$incoming_example" "${server_dir}/.env.example"

kill "$api_pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$api_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$api_pid" 2>/dev/null; then
  echo "The old API did not stop cleanly." >&2
  exit 1
fi

restart_api
health="$(curl --fail --silent "http://127.0.0.1:8000/api/health")"
if [[ "$health" != *'"configured":true'* || "$health" != *'"model":"gpt-4.1"'* ]]; then
  echo "The restarted API did not report a configured GPT-4.1 relay." >&2
  exit 1
fi

restart_required=false
cleanup_incoming
rm -rf "$backup_dir"
trap - ERR
echo "Simulator relay deployed; the existing ngrok tunnel remains active."
