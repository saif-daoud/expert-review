#!/usr/bin/env bash
set -euo pipefail

server_dir="${HOME}/clean-env/server"
conda_script="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
api_conda_environment="${CBT_LIVE_API_CONDA_ENV:-cbt-live-api}"

if [[ ! -d "$server_dir" || ! -f "${server_dir}/run.sh" ]]; then
  echo "Expected deployed API at ${server_dir}; refusing to continue." >&2
  exit 1
fi
if [[ ! -f "$conda_script" ]]; then
  echo "Conda initialization script not found at ${conda_script}." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$conda_script"
conda activate "$api_conda_environment"

mkdir -p "${server_dir}/logs"
if [[ -f "${server_dir}/logs/api.pid" ]]; then
  old_pid="$(cat "${server_dir}/logs/api.pid")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command="$(ps -p "$old_pid" -o args= || true)"
    if [[ "$old_command" != *"uvicorn"* || "$old_command" != *"app:app"* ]]; then
      echo "PID ${old_pid} is not the expected uvicorn process; refusing to stop it." >&2
      exit 1
    fi
    kill "$old_pid"
    for _ in $(seq 1 30); do
      if ! kill -0 "$old_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
fi

(
  cd "$server_dir"
  nohup bash run.sh > logs/api.log 2>&1 &
  echo "$!" > logs/api.pid
)

for _ in $(seq 1 45); do
  if curl --fail --silent "http://127.0.0.1:8000/api/health"; then
    echo
    echo "CBT live API recovered."
    exit 0
  fi
  sleep 1
done

echo "The API did not become healthy. Recent startup log:" >&2
tail -n 120 "${server_dir}/logs/api.log" >&2
exit 1
