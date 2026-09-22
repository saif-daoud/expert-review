# CBT live interaction study

This package contains the real six-therapist study website and its rootless GPU API.

- Each expert receives 20 distinct PatientAct role cards (10 anxiety and 10 depression profiles).
- Each patient starts on a full profile page, followed by six sequential blinded sessions (`Therapist A` through `Therapist F`).
- After every session, the expert must submit all 11 CTRS scores before the next therapist unlocks.
- Prompting, ProAct, Archer, ARIA, and Sweet-RL use bundled inference code matching the simulation policies.
- TOPAS returns a server-side static response until its policy is ready.
- A single FIFO queue serializes inference across all experts and panels.
- Separate persistent workers keep models loaded in their required Conda environments.
- The method-to-panel mapping is randomized deterministically and never returned to the browser.
- All messages, CTRS ratings, jobs, mappings, and completion state are stored in SQLite.

The GitHub Pages site is deployed to `https://saif-daoud.github.io/expert-review/`. Its workflow publishes only
`frontend/` and obtains the public API origin from the `API_BASE_URL` GitHub repository secret. The study access code
and token-signing secret belong only in the GPU server's `.env` and must never be added to GitHub.

## Runtime layout

```text
Browser / GitHub Pages
        |
        | HTTPS (ngrok)
        v
FastAPI + SQLite + one FIFO generation lane (cbt-live-api)
        |
        +-- base env       / GPU 0: Prompting + ProAct, one shared Qwen
        +-- archer_env     / GPU 0: Archer
        +-- aria_env       / GPU 3: ARIA
        +-- sweet_rl       / GPU 3: Sweet-RL
        +-- API process             TOPAS static stub
```

Workers are loaded lazily on first use and then retained. Only one worker generates at a time, even if two experts
submit messages together. The default GPU mapping uses GPUs 0 and 3 because those were free in the supplied
`nvidia-smi` snapshot; change the four `STUDY_GPU_*` values if allocations change.

## Standalone server package

The uploaded `server/` directory contains all Python code, prompts, and study-profile data needed by the API. It does
not import anything from `simulations/` or `baselines/`. The trained Archer, ARIA, and Sweet-RL weights remain external
model artifacts because they total roughly 49 GB; their locations are configured in `.env`.

```text
~/clean-env/
  server/
    app.py
    model_runtime/                # bundled baseline inference code
    assets/                       # bundled therapist prompt
    data/patient_act.json         # bundled assigned profiles
  model-artifacts/                # optional location; may be anywhere readable
```

Set `STUDY_ARCHER_CHECKPOINT`, `STUDY_ARIA_CHECKPOINT`, and `STUDY_SWEET_RL_MODEL` to the existing weight locations.

## Deploy the API without root access

Create a clean archive locally so the remote `.env`, database, and logs are not overwritten:

```bash
tar -czf cbt-live-server.tar.gz --exclude=server/.env --exclude='server/data/*.sqlite3*' --exclude=server/logs --exclude=server/__pycache__ -C interface/cbt-live-interaction server
scp cbt-live-server.tar.gz YOUR_USER@YOUR_SERVER:~/clean-env/
```

Then extract and configure it on the GPU server:

```bash
cd ~/clean-env
tar -xzf cbt-live-server.tar.gz
cd ~/clean-env/server
source ~/miniconda3/etc/profile.d/conda.sh

conda env create -f environment.yml   # only if cbt-live-api does not exist
conda activate cbt-live-api

cp .env.example .env
chmod 600 .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Edit `.env` and replace `STUDY_ACCESS_CODE` and `STUDY_TOKEN_SECRET`; use the generated random string for the token
secret. Confirm the three model-artifact paths, Conda environment names, GPU indices, public website origin, and
optional ngrok static domain. Then validate paths and environments without loading a model:

```bash
set -a
source .env
set +a
python preflight.py
```

Start the API and tunnel:

```bash
mkdir -p logs
nohup bash run.sh > logs/api.log 2>&1 & echo $! > logs/api.pid
curl --fail http://127.0.0.1:8000/api/health

nohup bash run_ngrok.sh > logs/ngrok.log 2>&1 & echo $! > logs/ngrok.pid
grep -o 'https://[^ ]*ngrok-free[^ ]*' logs/ngrok.log | tail -1
```

Paste the resulting HTTPS origin into `frontend/config.js`. The VPN is not needed by the browser once ngrok is
running. A free changing ngrok URL must be pasted again after each restart; a static ngrok domain avoids that.

To stop everything started above:

```bash
kill "$(cat logs/ngrok.pid)" 2>/dev/null || true
kill "$(cat logs/api.pid)" 2>/dev/null || true
```

Stopping FastAPI also terminates its child model workers and releases their GPU memory.

## Run the website locally

The website intentionally has no localhost API fallback. Configure `frontend/config.js`, then:

```bash
cd interface/cbt-live-interaction/frontend
python -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500`. Add that exact origin to `STUDY_ALLOWED_ORIGINS`. For GitHub Pages, use
`https://saif-daoud.github.io` as the origin (not the repository path).

## Safe smoke test before GPU inference

Set `STUDY_INFERENCE_MODE=static`, restart the API, and test all six sessions and their CTRS forms. This exercises authentication, profiles,
SQLite, ratings, queuing, the tunnel, and the website without loading a checkpoint. Set it back to `real` for the actual five
baselines; TOPAS remains static in both modes.

The first request to each runtime may take several minutes while its model loads. Later requests reuse that worker.
Model logs are written to `server/logs/model-<runtime>.log`; API and tunnel logs use `api.log` and `ngrok.log`.

## Inspect collected data

```bash
cd ~/clean-env/server
set -a; source .env; set +a
python inspect_db.py
python inspect_db.py --participant EXPERT-5834
python inspect_db.py --study STUDY_UUID --messages
python inspect_db.py --study STUDY_UUID --show-methods
```

Only use `--show-methods` for administrative checks because it reveals the blinded assignment.

## Automated tests

From `interface/cbt-live-interaction/` on the local machine:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite uses static inference and does not require CUDA.

## Before inviting experts

- Validate each real method from the website, including its first-turn response and one follow-up.
- Keep `uvicorn` at one worker; the in-process FIFO manager must have one owner.
- Do not expose the FastAPI port directly; keep it bound to `127.0.0.1` behind HTTPS ngrok.
- Add the approved consent, withdrawal, retention, and researcher-contact wording.
- Rotate and remove any plaintext external API credential present in experiment shell scripts before copying code.
- Do not treat TOPAS-panel data as a real baseline result until its worker replaces the stub.
