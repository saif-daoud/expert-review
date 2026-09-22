# GPU API server

Place this directory anywhere writable by your user, such as `~/clean-env/server`. It contains its own model-loading
code, policies, prompt, and PatientAct profile data; it does not import Python code from `simulations/` or `baselines/`.
The large trained checkpoint paths are configured in `.env`. FastAPI and all model workers run without root access.
Each configured expert receives 20 distinct profiles. Sessions run sequentially, and the API stores an 11-item CTRS
rating before unlocking the next therapist.

```bash
cd ~/clean-env/server
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cbt-live-api
cp .env.example .env
chmod 600 .env
```

Edit `.env`, including the Archer, ARIA, and Sweet-RL model-artifact paths, then run the preflight check:

```bash
set -a; source .env; set +a
python preflight.py
```

Start FastAPI first, verify its local health, then start ngrok:

```bash
mkdir -p logs
nohup bash run.sh > logs/api.log 2>&1 & echo $! > logs/api.pid
curl --fail http://127.0.0.1:8000/api/health
nohup bash run_ngrok.sh > logs/ngrok.log 2>&1 & echo $! > logs/ngrok.pid
```

The API must run with one Uvicorn worker because it owns the single FIFO generation lane. Worker placement defaults
to base + Archer on GPU 0 and ARIA + Sweet-RL on GPU 3. Override `STUDY_GPU_*` in `.env` if the allocation changes.
TOPAS is a static server-side stub until its implementation is ready.

Useful checks:

```bash
tail -f logs/api.log
tail -f logs/ngrok.log
tail -f logs/model-base.log
nvidia-smi
python inspect_db.py
python inspect_db.py --study STUDY_UUID --messages
```

Stop the tunnel and API with:

```bash
kill "$(cat logs/ngrok.pid)" 2>/dev/null || true
kill "$(cat logs/api.pid)" 2>/dev/null || true
```

The API shutdown handler terminates all model subprocesses.
