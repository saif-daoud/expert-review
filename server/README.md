# GPU API server

Place this directory anywhere writable by your user, such as `~/clean-env/server`. It contains its own model-loading
code, policies, prompt, and PatientAct profile data; it does not import Python code from `simulations/` or `baselines/`.
The large trained checkpoint paths are configured in `.env`. FastAPI and all model workers run without root access.
The first access code supports 30 email-linked registrations with one unique case drawn from underlying Patients 1–30 per email. The second access code is
restricted to one email with underlying Patients 31–40. Display numbering is account-relative and always begins at Patient 1. No participant code is entered on the website. First-time users complete a professional profile. Sessions run sequentially,
and the API stores an 11-item CTRS rating before unlocking the next therapist.

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

The API must run with one Uvicorn worker because it owns the single FIFO generation lane. For every new session worker,
the allocator checks live free memory with `nvidia-smi` and tries GPUs `0,1,2,3` in order. It skips GPUs below the
method-specific `STUDY_GPU_MIN_FREE_MB_*` threshold and retries the next GPU after a CUDA out-of-memory load failure.
If all four GPUs are unavailable, the website asks the expert to retry in five minutes. A loaded worker is also
unloaded after 60 seconds without another patient response (`STUDY_MODEL_IDLE_TIMEOUT_SECONDS=60`). TOPAS uses the
bundled standalone implementation and the artifact paths configured in `.env`.

Useful checks:

```bash
tail -f logs/api.log
tail -f logs/ngrok.log
tail -f logs/model-base.log
nvidia-smi
python inspect_db.py
python inspect_db.py --study STUDY_UUID --messages
```

To reset test activity while retaining a specific partially completed study,
stop the API and run the reset first without `--apply`. The command refuses to
continue unless the preserved study has exactly the expected number of ratings
and creates a timestamped SQLite backup before changing anything:

```bash
python reset_db.py --keep-study STUDY_UUID --expected-ratings 3
python reset_db.py --keep-study STUDY_UUID --expected-ratings 3 --apply
```

Completed sessions in the preserved study remain unchanged. Any unfinished
panel in that study is cleared so it can be started cleanly; all other test
studies and participant registrations are removed.

Stop the tunnel and API with:

```bash
kill "$(cat logs/ngrok.pid)" 2>/dev/null || true
kill "$(cat logs/api.pid)" 2>/dev/null || true
```

The API shutdown handler terminates all model subprocesses.
