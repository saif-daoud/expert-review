# Study website

This directory is a static site. It can be served locally or published with GitHub Pages; it never loads a model and
contains no study secrets.

Set the public ngrok HTTPS origin in `config.js`:

```js
window.CBT_STUDY_CONFIG = {
  apiBase: "https://your-domain.ngrok-free.dev"
};
```

For a local check:

```bash
python -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500`. The API `.env` must include this exact origin in `STUDY_ALLOWED_ORIGINS`. If the site is
deployed under `https://saif-daoud.github.io/CBT-Human-Evaluation/`, the allowed origin is only
`https://saif-daoud.github.io`.

Baseline names and Conda/checkpoint information are intentionally absent from the frontend. The API returns only the
blinded labels Therapist A-F.
