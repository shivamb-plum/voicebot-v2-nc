# DeepFilterNet3 live demo

Record mic audio in the browser, stream it over a WebSocket through DeepFilterNet3 in 0.5 s chunks, then compare noisy vs filtered playback.

## Local

```sh
uv sync
uv run uvicorn server:app
```

Open http://127.0.0.1:8000. Smoke test: `uv run python test_server.py`.

## Vercel

Deploys as a container Function via `Dockerfile.vercel` (Vercel detects it automatically).
Python 3.11 is required because DeepFilterLib ships no wheels for 3.12+, which rules out Vercel's native Python runtime.
The image bakes in the model weights and CPU-only torch.

```sh
vercel deploy        # preview
vercel deploy --prod
```

Notes:
- WebSocket connections close when the Function hits its max duration (300 s default), so keep recordings under 5 min or raise `maxDuration` in `vercel.json` on Pro.
- Test the image locally: `docker build -f Dockerfile.vercel -t dfn . && docker run -p 8000:80 dfn`
