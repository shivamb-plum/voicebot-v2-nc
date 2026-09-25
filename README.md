# Noise-suppression shootout

Record mic audio in the browser, stream it over a WebSocket in 0.5 s chunks through every model at once, watch per-model speed live, then listen to each result.

Models (`models.py`, one streaming interface):

| Model | Runs at | Source |
|---|---|---|
| DeepFilterNet3 | 48 kHz | `deepfilternet` pip package |
| DPDFNet2 | 8 / 16 / 48 kHz variant matching the mic rate | `dpdfnet` pip package (ONNX) |
| DTLN | 16 kHz | `models/dtln/model_{1,2}.onnx` from breizhn/DTLN |
| GTCRN | 16 kHz | `models/gtcrn/gtcrn_simple.onnx` from Xiaobin-Rong/gtcrn |
| RNNoise | 48 kHz | `pyrnnoise` pip package |

The mic rate (8/16/48 kHz) is resampled to each model's rate with a stateful resampler and back, so every card plays audio at the mic rate.

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
