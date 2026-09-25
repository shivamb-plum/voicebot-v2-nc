"""Multi-model realtime denoise comparison. Run: uv run uvicorn server:app"""
import asyncio, json, time
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from models import MODELS

app = FastAPI()


def run_all(models, chunk):
    # ponytail: sequential so per-model timings don't contend for CPU; thread pool if the sum ever exceeds the 500 ms budget
    out = []
    for m in models:
        t = time.perf_counter()
        y = m.process(chunk)
        out.append((m.name, m.sr, round((time.perf_counter() - t) * 1000), y))
    return out


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws(ws: WebSocket):
    """Client sends {"sr": N} then float32 mono PCM at N Hz. Per 0.5 s chunk and per model the server replies with
    a JSON header {"model", "model_sr", "ms", "sec"} followed by one binary frame of enhanced float32 PCM at N Hz."""
    await ws.accept()
    sr, models = 48000, []
    buf = np.zeros(0, np.float32)

    async def flush(chunk):
        for name, msr, ms, y in await asyncio.to_thread(run_all, models, chunk):
            await ws.send_text(json.dumps({"model": name, "model_sr": msr, "ms": ms, "sec": len(chunk) / sr}))
            await ws.send_bytes(y.tobytes())

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        if msg.get("bytes"):
            buf = np.concatenate([buf, np.frombuffer(msg["bytes"], np.float32)])
            while len(buf) >= sr // 2:
                await flush(buf[:sr // 2])
                buf = buf[sr // 2:]
        elif msg.get("text"):
            m = json.loads(msg["text"])
            if "sr" in m:  # session start: fresh model states for this recording
                sr = int(m["sr"])
                models = await asyncio.to_thread(lambda: [M(sr) for M in MODELS])
                buf = buf[:0]
            if m.get("stop"):
                if len(buf):
                    await flush(buf)
                buf = buf[:0]
                await ws.send_text(json.dumps({"done": True}))
