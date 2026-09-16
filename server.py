"""DeepFilterNet3 realtime demo. Run: uv run uvicorn server:app --reload"""
import asyncio, json, time
import numpy as np, torch, torchaudio
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from df.enhance import init_df, enhance

SR = 48000            # DeepFilterNet3 is a 48 kHz model
CHUNK = SR // 2       # 0.5 s of audio per model call
CONTEXT = SR // 2     # previous 0.5 s prepended as warm-up context, its output discarded

app = FastAPI()
model, df_state, _ = init_df(log_level="WARNING")


def denoise(ctx: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.concatenate([ctx, chunk]))[None]
    return enhance(model, df_state, x)[0, len(ctx):].numpy()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws(ws: WebSocket):
    """Client sends float32 mono PCM frames; server replies with enhanced float32 PCM per CHUNK."""
    await ws.accept()
    sr = SR
    buf = ctx = np.zeros(0, np.float32)

    async def flush(chunk):
        nonlocal ctx
        t = time.perf_counter()
        out = await asyncio.to_thread(denoise, ctx, chunk)
        ctx = chunk[-CONTEXT:]
        await ws.send_bytes(out.astype(np.float32).tobytes())
        await ws.send_text(json.dumps({"ms": round((time.perf_counter() - t) * 1000), "sec": len(chunk) / SR}))

    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        if msg.get("bytes"):
            x = np.frombuffer(msg["bytes"], np.float32)
            if sr != SR:  # ponytail: per-packet resample, tiny seams at packet edges; move resampling client-side if audible
                x = torchaudio.functional.resample(torch.from_numpy(x.copy()), sr, SR).numpy()
            buf = np.concatenate([buf, x])
            while len(buf) >= CHUNK:
                await flush(buf[:CHUNK])
                buf = buf[CHUNK:]
        elif msg.get("text"):
            m = json.loads(msg["text"])
            sr = m.get("sr", sr)
            if m.get("stop"):
                if len(buf):
                    await flush(buf)
                buf = ctx = buf[:0]
                await ws.send_text(json.dumps({"done": True}))
