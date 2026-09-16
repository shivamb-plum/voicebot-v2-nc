"""DeepFilterNet3 realtime demo. Run: uv run uvicorn server:app --reload"""
import asyncio, json, time
import numpy as np, torch, torchaudio
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from df.enhance import init_df, enhance

SR = 48000  # DeepFilterNet3 is a 48 kHz model; other client rates are resampled around it

app = FastAPI()
model, df_state, _ = init_df(log_level="WARNING")


def denoise(ctx: np.ndarray, chunk: np.ndarray, sr: int) -> np.ndarray:
    """ctx is the previous chunk, fed as warm-up context and dropped from the output."""
    x = torch.from_numpy(np.concatenate([ctx, chunk]))[None]
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    y = enhance(model, df_state, x)
    if sr != SR:
        y = torchaudio.functional.resample(y, SR, sr)
    return y[0, len(ctx):len(ctx) + len(chunk)].numpy()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws(ws: WebSocket):
    """Client sends {"sr": N} then float32 mono PCM at N Hz; server replies with enhanced PCM at N Hz per 0.5 s chunk."""
    await ws.accept()
    sr = SR
    buf = ctx = np.zeros(0, np.float32)

    async def flush(chunk):
        nonlocal ctx
        t = time.perf_counter()
        out = await asyncio.to_thread(denoise, ctx, chunk, sr)
        ctx = chunk[-(sr // 2):]
        await ws.send_bytes(out.astype(np.float32).tobytes())
        await ws.send_text(json.dumps({"ms": round((time.perf_counter() - t) * 1000), "sec": len(chunk) / sr}))

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
            sr = int(m.get("sr", sr))
            if m.get("stop"):
                if len(buf):
                    await flush(buf)
                buf = ctx = buf[:0]
                await ws.send_text(json.dumps({"done": True}))
