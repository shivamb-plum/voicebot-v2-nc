"""Smoke test: uv run python test_server.py"""
import json, numpy as np
from fastapi.testclient import TestClient
from server import app
from models import MODELS

def main(SR):
    x = (np.sin(np.arange(int(SR * 1.3)) * 0.05) * 0.3 + np.random.randn(int(SR * 1.3)) * 0.1).astype(np.float32)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"sr": SR}))
        for i in range(0, len(x), 4096):
            ws.send_bytes(x[i:i + 4096].tobytes())
        ws.send_text(json.dumps({"stop": True}))
        out, ms, hdr = {}, {}, None
        while True:
            m = ws.receive()
            if m.get("bytes"):
                out.setdefault(hdr["model"], []).append(np.frombuffer(m["bytes"], np.float32))
                ms.setdefault(hdr["model"], []).append(hdr["ms"])
            else:
                d = json.loads(m["text"])
                if d.get("done"): break
                hdr = d
    assert set(out) == {M.name for M in MODELS}, set(out)
    for name, chunks in out.items():
        y = np.concatenate(chunks)
        assert len(ms[name]) == 3, (name, ms[name])            # 0.5 + 0.5 + 0.3 s
        assert abs(len(y) - len(x)) < SR * 0.35, (name, len(y), len(x))  # bounded window + resampler delay, tail not flushed
        assert np.isfinite(y).all() and np.std(y) < np.std(x), name
    print("ok", SR, "Hz", {k: v for k, v in ms.items()})

if __name__ == "__main__":
    for sr in (48000, 8000):
        main(sr)
