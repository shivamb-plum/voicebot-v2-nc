"""Smoke test: uv run python test_server.py"""
import json, numpy as np
from fastapi.testclient import TestClient
from server import app, SR

def main():
    x = (np.sin(np.arange(int(SR * 1.3)) * 0.05) * 0.3 + np.random.randn(int(SR * 1.3)) * 0.1).astype(np.float32)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"sr": SR}))
        for i in range(0, len(x), 4096):
            ws.send_bytes(x[i:i + 4096].tobytes())
        ws.send_text(json.dumps({"stop": True}))
        out, lat = [], []
        while True:
            m = ws.receive()
            if m.get("bytes"): out.append(np.frombuffer(m["bytes"], np.float32))
            else:
                d = json.loads(m["text"])
                if d.get("done"): break
                lat.append(d["ms"])
    y = np.concatenate(out)
    assert len(y) == len(x), (len(y), len(x))
    assert len(lat) == 3, lat  # 0.5 + 0.5 + 0.3 s
    assert np.std(y) < np.std(x), "enhanced should have less energy than white-noise input"
    print("ok", len(y), "samples, latencies ms:", lat)

if __name__ == "__main__":
    main()
