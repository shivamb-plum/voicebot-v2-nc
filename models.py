"""Streaming denoisers behind one interface: m = Cls(input_sr); m.process(float32 @ input_sr) -> float32 @ input_sr."""
import numpy as np, torch, soxr, onnxruntime as ort
from df.enhance import init_df, enhance

# heavy, stateless-per-call singletons shared by all connections
_DFN = init_df(log_level="WARNING")
_DTLN = [ort.InferenceSession(f"models/dtln/model_{i}.onnx", providers=["CPUExecutionProvider"]) for i in (1, 2)]
_GTCRN = ort.InferenceSession("models/gtcrn/gtcrn_simple.onnx", providers=["CPUExecutionProvider"])


class Base:
    name = ""
    sr = 48000  # model rate; process() resamples from/to the input rate with stateful resamplers

    def __init__(self, input_sr: int):
        self.input_sr = input_sr
        self.rs_in = soxr.ResampleStream(input_sr, self.sr, 1, dtype="float32") if input_sr != self.sr else None
        self.rs_out = soxr.ResampleStream(self.sr, input_sr, 1, dtype="float32") if input_sr != self.sr else None

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.rs_in is not None:
            x = self.rs_in.resample_chunk(x)
        y = self.run(x.astype(np.float32)) if len(x) else x
        if self.rs_out is not None and len(y):
            y = self.rs_out.resample_chunk(y.astype(np.float32))
        return y.astype(np.float32)

    def run(self, x):  # x at self.sr
        raise NotImplementedError


class DeepFilterNet3(Base):
    name, sr = "DeepFilterNet3", 48000
    # not a streaming API: re-run with the previous chunk as warm-up context, drop that part of the output
    def __init__(self, input_sr):
        super().__init__(input_sr); self.ctx = np.zeros(0, np.float32)

    def run(self, x):
        model, state, _ = _DFN
        y = enhance(model, state, torch.from_numpy(np.concatenate([self.ctx, x]))[None])[0, len(self.ctx):].numpy()
        self.ctx = x[-self.sr // 2:]
        return y


class DPDFNet(Base):
    name = "DPDFNet2"
    VARIANT = {8000: "dpdfnet2_8khz", 16000: "dpdfnet2"}

    def __init__(self, input_sr):
        import dpdfnet
        self.sr = input_sr  # the package resamples internally; pick the variant trained closest to the input rate
        super().__init__(input_sr)
        self.enh = dpdfnet.StreamEnhancer(model=self.VARIANT.get(input_sr, "dpdfnet2_48khz_hr"))

    def run(self, x):
        return self.enh.process(x, sample_rate=self.sr)


class DTLN(Base):
    name, sr = "DTLN", 16000
    LEN, SHIFT = 512, 128

    def __init__(self, input_sr):
        super().__init__(input_sr)
        self.inp = [{i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for i in s.get_inputs()} for s in _DTLN]
        self.names = [[i.name for i in s.get_inputs()] for s in _DTLN]
        self.in_buf = np.zeros(self.LEN, np.float32); self.out_buf = np.zeros(self.LEN, np.float32)
        self.pending = np.zeros(0, np.float32)

    def run(self, x):
        self.pending = np.concatenate([self.pending, x]); out = []
        while len(self.pending) >= self.SHIFT:
            blk, self.pending = self.pending[:self.SHIFT], self.pending[self.SHIFT:]
            self.in_buf = np.concatenate([self.in_buf[self.SHIFT:], blk])
            spec = np.fft.rfft(self.in_buf); mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
            self.inp[0][self.names[0][0]] = mag
            mask, self.inp[0][self.names[0][1]] = _DTLN[0].run(None, self.inp[0])
            est = np.fft.irfft(mag * mask * np.exp(1j * np.angle(spec))).reshape(1, 1, -1).astype(np.float32)
            self.inp[1][self.names[1][0]] = est
            blk_out, self.inp[1][self.names[1][1]] = _DTLN[1].run(None, self.inp[1])
            self.out_buf = np.concatenate([self.out_buf[self.SHIFT:], np.zeros(self.SHIFT, np.float32)]) + blk_out.squeeze()
            out.append(self.out_buf[:self.SHIFT].copy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class GTCRN(Base):
    name, sr = "GTCRN", 16000
    NFFT, HOP = 512, 256
    WIN = np.sqrt(np.hanning(513)[:512]).astype(np.float32)  # sqrt-hann, periodic

    def __init__(self, input_sr):
        super().__init__(input_sr)
        self.cache = {"conv_cache": np.zeros([2, 1, 16, 16, 33], np.float32), "tra_cache": np.zeros([2, 3, 1, 1, 16], np.float32), "inter_cache": np.zeros([2, 1, 33, 16], np.float32)}
        self.in_buf = np.zeros(self.NFFT, np.float32); self.out_buf = np.zeros(self.NFFT, np.float32)
        self.pending = np.zeros(0, np.float32)

    def run(self, x):
        self.pending = np.concatenate([self.pending, x]); out = []
        while len(self.pending) >= self.HOP:
            blk, self.pending = self.pending[:self.HOP], self.pending[self.HOP:]
            self.in_buf = np.concatenate([self.in_buf[self.HOP:], blk])
            spec = np.fft.rfft(self.in_buf * self.WIN)
            mix = np.stack([spec.real, spec.imag], -1).reshape(1, 257, 1, 2).astype(np.float32)
            enh, *caches = _GTCRN.run(None, {"mix": mix, **self.cache})
            self.cache = dict(zip(self.cache, caches))
            frame = np.fft.irfft(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]) * self.WIN
            self.out_buf = np.concatenate([self.out_buf[self.HOP:], np.zeros(self.HOP, np.float32)]) + frame
            out.append(self.out_buf[:self.HOP].copy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class RNNoise(Base):
    name, sr = "RNNoise", 48000

    def __init__(self, input_sr):
        from pyrnnoise import RNNoise as R
        super().__init__(input_sr); self.d = R(sample_rate=self.sr)

    def run(self, x):
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        frames = [f.reshape(-1) for _, f in self.d.denoise_chunk(pcm)]
        return (np.concatenate(frames).astype(np.float32) / 32767) if frames else np.zeros(0, np.float32)


MODELS = [DeepFilterNet3, DPDFNet, DTLN, GTCRN, RNNoise]
