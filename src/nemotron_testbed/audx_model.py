"""AudX (CNN-LSTM noise reduction on the STFT magnitude, 16 kHz) for audio_preprocessing mode `audx`.

Separate from the modes `denoiser` / `enhancer`, which are resemble-enhance (resemble_models.py): mode `audx`
loads only this model and never resemble-enhance.

The network and its feature chain are written here from the AudX architecture so that a checkpoint trained
with the AudX recipe (state_dict keys cnn.*, lstm.*, fc.*) loads unchanged; no AudX code is imported.
The chain reproduces training and the reference inference:
  * STFT as librosa.stft defaults: n_fft 512, hop 256, periodic Hann, centred frames, zero padding,
  * network input: raw linear magnitude, (frames x 257), no log and no level normalisation,
  * network: Conv2d 1->16 3x3, ReLU, Conv2d 16->32 3x3, ReLU (both padded in time and frequency),
    per frame the 32 channels x 257 bins are flattened channel-major into a 2-layer LSTM (256 units,
    one direction), Linear 256 -> 257 predicts the clean magnitude (unbounded like the reference, unless
    clamp_to_input / max_attenuation_db keep it within [input x 10^(-dB/20), input] per bin),
  * the noisy phase is kept and an inverse STFT returns exactly as many samples as it was given.
Look-ahead: the centred 512-sample frame (32 ms) plus one frame per 3x3 conv (2 x 16 ms) = 64 ms.

Two ways to run it:
  * AudxModel(window): the whole signal at once (a window model, used by the probes),
  * AudxModel.step(state, samples): one stream piece by piece; the LSTM state, the last STFT frames and the
    overlap-add tail live in AudxStreamState, so the concatenated output equals the whole-signal output.
The level matters (no normalisation inside the model: speech at -38 dBFS already comes out worse than it went
in), so step() runs a boost-only AGC before the model and divides the same gain out of its output. With
agc_ceiling_dbfs the AGC never pushes the block it is applied to above that level (docs/audx-stateful-design.md
section 9).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .device_spec import parse_device

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 256
FREQ_BINS = N_FFT // 2 + 1
CHANNELS = 32
HIDDEN = 256
LOOKAHEAD_SAMPLES = N_FFT + 2 * HOP  # a centred frame reaches N_FFT samples ahead, + one hop per 3x3 conv
LOOKAHEAD_MS = LOOKAHEAD_SAMPLES * 1000 // SAMPLE_RATE
CONV_CONTEXT_FRAMES = 2  # two 3x3 convolutions: an output frame needs 2 frames on each side

# AGC: mean power of the 20 ms blocks above the floor, first such block taken as is, then smoothed with
# time constant AGC_TAU_S. Measured (pink noise 5 dB, inputs 0..-30 dB): tau 1-3 s matches a per-file oracle
# gain; a fast-attack tracker (0.3 s up / 5 s down) follows the peaks and boosts ~3 dB SI-SDR too little.
AGC_BLOCK = SAMPLE_RATE // 50
AGC_FLOOR_DBFS = -65.0
AGC_TAU_S = 2.0
_ENVELOPE_FLOOR = 1e-11  # torch.istft refuses a window envelope below this

# state_dict of the AudX network: name -> shape
EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "cnn.0.weight": (16, 1, 3, 3),
    "cnn.0.bias": (16,),
    "cnn.2.weight": (CHANNELS, 16, 3, 3),
    "cnn.2.bias": (CHANNELS,),
    "lstm.weight_ih_l0": (4 * HIDDEN, CHANNELS * FREQ_BINS),
    "lstm.weight_hh_l0": (4 * HIDDEN, HIDDEN),
    "lstm.bias_ih_l0": (4 * HIDDEN,),
    "lstm.bias_hh_l0": (4 * HIDDEN,),
    "lstm.weight_ih_l1": (4 * HIDDEN, HIDDEN),
    "lstm.weight_hh_l1": (4 * HIDDEN, HIDDEN),
    "lstm.bias_ih_l1": (4 * HIDDEN,),
    "lstm.bias_hh_l1": (4 * HIDDEN,),
    "fc.weight": (FREQ_BINS, HIDDEN),
    "fc.bias": (FREQ_BINS,),
}


def build_network():
    """The AudX CNN-LSTM, randomly initialised (load a checkpoint into it)."""
    import torch.nn as nn

    class AudxNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, CHANNELS, 3, padding=1),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(CHANNELS * FREQ_BINS, HIDDEN, num_layers=2, batch_first=True)
            self.fc = nn.Linear(HIDDEN, FREQ_BINS)

        def forward(self, magnitude):  # (B, T, F) -> (B, T, F)
            x = self.cnn(magnitude.unsqueeze(1))  # (B, C, T, F)
            x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)  # (B, T, C*F), channel-major
            x, _ = self.lstm(x)
            return self.fc(x)

    return AudxNet()


def check_state_dict(state: object) -> None:
    """Raise ValueError unless state is exactly the AudX state_dict (names and shapes)."""
    if not isinstance(state, dict):
        raise ValueError(f"AudX checkpoint must hold a state_dict, got {type(state).__name__}")
    missing = sorted(set(EXPECTED_SHAPES) - set(state))
    unexpected = sorted(set(state) - set(EXPECTED_SHAPES))
    wrong = sorted(
        f"{name} {tuple(state[name].shape)} != {shape}"
        for name, shape in EXPECTED_SHAPES.items()
        if name in state and tuple(getattr(state[name], "shape", ())) != shape
    )
    if missing or unexpected or wrong:
        raise ValueError(
            f"not an AudX state_dict: missing={missing} unexpected={unexpected} wrong_shape={wrong}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unwrap_checkpoint(loaded: object) -> tuple[object, dict[str, Any]]:
    """(state_dict, metadata): a bare state_dict, or {'state_dict': ..., plain metadata} as the POC converter writes."""
    if isinstance(loaded, dict) and "state_dict" in loaded:
        meta = {k: v for k, v in loaded.items() if k != "state_dict"}
        bad = sorted(k for k, v in meta.items() if not isinstance(v, (str, int, float, bool)))
        if bad:
            raise ValueError(f"not an AudX state_dict: wrapper has non-metadata entries {bad}")
        return loaded["state_dict"], meta
    return loaded, {}


@dataclass(eq=False)
class AudxStreamState:
    """Everything AudxModel.step() keeps between the pieces of one stream; one instance per stream, used in order.

    Positions are input sample indices; output sample i is the enhanced input sample i. STFT frame f covers
    input [f*HOP - HOP, f*HOP + HOP) (centred), so `pending` starts at frames*HOP - HOP (the left zero pad first).
    """

    received: int = 0
    emitted: int = 0
    frames: int = 0  # STFT frames computed
    next_frame: int = 0  # frames already through the network and overlap-add
    pending: np.ndarray = field(default_factory=lambda: np.zeros(HOP, dtype=np.float32))
    spectra: np.ndarray = field(default_factory=lambda: np.zeros((0, FREQ_BINS), dtype=np.complex64))
    hidden: Any = None  # LSTM (h, c)
    tail: np.ndarray = field(default_factory=lambda: np.zeros(HOP, dtype=np.float32))
    tail_norm: np.ndarray = field(default_factory=lambda: np.zeros(HOP, dtype=np.float32))
    gains: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))  # inputs [emitted, received)
    agc_level: float = 0.0  # tracked power; 0 = nothing above the floor yet
    agc_from: float = 1.0
    agc_to: float = 1.0
    agc_power: float = 0.0  # sum of squares of the current, incomplete 20 ms block
    finished: bool = False


class AudxModel:
    """AudX on float32 16 kHz audio: __call__ for a whole signal, step() for a stream piece by piece."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        *,
        agc_target_dbfs: float = -28.0,
        agc_max_gain_db: float = 30.0,
        agc_ceiling_dbfs: float = 0.0,
        clamp_to_input: bool = False,
        max_attenuation_db: float = 0.0,
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AudX checkpoint not found: {checkpoint}")
        kind, index = parse_device(device)
        import torch

        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("audio preprocessing device is cuda but torch.cuda is not available")
        self._torch = torch
        self._device = torch.device("cpu" if kind == "cpu" else f"cuda:{index}")
        # weights_only: a checkpoint is a pickle; never let it build arbitrary objects.
        state, meta = unwrap_checkpoint(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
        check_state_dict(state)
        net = build_network()
        net.load_state_dict(state, strict=True)
        self.net = net.to(self._device).eval()
        lstm = self.net.lstm
        # (W_ih, W_hh, b_ih + b_hh) per layer for the step loop
        self._lstm_layers = [
            (
                getattr(lstm, f"weight_ih_l{k}").detach(),
                getattr(lstm, f"weight_hh_l{k}").detach(),
                (getattr(lstm, f"bias_ih_l{k}") + getattr(lstm, f"bias_hh_l{k}")).detach(),
            )
            for k in range(lstm.num_layers)
        ]
        self._window = torch.hann_window(N_FFT, periodic=True, device=self._device)
        self._np_window = torch.hann_window(N_FFT, periodic=True).numpy().astype(np.float32)
        square = self._np_window * self._np_window
        self._w2_head, self._w2_tail = square[:HOP].copy(), square[HOP:].copy()
        self._agc_target = 10.0 ** (agc_target_dbfs / 10.0)
        self._agc_max = 10.0 ** (agc_max_gain_db / 20.0)
        self._agc_alpha = math.exp(-AGC_BLOCK / SAMPLE_RATE / AGC_TAU_S)
        self._agc_floor = 10.0 ** (AGC_FLOOR_DBFS / 10.0)
        # 0 = off for both: no ceiling on the boosted block, no limit on how far a bin may be cut
        self._agc_ceiling = 10.0 ** (agc_ceiling_dbfs / 10.0) if agc_ceiling_dbfs != 0.0 else None
        self._clamp_to_input = bool(clamp_to_input)
        self._min_ratio = 10.0 ** (-max_attenuation_db / 20.0) if max_attenuation_db > 0.0 else None
        self.sha256 = file_sha256(checkpoint)
        agc = "off"
        if agc_max_gain_db > 0:
            ceiling = f"{agc_ceiling_dbfs} dBFS" if self._agc_ceiling is not None else "off"
            agc = f"target {agc_target_dbfs} dBFS max +{agc_max_gain_db} dB ceiling {ceiling}"
        clamp = []
        if self._clamp_to_input:
            clamp.append("to input")
        if self._min_ratio is not None:
            clamp.append(f"max attenuation {max_attenuation_db} dB")
        logger.info(
            "Audio preprocessing AudX ready: checkpoint=%s sha256=%s device=%s meta=%s agc=%s clamp=%s",
            checkpoint,
            self.sha256,
            device,
            meta,
            agc,
            ", ".join(clamp) or "off",
        )

    @staticmethod
    def new_state() -> AudxStreamState:
        return AudxStreamState()

    def step(self, state: AudxStreamState, samples: np.ndarray, final: bool = False) -> np.ndarray:
        """Feed the next samples of a stream; return the enhanced samples that are now complete.

        Output lags input by at most LOOKAHEAD_SAMPLES; final=True ends the stream and returns the rest, so
        the outputs of all steps add up to exactly the input length. Not thread safe per state.
        """
        if state.finished:
            raise RuntimeError("AudX stream already finished")
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError("AudX input contains NaN or Inf")
        if x.size:
            gains = self._agc(state, x)
            if gains is not None:
                x = x * gains
                state.gains = np.concatenate((state.gains, gains))
            state.pending = np.concatenate((state.pending, x))
            state.received += x.size
        if final:
            state.finished = True
            if state.received == 0:
                return np.zeros(0, dtype=np.float32)
            target = state.received // HOP + 1  # as torch.stft(center=True)
            short = (target - state.frames - 1) * HOP + N_FFT - state.pending.size
            if short > 0:
                state.pending = np.concatenate((state.pending, np.zeros(short, dtype=np.float32)))
            self._analyse(state, limit=target - state.frames)
        else:
            self._analyse(state)
        ready = state.frames if final else state.frames - CONV_CONTEXT_FRAMES
        parts = []
        if ready > state.next_frame:
            parts.append(self._synthesise(state, ready))
        if final:
            # the last HOP positions only have the last frame; cut at the input length
            start = (state.frames - 1) * HOP
            last = state.tail / np.maximum(state.tail_norm, _ENVELOPE_FLOOR)
            parts.append(last[: state.received - start])
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        if self._agc_max > 1.0 and out.size:
            out = out / state.gains[: out.size]
            state.gains = state.gains[out.size :]
        state.emitted += out.size
        if final and state.emitted != state.received:
            raise RuntimeError(f"AudX stream ended with {state.emitted} of {state.received} samples")
        return out.astype(np.float32, copy=False)

    def _agc(self, state: AudxStreamState, x: np.ndarray) -> np.ndarray | None:
        """Per-sample gain for x, decided from earlier 20 ms blocks only, ramped across each block.

        With a ceiling, the gain also never lifts the block it is applied to (the part of it in x) above the
        ceiling: it drops at once, at the start of that part, and rises again ramped as usual.
        """
        if self._agc_max <= 1.0:
            return None
        gains = np.empty(x.size, dtype=np.float32)
        i = 0
        while i < x.size:
            offset = (state.received + i) % AGC_BLOCK
            if offset == 0:
                state.agc_from = state.agc_to
                level = state.agc_level
                state.agc_to = (
                    1.0 if level <= 0.0 else min(max(math.sqrt(self._agc_target / level), 1.0), self._agc_max)
                )
            take = min(AGC_BLOCK - offset, x.size - i)
            seg = x[i : i + take].astype(np.float64)
            energy = float(np.dot(seg, seg))
            if self._agc_ceiling is not None and energy > 0.0:
                cap = max(math.sqrt(self._agc_ceiling * take / energy), 1.0)
                state.agc_from = min(state.agc_from, cap)
                state.agc_to = min(state.agc_to, cap)
            ramp = np.arange(offset + 1, offset + take + 1, dtype=np.float64) / AGC_BLOCK
            gains[i : i + take] = state.agc_from + (state.agc_to - state.agc_from) * ramp
            state.agc_power += energy
            if offset + take == AGC_BLOCK:
                power = state.agc_power / AGC_BLOCK
                state.agc_power = 0.0
                if power > self._agc_floor:
                    if state.agc_level <= 0.0:
                        state.agc_level = power
                    else:
                        a = self._agc_alpha
                        state.agc_level = a * state.agc_level + (1.0 - a) * power
            i += take
        return gains

    def _analyse(self, state: AudxStreamState, limit: int | None = None) -> None:
        """STFT frames for the pending input that is long enough."""
        count = (state.pending.size - N_FFT) // HOP + 1 if state.pending.size >= N_FFT else 0
        if limit is not None:
            count = min(count, limit)
        if count <= 0:
            return
        frames = np.lib.stride_tricks.sliding_window_view(state.pending, N_FFT)[::HOP][:count]
        spectra = np.fft.rfft(frames * self._np_window, axis=1).astype(np.complex64)
        state.spectra = np.concatenate((state.spectra, spectra))
        state.pending = state.pending[count * HOP :].copy()
        state.frames += count

    def _synthesise(self, state: AudxStreamState, ready: int) -> np.ndarray:
        """Run frames [next_frame, ready) through the network and the overlap-add; return the complete samples."""
        torch = self._torch
        base = state.frames - state.spectra.shape[0]
        magnitude = np.abs(state.spectra)
        with torch.inference_mode():
            m = torch.from_numpy(magnitude).to(self._device).unsqueeze(0)
            feat = self.net.cnn(m.unsqueeze(1))  # (1, C, L, F); the edge frames of the local window are unused
            feat = feat.permute(0, 2, 1, 3).reshape(1, feat.size(2), -1)
            seq, state.hidden = self._lstm(feat[0, state.next_frame - base : ready - base], state.hidden)
            estimate = self.net.fc(seq).float().cpu().numpy()  # (n, F)
        noisy = state.spectra[state.next_frame - base : ready - base]
        size = np.abs(noisy)
        if self._clamp_to_input:
            estimate = np.minimum(estimate, size)
        if self._min_ratio is not None:
            estimate = np.maximum(estimate, size * np.float32(self._min_ratio))
        phase = np.where(size > 0.0, noisy / np.maximum(size, 1e-30), 1.0)  # angle(0) = 0 as torch.angle
        frames = np.fft.irfft(estimate * phase, n=N_FFT, axis=1).astype(np.float32) * self._np_window
        n = ready - state.next_frame
        y = np.zeros((n + 1) * HOP, dtype=np.float32)
        norm = np.zeros((n + 1) * HOP, dtype=np.float32)
        y[:HOP] += state.tail
        norm[:HOP] += state.tail_norm
        y[: n * HOP] += frames[:, :HOP].reshape(-1)
        norm[: n * HOP] += np.tile(self._w2_head, n)
        y[HOP:] += frames[:, HOP:].reshape(-1)
        norm[HOP:] += np.tile(self._w2_tail, n)
        complete = y[: n * HOP] / np.maximum(norm[: n * HOP], _ENVELOPE_FLOOR)
        state.tail, state.tail_norm = y[n * HOP :], norm[n * HOP :]
        start = (state.next_frame - 1) * HOP  # frame next_frame starts one hop before its centre
        state.next_frame = ready
        keep_from = max(0, ready - CONV_CONTEXT_FRAMES)
        state.spectra = state.spectra[keep_from - base :].copy()
        return complete[-start:] if start < 0 else complete

    def _lstm(self, x, hidden):
        """nn.LSTM (gates i, f, g, o) for a few frames (T, In) -> (T, H); a plain loop because torch's CPU LSTM
        spends ~8 ms per call on fixed overhead (oneDNN weight reorder), which a 5-frame step cannot afford."""
        torch = self._torch
        F = torch.nn.functional
        if hidden is None:
            zero = x.new_zeros(len(self._lstm_layers), 1, HIDDEN)
            hidden = (zero, zero)
        elif hidden[0].device != x.device:
            hidden = tuple(t.to(x.device) for t in hidden)
        h_all, c_all = hidden
        h_out, c_out = [], []
        layer_in = x
        for k, (w_ih, w_hh, bias) in enumerate(self._lstm_layers):
            gates_x = F.linear(layer_in, w_ih, bias)
            h, c = h_all[k], c_all[k]
            outs = []
            for t in range(gates_x.size(0)):
                i, f, g, o = (gates_x[t : t + 1] + F.linear(h, w_hh)).chunk(4, dim=-1)
                c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h = torch.sigmoid(o) * torch.tanh(c)
                outs.append(h)
            layer_in = torch.cat(outs, dim=0)
            h_out.append(h)
            c_out.append(c)
        return layer_in, (torch.stack(h_out), torch.stack(c_out))

    def __call__(self, window: np.ndarray) -> np.ndarray:
        torch = self._torch
        n = len(window)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).to(self._device)
        with torch.inference_mode():
            spec = torch.stft(
                x,
                n_fft=N_FFT,
                hop_length=HOP,
                win_length=N_FFT,
                window=self._window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )  # (F, T)
            size = spec.abs()
            estimate = self.net(size.T.unsqueeze(0))[0].T  # (F, T), may be negative like the reference
            if self._clamp_to_input:
                estimate = torch.minimum(estimate, size)
            if self._min_ratio is not None:
                estimate = torch.maximum(estimate, size * self._min_ratio)
            phase = torch.angle(spec)
            enhanced = torch.complex(estimate * torch.cos(phase), estimate * torch.sin(phase))
            y = torch.istft(
                enhanced,
                n_fft=N_FFT,
                hop_length=HOP,
                win_length=N_FFT,
                window=self._window,
                center=True,
                length=n,
            )
        return y.float().cpu().numpy()
