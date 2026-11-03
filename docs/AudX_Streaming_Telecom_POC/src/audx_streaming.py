"""AudX CNN-LSTM streaming proof of concept (16kHz mono float32).

Uses the trained unidirectional LSTM and an exact 2-frame CNN lookahead.
Model state is immutable/shared; StreamState is per media leg.

IMPORTANT: Spectral convention here is Hann, n_fft=512, hop=256,
center=True, pad_mode=constant and input-magnitude/noisy-phase reconstruction.
Check numerical parity with the original repo utils.audio before deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 16_000
N_FFT = 512
HOP = 256
FREQ_BINS = N_FFT // 2 + 1
CONV_FUTURE_FRAMES = 2


class AudXCNNLSTM(nn.Module):
    """Architecture identified from the uploaded 9,282,753-parameter checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=32 * FREQ_BINS,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(256, FREQ_BINS)

    def forward(self, magnitudes: torch.Tensor) -> torch.Tensor:
        x = self.cnn(magnitudes.unsqueeze(1))
        x = x.permute(0, 2, 1, 3).contiguous().flatten(2)
        x, _state = self.lstm(x)
        return self.fc(x)


def load_safe_weights(weights_path: str | Path, device: str = "cpu") -> AudXCNNLSTM:
    """Do not deserialize the original full-model checkpoint in production."""
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model = AudXCNNLSTM()
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model


def _rfft(spec_frame: np.ndarray, window: np.ndarray) -> np.ndarray:
    return np.fft.rfft(spec_frame * window, n=N_FFT).astype(np.complex64)


def _unit_phase(spectrum: np.ndarray) -> np.ndarray:
    magnitude = np.abs(spectrum)
    return (spectrum / np.maximum(magnitude, 1.0e-12)).astype(np.complex64)


def _check_wave(wave: np.ndarray) -> np.ndarray:
    arr = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Input audio contains NaN/Inf")
    return arr


@dataclass
class AudXTimings:
    model_ms: float = 0.0
    dsp_ms: float = 0.0
    calls: int = 0
    frames: int = 0


class AudXStream:
    """Incremental audio samples -> incremental enhanced audio samples.

    Each stream tracks its own STFT, 2 CNN context frames, LSTM h/c state,
    overlap-add iSTFT, timestamps via emitted sample count.

    `push()` returns 0 or more samples and must be called serially per stream.
    `flush()` returns the remaining audio and then invalidates the stream.
    """

    def __init__(self, model: AudXCNNLSTM, device: str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.window = torch.hann_window(N_FFT, periodic=True).numpy().astype(np.float32)
        self.window_sq = self.window * self.window
        self._stft_audio = np.zeros(HOP, dtype=np.float32)  # center=True, constant left pad
        self._input_total = 0
        self._emitted_total = 0
        self._total_frames = 0
        self._features_base = 0
        self._next_frame = 0
        self._mags: list[np.ndarray] = []
        self._phases: list[np.ndarray] = []
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._ola = np.empty(0, dtype=np.float32)
        self._norm = np.empty(0, dtype=np.float32)
        self._ola_origin = 0
        self._closed = False
        self.timings = AudXTimings()

    @property
    def emitted_samples(self) -> int:
        return self._emitted_total

    @property
    def received_samples(self) -> int:
        return self._input_total

    def push(self, samples: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudXStream has been flushed")
        x = _check_wave(samples)
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        t0 = time.perf_counter()
        self._input_total += x.size
        self._stft_audio = np.concatenate((self._stft_audio, x))
        self._append_stft_frames()
        self.timings.dsp_ms += (time.perf_counter() - t0) * 1000
        return self._run_and_output(final=False)

    def flush(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudXStream has been flushed")
        self._closed = True
        if self._input_total == 0:
            return np.empty(0, dtype=np.float32)
        # center=True STFT adds an extra HOP samples of right zero padding.
        target_frames = self._input_total // HOP + 1
        self._stft_audio = np.concatenate((self._stft_audio, np.zeros(HOP + N_FFT, np.float32)))
        self._append_stft_frames(max_frames=target_frames)
        if self._total_frames != target_frames:
            raise RuntimeError(f"STFT flush produced {self._total_frames} frames, need {target_frames}")
        return self._run_and_output(final=True)

    def _append_stft_frames(self, max_frames: int | None = None) -> None:
        while len(self._stft_audio) >= N_FFT and (max_frames is None or self._total_frames < max_frames):
            spec = _rfft(self._stft_audio[:N_FFT], self.window)
            self._mags.append(np.abs(spec).astype(np.float32))
            self._phases.append(_unit_phase(spec))
            self._total_frames += 1
            self._stft_audio = self._stft_audio[HOP:]

    def _run_and_output(self, *, final: bool) -> np.ndarray:
        # Two 3x3 temporal convolutions create right context of exactly 2 frames.
        ready_count = self._total_frames if final else max(0, self._total_frames - CONV_FUTURE_FRAMES)
        if ready_count <= self._next_frame:
            return np.empty(0, dtype=np.float32)

        t0 = time.perf_counter()
        feats = np.stack(self._mags, axis=0)
        tensor = torch.from_numpy(feats).to(device=self.device).unsqueeze(0)
        with torch.inference_mode():
            conv = self.model.cnn(tensor.unsqueeze(1))
            conv = conv.permute(0, 2, 1, 3).contiguous().flatten(2)
            start = self._next_frame - self._features_base
            stop = ready_count - self._features_base
            sequence = conv[:, start:stop, :]
            lstm_output, self._hidden = self.model.lstm(sequence, self._hidden)
            # Retain recurrent values but release graph (no autograd).
            self._hidden = tuple(state.detach() for state in self._hidden)
            predicted = self.model.fc(lstm_output)[0].detach().cpu().numpy()
        self.timings.model_ms += (time.perf_counter() - t0) * 1000
        self.timings.calls += 1
        self.timings.frames += int(predicted.shape[0])

        t1 = time.perf_counter()
        results = []
        for j, magnitude in enumerate(predicted):
            offset = self._next_frame + j - self._features_base
            self._add_ola(magnitude, self._phases[offset], self._next_frame + j)
            # Future frames may overlap only at >= idx * HOP.
            finalized = (self._next_frame + j) * HOP
            if final and self._next_frame + j == ready_count - 1:
                finalized = self._input_total
            audio = self._emit_until(finalized)
            if audio.size:
                results.append(audio)
        self._next_frame = ready_count
        # Only the previous two features are required for the CNN receptive field.
        drop_until = max(0, ready_count - CONV_FUTURE_FRAMES)
        count_to_drop = drop_until - self._features_base
        if count_to_drop > 0:
            self._mags = self._mags[count_to_drop:]
            self._phases = self._phases[count_to_drop:]
            self._features_base = drop_until
        self.timings.dsp_ms += (time.perf_counter() - t1) * 1000
        return np.concatenate(results) if results else np.empty(0, dtype=np.float32)

    def _add_ola(self, magnitude: np.ndarray, phase: np.ndarray, frame_idx: int) -> None:
        spectrum = magnitude.astype(np.float32) * phase
        wav_frame = np.fft.irfft(spectrum, n=N_FFT).real.astype(np.float32)
        start = frame_idx * HOP - HOP
        finish = start + N_FFT
        valid_start = max(0, start)
        required = finish - self._ola_origin
        if required > len(self._ola):
            extent = required - len(self._ola)
            self._ola = np.pad(self._ola, (0, extent))
            self._norm = np.pad(self._norm, (0, extent))
        if valid_start < finish:
            win_start = valid_start - start
            write_start = valid_start - self._ola_origin
            length = finish - valid_start
            self._ola[write_start:write_start+length] += wav_frame[win_start:] * self.window[win_start:]
            self._norm[write_start:write_start+length] += self.window_sq[win_start:]

    def _emit_until(self, sample_idx: int) -> np.ndarray:
        end = min(sample_idx, self._input_total)
        length = end - self._ola_origin
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        pcm = self._ola[:length] / np.maximum(self._norm[:length], 1.0e-10)
        out = pcm.astype(np.float32, copy=True)
        self._ola = self._ola[length:]
        self._norm = self._norm[length:]
        self._ola_origin += length
        self._emitted_total += length
        return out


def offline_reference(model: AudXCNNLSTM, audio: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Reimplementation of full-waveform magnitude+noisy-phase inference.

    It uses an explicit centered Hann STFT to make streaming parity testable.
    This reference still needs comparison to original AudX `utils.audio`.
    """
    x = _check_wave(audio)
    if x.size == 0:
        return np.empty(0, dtype=np.float32)
    win = torch.hann_window(N_FFT, periodic=True)
    stft = torch.stft(torch.from_numpy(x), n_fft=N_FFT, hop_length=HOP,
                      window=win, center=True, pad_mode="constant", return_complex=True)
    mag = stft.abs().transpose(0, 1).to(device)
    with torch.inference_mode():
        pred = model(mag.unsqueeze(0))[0].transpose(0, 1).cpu()
    est = pred * (stft / stft.abs().clamp(min=1.e-12))
    return torch.istft(est, n_fft=N_FFT, hop_length=HOP, window=win, center=True, length=x.size).numpy()
