from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np

from .device_spec import parse_device  # also used as rm.parse_device by probe_preproc_gpu.py and tests

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MODEL_RATE = 44_100
RESAMPLE_WINDOW = ("kaiser", 14.769656459379492)
ONNX_HOP = 420
ONNX_WIN = 4 * ONNX_HOP
ONNX_PAD = 441
ALIGN_MIN_CORR = 0.1

# Same ORT settings as the Piper sessions (tts_runtime.py) so both behave alike on a shared GPU/cpuset.
ORT_CUDA_PROVIDER_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"}
ORT_CUDA_INTRA_THREADS = 2
ORT_LOG_SEVERITY = 3

ENHANCER_CHECKPOINT_FILES = (
    "hparams.yaml",
    "ds/G/latest",
    "ds/G/default/mp_rank_00_model_states.pt",
)

# float32 16 kHz window in, float32 16 kHz window of the same length out
WindowModel = Callable[[np.ndarray], np.ndarray]


def ort_session_config(kind: str, index: int) -> tuple[dict, list]:
    """SessionOptions attributes and providers for the denoiser; module settings are read at call time."""
    options: dict = {"log_severity_level": ORT_LOG_SEVERITY}
    if kind != "cuda":
        return options, ["CPUExecutionProvider"]
    # Unset thread counts make ORT size its pool from the host CPUs and fail to set affinity inside a cpuset.
    options["intra_op_num_threads"] = ORT_CUDA_INTRA_THREADS
    options["inter_op_num_threads"] = 1
    cuda = ("CUDAExecutionProvider", {"device_id": index, **ORT_CUDA_PROVIDER_OPTIONS})
    return options, [cuda, "CPUExecutionProvider"]


def _resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(np.asarray(x, dtype=np.float64), up, down, window=RESAMPLE_WINDOW)


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def estimate_lag(ref: np.ndarray, y: np.ndarray, max_shift: int) -> int:
    """Samples by which y is delayed against ref (y[t] ~ ref[t - lag]), within +-max_shift; 0 if unsure."""
    n = len(ref)
    if n == 0 or max_shift <= 0:
        return 0
    size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(ref, size)), size)
    lags = np.arange(-max_shift, max_shift + 1)
    values = corr[lags % size]
    best = int(np.argmax(values))
    norm = float(np.sqrt(np.dot(ref, ref) * np.dot(y, y)))
    if norm < 1e-12 or values[best] / norm < ALIGN_MIN_CORR:
        return 0
    return int(lags[best])


def shift_earlier(y: np.ndarray, lag: int) -> np.ndarray:
    """out[t] = y[t + lag], zero filled at the vacated edge; same length."""
    if lag == 0:
        return y
    out = np.zeros_like(y)
    if lag > 0:
        out[:-lag] = y[lag:]
    else:
        out[-lag:] = y[:lag]
    return out


class OnnxDenoiserModel:
    """Third-party ONNX export of the resemble-enhance denoiser with the official pre/post processing."""

    def __init__(self, onnx_path: Path, device: str) -> None:
        if not onnx_path.is_file():
            raise FileNotFoundError(f"denoiser ONNX not found: {onnx_path}")
        kind, index = parse_device(device)
        import onnxruntime as ort
        import torch

        attributes, providers = ort_session_config(kind, index)
        opts = ort.SessionOptions()
        for name, value in attributes.items():
            setattr(opts, name, value)
        self.session = ort.InferenceSession(str(onnx_path), providers=providers, sess_options=opts)
        self.provider = self.session.get_providers()[0]
        if kind == "cuda" and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"onnxruntime has no CUDA provider (got {self.provider}); install onnxruntime-gpu "
                "matching the image CUDA/cuDNN or set audio_preprocessing.devices to cpu"
            )
        self._window = torch.hann_window(ONNX_WIN)
        logger.info("Audio preprocessing denoiser ready: onnx=%s provider=%s device=%s", onnx_path, self.provider, device)

    def _run_chunk(self, chunk: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        length = len(chunk)
        abs_max = max(float(np.abs(chunk).max()), 1e-7)
        x = torch.from_numpy(np.pad(chunk / abs_max, (0, ONNX_PAD)).astype(np.float32))[None]
        x = x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)
        spec = torch.stft(
            x, n_fft=ONNX_WIN, hop_length=ONNX_HOP, win_length=ONNX_WIN, window=self._window, return_complex=True
        )[..., :-1]
        phase = spec.angle()
        feeds = {"mag": spec.abs().numpy(), "cos": phase.cos().numpy(), "sin": phase.sin().numpy()}
        mag, cos, sin = (torch.from_numpy(o) for o in self.session.run(None, feeds))
        spec = F.pad(torch.complex(mag * cos, mag * sin), (0, 1), "replicate")
        y = torch.istft(
            spec, n_fft=ONNX_WIN, hop_length=ONNX_HOP, win_length=ONNX_WIN, window=self._window, return_complex=False
        )
        y = F.pad(y, (0, x.shape[-1] - y.shape[-1]))[0].numpy().astype(np.float64)
        return y[:length] * abs_max

    def __call__(self, window: np.ndarray) -> np.ndarray:
        n = len(window)
        y44 = self._run_chunk(_resample(window, 441, 160))
        return _fit_length(_resample(y44, 160, 441), n).astype(np.float32)


def _stub_deepspeed() -> None:
    """deepspeed is imported by the training utilities of resemble-enhance only."""
    ds = types.ModuleType("deepspeed")
    acc = types.ModuleType("deepspeed.accelerator")
    acc.get_accelerator = lambda: None
    rt = types.ModuleType("deepspeed.runtime")
    eng = types.ModuleType("deepspeed.runtime.engine")
    eng.DeepSpeedEngine = type("DeepSpeedEngine", (), {})
    ut = types.ModuleType("deepspeed.runtime.utils")
    ut.clip_grad_norm_ = None
    ds.DeepSpeedConfig = type("DeepSpeedConfig", (), {})
    ds.accelerator, ds.runtime = acc, rt
    rt.engine, rt.utils = eng, ut
    mods = (("deepspeed", ds), ("deepspeed.accelerator", acc), ("deepspeed.runtime", rt))
    mods += (("deepspeed.runtime.engine", eng), ("deepspeed.runtime.utils", ut))
    for name, mod in mods:
        sys.modules.setdefault(name, mod)


def _shim_fsolve() -> None:
    """cfm.py does float(fsolve(...)), which numpy >= 2.5 rejects for a 1-element array."""
    import scipy.optimize

    original = scipy.optimize.fsolve
    scipy.optimize.fsolve = lambda *a, **k: np.squeeze(original(*a, **k))


class EnhancerModel:
    """Official resemble-enhance enhancer (denoiser UNet + CFM + vocoder) on one 16 kHz window."""

    def __init__(
        self,
        assets_dir: Path,
        device: str,
        *,
        nfe: int,
        lambd: float,
        tau: float,
        align: bool,
        align_max_samples: int,
    ) -> None:
        source = assets_dir / "resemble-enhance"
        checkpoint = assets_dir / "enhancer_stage2"
        if not (source / "resemble_enhance").is_dir():
            raise FileNotFoundError(f"resemble-enhance source not found: {source}")
        # Never reach load_enhancer with a missing file: its download() would go to the network.
        for rel in ENHANCER_CHECKPOINT_FILES:
            if not (checkpoint / rel).is_file():
                raise FileNotFoundError(f"enhancer checkpoint file not found: {checkpoint / rel}")
        kind, _ = parse_device(device)
        import torch

        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("audio preprocessing device is cuda but torch.cuda is not available")

        _stub_deepspeed()
        _shim_fsolve()
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from resemble_enhance.enhancer.inference import load_enhancer
        from resemble_enhance.inference import inference_chunk, remove_weight_norm_recursively

        self._torch = torch
        self._inference_chunk = inference_chunk
        self._device = device
        self._align = align
        self._align_max = align_max_samples
        self.model = load_enhancer(checkpoint, device)
        self.model.configurate_(nfe=nfe, solver="midpoint", lambd=lambd, tau=tau)
        remove_weight_norm_recursively(self.model)
        logger.info(
            "Audio preprocessing enhancer ready: device=%s nfe=%s lambd=%s tau=%s align=%s",
            device,
            nfe,
            lambd,
            tau,
            align,
        )

    def __call__(self, window: np.ndarray) -> np.ndarray:
        from torchaudio.functional import resample

        n = len(window)
        with self._torch.inference_mode():
            x44 = resample(
                self._torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)),
                orig_freq=SAMPLE_RATE,
                new_freq=MODEL_RATE,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
            y44 = self._inference_chunk(self.model, x44, MODEL_RATE, self._device).numpy()
        y16 = _fit_length(_resample(y44, 160, 441), n)
        if self._align:
            y16 = shift_earlier(y16, estimate_lag(window.astype(np.float64), y16, self._align_max))
        return y16.astype(np.float32)


def load_window_model(
    mode: str,
    assets_dir: str,
    device: str,
    *,
    enhancer_nfe: int,
    enhancer_lambd: float,
    enhancer_tau: float,
    enhancer_align: bool,
    enhancer_align_max_ms: int,
) -> WindowModel:
    root = Path(assets_dir)
    if mode == "denoiser":
        return OnnxDenoiserModel(root / "denoiser.onnx", device)
    if mode == "enhancer":
        return EnhancerModel(
            root,
            device,
            nfe=enhancer_nfe,
            lambd=enhancer_lambd,
            tau=enhancer_tau,
            align=enhancer_align,
            align_max_samples=SAMPLE_RATE * enhancer_align_max_ms // 1000,
        )
    raise ValueError(f"unknown audio preprocessing mode: {mode!r}")
