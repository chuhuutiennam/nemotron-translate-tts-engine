from __future__ import annotations

import logging as _logging
import os
import re
from dataclasses import dataclass, field
from typing import Any


_LOOKAHEAD_TO_CHUNK_MS = {
    0: 80,
    1: 160,
    3: 320,
    6: 560,
    13: 1120,
}

_VALID_CT2_COMPUTE = frozenset({
    "int8", "int8_float16", "int8_bfloat16",
    "float16", "bfloat16", "float32", "int8_float32",
})


_CONFIG_FILE_NAME = "nemotron-engine.yaml"

# Sentinel — distinguishes "key absent from YAML" from explicit null/0/False.
_MISSING: Any = object()


# ── Low-level env helpers ──────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid number")


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_devices(name: str, fallback: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return [fallback]
    devices = [d.strip() for d in raw.split(",") if d.strip()]
    return devices if devices else [fallback]


# ── YAML file loader ───────────────────────────────────────────────────────────

def _load_yaml_file(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for file-based config. "
            "Install with: pip install pyyaml"
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path!r}: expected a YAML mapping at the top level, "
            f"got {type(data).__name__}"
        )
    return data


def _discover_config_file(explicit_path: str | None) -> str | None:
    """Return the config file path to use, or None if none found."""
    # 1. Caller-supplied explicit path (from test or custom invocation)
    if explicit_path:
        return explicit_path

    # 2. ENGINE_CONFIG_FILE env var (operator override)
    from_env = os.getenv("ENGINE_CONFIG_FILE", "").strip()
    if from_env:
        return from_env

    # 3. $APP_DIR/etc/nemotron-engine.yaml
    app_dir = os.getenv("APP_DIR", "").strip()
    if app_dir:
        candidate = os.path.join(app_dir, "etc", _CONFIG_FILE_NAME)
        if os.path.isfile(candidate):
            return candidate

    # 4. ./etc/nemotron-engine.yaml relative to cwd
    candidate = os.path.abspath(os.path.join("etc", _CONFIG_FILE_NAME))
    if os.path.isfile(candidate):
        return candidate

    return None


# ── Unified config accessor ────────────────────────────────────────────────────

class _Cfg:
    """Reads config values in priority order: YAML file > env var > code default.

    YAML keys use dot-notation paths (e.g. "tts.magpie.model_path").
    If a key is absent from the YAML file the env var is consulted instead.
    Explicit YAML ``null`` values fall through to the code default (same
    semantics as an unset env var).
    """

    def __init__(self, data: dict | None, source: str) -> None:
        self._data: dict = data if data is not None else {}
        self.source = source  # "file:/path/to/file" or "env"
        self.warnings: list[str] = []  # e.g. deprecated YAML paths; main logs them after logging is set up

    def _get(self, path: str) -> Any:
        """Navigate a dot-separated key path. Returns _MISSING when any key absent."""
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return _MISSING
            node = node[key]
        return node  # may be None for explicit YAML null

    def str(self, yaml_path: str, env_name: str, default: str = "", strip: bool = True) -> str:
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return default
            s = str(val)
            return s.strip() if strip else s
        raw = os.getenv(env_name, default)
        return raw.strip() if strip else raw

    def int(self, yaml_path: str, env_name: str, default: int) -> int:
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return default
            if isinstance(val, bool):
                raise ValueError(f"Config {yaml_path!r}: expected int, got boolean")
            try:
                return int(val)
            except (ValueError, TypeError):
                raise ValueError(f"Config {yaml_path!r}={val!r} is not a valid integer")
        return _env_int(env_name, default)

    def float(self, yaml_path: str, env_name: str, default: float) -> float:
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Config {yaml_path!r}={val!r} is not a valid number")
        return _env_float(env_name, default)

    def bool(self, yaml_path: str, env_name: str, default: bool) -> bool:
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in {"1", "true", "yes", "on"}
        return _env_bool(env_name, default)

    def devices(
        self,
        yaml_path: str,
        env_name: str,
        fallback_yaml: str | None = None,
        fallback_env: str | None = None,
    ) -> list[str]:
        """Read a device list from YAML (list or comma-string) or env var.

        Empty or absent -> [single-device fallback key], or ["cpu"] when no fallback key is given.
        """
        fallback = self.str(fallback_yaml, fallback_env, "cpu") if fallback_yaml and fallback_env else "cpu"
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return [fallback]
            if isinstance(val, list):
                devices = [str(d).strip() for d in val if d is not None and str(d).strip()]
                return devices if devices else [fallback]
            devices = [d.strip() for d in str(val).split(",") if d.strip()]
            return devices if devices else [fallback]
        return _env_devices(env_name, _env_str(fallback_env, "cpu") if fallback_env else "cpu")


# ── Audio pre-processing (resemble-enhance) ────────────────────────────────────

# (left_context_ms, hop_ms, right_context_ms, compute_budget_ms) used when the
# config value is 0. audx is stateful (no window): left is unused, right is the
# model look-ahead (fixed), hop is how often a stream sends its audio to the model.
AUDX_LOOKAHEAD_MS = 64  # audx_model.LOOKAHEAD_MS, kept here so config needs no torch
_AUDIO_PREPROC_WINDOW_DEFAULTS = {
    "denoiser": (1000, 500, 500, 500),
    "enhancer": (1000, 2000, 1000, 1500),
    "audx": (0, 80, AUDX_LOOKAHEAD_MS, 100),
}


@dataclass(frozen=True, slots=True)
class AudioPreprocessSettings:
    """Streaming audio pre-processing; every default describes the OFF state."""

    enabled: bool = False
    model: str = "resemble"  # resemble | audx (YAML audio_preprocessing.model)
    mode: str = "denoiser"  # effective: denoiser | enhancer (resemble.mode) | audx (model audx)
    sample_percent: int = 100
    assets_dir: str = "models/resemble-enhance"
    devices: tuple[str, ...] = ("cpu",)
    workers: int = 1
    queue_max: int = 64
    max_streams: int = 0
    # Effective window of the selected model (resemble.window or audx.window, defaults already applied).
    left_context_ms: int = 1000
    hop_ms: int = 500
    right_context_ms: int = 500
    crossfade_ms: int = 40
    compute_budget_ms: int = 500
    enhancer_nfe: int = 32
    enhancer_lambd: float = 1.0
    enhancer_tau: float = 0.5
    enhancer_align_regions: bool = True
    enhancer_align_max_ms: int = 40
    dither_sigma_lsb: float = 3.0
    dither_min_zero_run: int = 64
    dither_seed: int = 0
    vad_input: str = "enhanced"
    fallback_bad_regions: int = 3
    fallback_bypass_s: float = 10.0
    # mode audx only
    audx_checkpoint: str = "models/audx/audx_weights_safe.pth"
    audx_agc_target_dbfs: float = -28.0
    audx_agc_max_gain_db: float = 30.0
    audx_agc_ceiling_dbfs: float = 0.0  # 0 = off
    audx_clamp_to_input: bool = False
    audx_max_attenuation_db: float = 0.0  # 0 = no limit

    @property
    def delay_ms(self) -> int:
        """Constant latency D that the pre-processor adds (the selected model's window; audx: right = look-ahead)."""
        return self.hop_ms + self.right_context_ms + self.compute_budget_ms


# Keys that moved in the YAML (2026-10-30): resemble-enhance-only keys now sit under audio_preprocessing.resemble,
# the digital-zero run length (used by every mode) is audio_preprocessing.zero_run_min_samples. Env names unchanged.
_AUDIO_PREPROC_MOVED_KEYS = {
    "resemble.assets_dir": "assets_dir",
    "resemble.enhancer.nfe": "enhancer.nfe",
    "resemble.enhancer.lambd": "enhancer.lambd",
    "resemble.enhancer.tau": "enhancer.tau",
    "resemble.enhancer.align_regions": "enhancer.align_regions",
    "resemble.enhancer.align_max_ms": "enhancer.align_max_ms",
    "resemble.dither.sigma_lsb": "dither.sigma_lsb",
    "resemble.dither.seed": "dither.seed",
    "zero_run_min_samples": "dither.min_zero_run_samples",
}


def _build_audio_preprocessing(cfg: _Cfg) -> AudioPreprocessSettings:
    p = "audio_preprocessing."

    def k(new: str) -> str:
        """YAML path to read for a moved key: the new one, or the old one (still accepted, with a warning)."""
        old = _AUDIO_PREPROC_MOVED_KEYS[new]
        if cfg._get(p + old) is _MISSING:
            return p + new
        if cfg._get(p + new) is not _MISSING:
            cfg.warnings.append(f"{p}{old} is ignored because {p}{new} is set; remove the old key")
            return p + new
        cfg.warnings.append(f"{p}{old} is deprecated; move it to {p}{new}")
        return p + old

    model = cfg.str(p + "model", "AUDIO_PREPROCESSING_MODEL", "resemble").lower()
    if model not in ("resemble", "audx"):
        raise ValueError(f"{p}model must be 'resemble' or 'audx', got {model!r}")
    audx = model == "audx"
    resemble_mode = cfg.str(p + "resemble.mode", "AUDIO_PREPROCESSING_MODE", "denoiser").lower()
    if resemble_mode not in ("denoiser", "enhancer"):
        hint = "; AudX is selected with model: audx (AUDIO_PREPROCESSING_MODEL=audx)" if resemble_mode == "audx" else ""
        raise ValueError(f"{p}resemble.mode must be 'denoiser' or 'enhancer', got {resemble_mode!r}{hint}")
    mode = "audx" if audx else resemble_mode  # effective mode used by the rest of the engine
    # The layout before 2026-10-30 (one `mode` for all, one shared `window`) is not read any more, only reported.
    for old, new in (("mode", "model (resemble | audx) and resemble.mode"), ("window", "resemble.window / audx.window")):
        if cfg._get(p + old) is not _MISSING:
            cfg.warnings.append(f"{p}{old} is no longer read; use {p}{new}")
    d_left, d_hop, d_right, d_budget = _AUDIO_PREPROC_WINDOW_DEFAULTS[mode]
    w = p + ("audx.window." if audx else "resemble.window.")  # each model has its own window

    def _ms(key: str, env: str, default: int, minimum: int) -> int:
        value = cfg.int(w + key, env, 0)
        if value == 0:
            return default
        if value < minimum:
            raise ValueError(f"{w}{key} must be >= {minimum} (or 0 for the default), got {value}")
        return value

    if audx:
        # stateful: no left context; the right context is the model look-ahead, fixed
        left, right = d_left, d_right
        for key in ("left_context_ms", "right_context_ms"):
            if cfg._get(w + key) is not _MISSING:
                cfg.warnings.append(f"{w}{key} is not used: AudX has no window context (look-ahead fixed at "
                                    f"{AUDX_LOOKAHEAD_MS} ms)")
        hop = _ms("hop_ms", "AUDIO_PREPROCESSING_AUDX_HOP_MS", d_hop, 16)
        budget = _ms("compute_budget_ms", "AUDIO_PREPROCESSING_AUDX_COMPUTE_BUDGET_MS", d_budget, 1)
        crossfade = cfg.int(w + "crossfade_ms", "AUDIO_PREPROCESSING_AUDX_CROSSFADE_MS", 10)
    else:
        left = _ms("left_context_ms", "AUDIO_PREPROCESSING_LEFT_CONTEXT_MS", d_left, 1)
        right = _ms("right_context_ms", "AUDIO_PREPROCESSING_RIGHT_CONTEXT_MS", d_right, 1)
        hop = _ms("hop_ms", "AUDIO_PREPROCESSING_HOP_MS", d_hop, 20)
        budget = _ms("compute_budget_ms", "AUDIO_PREPROCESSING_COMPUTE_BUDGET_MS", d_budget, 1)
        crossfade = cfg.int(w + "crossfade_ms", "AUDIO_PREPROCESSING_CROSSFADE_MS", 40)
    if not (0 <= crossfade <= hop):
        raise ValueError(f"{w}crossfade_ms must be in [0, hop_ms={hop}], got {crossfade}")
    if audx and hop % 16:
        raise ValueError(f"{w}hop_ms must be a multiple of 16 ms (the AudX STFT hop), got {hop}")
    if audx and crossfade >= budget:
        raise ValueError(
            f"{w}crossfade_ms={crossfade} must be < compute_budget_ms={budget} "
            "(the crossfade looks ahead into the budget)"
        )

    sample_percent = cfg.int(p + "sample_percent", "AUDIO_PREPROCESSING_SAMPLE_PERCENT", 100)
    if not (0 <= sample_percent <= 100):
        raise ValueError(f"{p}sample_percent must be in 0-100, got {sample_percent}")
    workers = cfg.int(p + "workers", "AUDIO_PREPROCESSING_WORKERS", 1)
    if workers < 1:
        raise ValueError(f"{p}workers must be >= 1, got {workers}")
    queue_max = cfg.int(p + "queue_max", "AUDIO_PREPROCESSING_QUEUE_MAX", 64)
    if queue_max < 1:
        raise ValueError(f"{p}queue_max must be >= 1, got {queue_max}")
    max_streams = cfg.int(p + "max_streams", "AUDIO_PREPROCESSING_MAX_STREAMS", 0)
    if max_streams < 0:
        raise ValueError(f"{p}max_streams must be >= 0 (0 = unlimited), got {max_streams}")

    nfe = cfg.int(k("resemble.enhancer.nfe"), "AUDIO_PREPROCESSING_ENHANCER_NFE", 32)
    if nfe < 2:
        raise ValueError(f"{p}resemble.enhancer.nfe must be >= 2, got {nfe}")
    lambd = cfg.float(k("resemble.enhancer.lambd"), "AUDIO_PREPROCESSING_ENHANCER_LAMBD", 1.0)
    tau = cfg.float(k("resemble.enhancer.tau"), "AUDIO_PREPROCESSING_ENHANCER_TAU", 0.5)
    if not (0.0 <= lambd <= 1.0) or not (0.0 <= tau <= 1.0):
        raise ValueError(f"{p}resemble.enhancer.lambd and .tau must be in [0, 1], got {lambd}, {tau}")
    align_max_ms = cfg.int(k("resemble.enhancer.align_max_ms"), "AUDIO_PREPROCESSING_ENHANCER_ALIGN_MAX_MS", 40)
    if align_max_ms < 0:
        raise ValueError(f"{p}resemble.enhancer.align_max_ms must be >= 0, got {align_max_ms}")

    sigma = cfg.float(k("resemble.dither.sigma_lsb"), "AUDIO_PREPROCESSING_DITHER_SIGMA_LSB", 3.0)
    if sigma < 0.0:
        raise ValueError(f"{p}resemble.dither.sigma_lsb must be >= 0, got {sigma}")
    min_zero_run = cfg.int(k("zero_run_min_samples"), "AUDIO_PREPROCESSING_DITHER_MIN_ZERO_RUN", 64)
    if min_zero_run < 1:
        raise ValueError(f"{p}zero_run_min_samples must be >= 1, got {min_zero_run}")

    # The crossfade tail of a region is played from the window that ends L after the region,
    # so the zero-run mask there must be final (>= min_zero_run samples before the window end).
    if not audx and (right - crossfade) * 16 < min_zero_run:
        raise ValueError(
            f"{w}right_context_ms={right} must exceed crossfade_ms={crossfade} "
            f"by at least zero_run_min_samples={min_zero_run} samples"
        )
    align_regions = cfg.bool(k("resemble.enhancer.align_regions"), "AUDIO_PREPROCESSING_ENHANCER_ALIGN_REGIONS", True)
    if mode == "enhancer" and align_regions and right - crossfade < align_max_ms:
        raise ValueError(
            f"{w}right_context_ms={right} must exceed crossfade_ms={crossfade} "
            f"by at least resemble.enhancer.align_max_ms={align_max_ms}"
        )

    vad_input = cfg.str(p + "vad_input", "AUDIO_PREPROCESSING_VAD_INPUT", "enhanced").lower()
    if vad_input not in {"enhanced", "raw"}:
        raise ValueError(f"{p}vad_input must be 'enhanced' or 'raw', got {vad_input!r}")

    bad_regions = cfg.int(p + "fallback.consecutive_bad_regions", "AUDIO_PREPROCESSING_FALLBACK_BAD_REGIONS", 3)
    if bad_regions < 1:
        raise ValueError(f"{p}fallback.consecutive_bad_regions must be >= 1, got {bad_regions}")
    bypass_s = cfg.float(p + "fallback.bypass_seconds", "AUDIO_PREPROCESSING_FALLBACK_BYPASS_S", 10.0)
    if bypass_s < 0.0:
        raise ValueError(f"{p}fallback.bypass_seconds must be >= 0 (0 = rest of the stream), got {bypass_s}")

    enabled = cfg.bool(p + "enabled", "AUDIO_PREPROCESSING_ENABLED", False)
    # Only `devices` (YAML list or comma string, env AUDIO_PREPROCESSING_DEVICES); absent = cpu. The single
    # `device` key was removed (2026-10-30): it is not read any more, only reported so a stale config is seen.
    devices = tuple(cfg.devices(p + "devices", "AUDIO_PREPROCESSING_DEVICES"))
    if cfg._get(p + "device") is not _MISSING or os.getenv("AUDIO_PREPROCESSING_DEVICE", "").strip():
        cfg.warnings.append(
            f"{p}device (AUDIO_PREPROCESSING_DEVICE) is no longer read, only {p}devices; devices = {list(devices)}"
        )
    assets_dir = cfg.str(k("resemble.assets_dir"), "AUDIO_PREPROCESSING_ASSETS_DIR", "models/resemble-enhance")
    if enabled and not audx and not assets_dir:
        raise ValueError(f"{p}resemble.assets_dir (AUDIO_PREPROCESSING_ASSETS_DIR) is required when {p}enabled=true")
    audx_checkpoint = cfg.str(
        p + "audx.checkpoint", "AUDIO_PREPROCESSING_AUDX_CHECKPOINT", "models/audx/audx_weights_safe.pth"
    )
    if enabled and audx and not audx_checkpoint:
        raise ValueError(f"{p}audx.checkpoint (AUDIO_PREPROCESSING_AUDX_CHECKPOINT) is required for model audx")
    agc_target = cfg.float(p + "audx.agc_target_dbfs", "AUDIO_PREPROCESSING_AUDX_AGC_TARGET_DBFS", -28.0)
    if not (-60.0 <= agc_target <= -6.0):
        raise ValueError(f"{p}audx.agc_target_dbfs must be in [-60, -6], got {agc_target}")
    agc_max_gain = cfg.float(p + "audx.agc_max_gain_db", "AUDIO_PREPROCESSING_AUDX_AGC_MAX_GAIN_DB", 30.0)
    if not (0.0 <= agc_max_gain <= 60.0):
        raise ValueError(f"{p}audx.agc_max_gain_db must be in [0, 60] (0 = AGC off), got {agc_max_gain}")
    # docs/audx-stateful-design.md section 9: AGC ceiling, clamp of the model output
    agc_ceiling = cfg.float(p + "audx.agc_ceiling_dbfs", "AUDIO_PREPROCESSING_AUDX_AGC_CEILING_DBFS", 0.0)
    if agc_ceiling != 0.0 and not (agc_target < agc_ceiling < 0.0):
        raise ValueError(
            f"{p}audx.agc_ceiling_dbfs must be 0 (off) or between agc_target_dbfs={agc_target} and 0, got {agc_ceiling}"
        )
    clamp_to_input = cfg.bool(p + "audx.clamp_to_input", "AUDIO_PREPROCESSING_AUDX_CLAMP_TO_INPUT", False)
    max_attenuation = cfg.float(p + "audx.max_attenuation_db", "AUDIO_PREPROCESSING_AUDX_MAX_ATTENUATION_DB", 0.0)
    if not (0.0 <= max_attenuation <= 60.0):
        raise ValueError(f"{p}audx.max_attenuation_db must be in [0, 60] (0 = no limit), got {max_attenuation}")

    return AudioPreprocessSettings(
        enabled=enabled,
        model=model,
        mode=mode,
        sample_percent=sample_percent,
        assets_dir=assets_dir,
        devices=devices,
        workers=workers,
        queue_max=queue_max,
        max_streams=max_streams,
        left_context_ms=left,
        hop_ms=hop,
        right_context_ms=right,
        crossfade_ms=crossfade,
        compute_budget_ms=budget,
        enhancer_nfe=nfe,
        enhancer_lambd=lambd,
        enhancer_tau=tau,
        enhancer_align_regions=align_regions,
        enhancer_align_max_ms=align_max_ms,
        dither_sigma_lsb=sigma,
        dither_min_zero_run=min_zero_run,
        dither_seed=cfg.int(k("resemble.dither.seed"), "AUDIO_PREPROCESSING_DITHER_SEED", 0),
        vad_input=vad_input,
        fallback_bad_regions=bad_regions,
        fallback_bypass_s=bypass_s,
        audx_checkpoint=audx_checkpoint,
        audx_agc_target_dbfs=agc_target,
        audx_agc_max_gain_db=agc_max_gain,
        audx_agc_ceiling_dbfs=agc_ceiling,
        audx_clamp_to_input=clamp_to_input,
        audx_max_attenuation_db=max_attenuation,
    )


# ── Trace pcap (docs/trace-pcap-design.md) ─────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TracePcapSettings:
    """Per-stream pcap of the audio received and the audio after AudX; every default describes the OFF state."""

    enabled: bool = False
    dir: str = "logs/pcap"
    max_streams: int = 0  # 0 = every stream
    max_seconds: int = 0  # per stream, 0 = no limit
    min_free_mb: int = 5120


def _build_trace_pcap(cfg: _Cfg) -> TracePcapSettings:
    p = "trace_pcap."
    settings = TracePcapSettings(
        enabled=cfg.bool(p + "enabled", "TRACE_PCAP_ENABLED", False),
        dir=cfg.str(p + "dir", "TRACE_PCAP_DIR", "logs/pcap") or "logs/pcap",
        max_streams=cfg.int(p + "max_streams", "TRACE_PCAP_MAX_STREAMS", 0),
        max_seconds=cfg.int(p + "max_seconds", "TRACE_PCAP_MAX_SECONDS", 0),
        min_free_mb=cfg.int(p + "min_free_mb", "TRACE_PCAP_MIN_FREE_MB", 5120),
    )
    for name in ("max_streams", "max_seconds", "min_free_mb"):
        if getattr(settings, name) < 0:
            raise ValueError(f"trace_pcap.{name} must be >= 0, got {getattr(settings, name)}")
    return settings


# ── Settings dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Settings:
    # Origin — "file:/path" when loaded from YAML, "env" when using env vars only.
    config_source: str

    # Transport / service
    grpc_address: str
    max_rpc_message_mb: int
    response_queue_size: int
    metrics_port: int
    log_level: str
    console_log_level: str
    third_party_log_level: str
    log_file: str
    log_backup_count: int
    pipeline_trace: bool
    pipeline_trace_text: bool
    pipeline_trace_partial: bool
    pipeline_trace_level: str
    suppress_model_init_output: bool
    grpc_keepalive_time_ms: int
    grpc_keepalive_timeout_ms: int
    grpc_keepalive_permit_without_calls: bool
    grpc_min_ping_interval_without_data_ms: int
    grpc_max_ping_strikes: int
    grpc_stop_grace_s: float
    result_queue_maxsize: int
    shutdown_timeout_s: float
    stats_log_interval_s: int
    gpu_monitor_interval_s: int
    # Previously hardcoded tuning constants — now env-configurable
    model_loading_timeout_s: float
    asr_backlog_soft_ms: float
    asr_backlog_hard_ms: float
    short_utterance_ms: float

    # Nemotron ASR
    model_path: str
    device: str
    language: str
    use_language_source: bool
    left_context: int
    lookahead_tokens: int
    model_chunk_ms: int
    warmup_seconds: float
    runtime_queue_size: int
    asr_workers: int
    asr_devices: list[str]
    asr_beam_size: int

    # CPU control for PyTorch/ASR
    cpu_threads: int
    cpu_interop_threads: int

    # GPU memory isolation
    cuda_memory_fraction: float

    # VAD / endpointing
    vad_backend: str
    vad_enabled: bool
    vad_threshold: float
    vad_neg_threshold: float
    vad_energy_db_threshold: float
    vad_min_speech_ms: int
    vad_end_silence_ms: int
    vad_pre_roll_ms: int
    vad_no_start_warn_ms: int
    silero_model_path: str | None
    vad_device: str
    max_utterance_ms: int

    # Sentence segmentation (off | shadow: log/measure only | on: per-segment translate+TTS+playback)
    segmentation_mode: str
    segmentation_stable_n: int
    segmentation_stable_ms: float
    segmentation_punct_stable_n: int
    segmentation_punct_stable_ms: float
    segmentation_min_tokens: int
    segmentation_max_segment_ms: float
    segmentation_comma_after_ms: float
    segmentation_speech_to_text: bool
    segmentation_stt_punct_stable_n: int
    segmentation_stt_punct_stable_ms: float

    # Backpressure
    max_ingress_audio_ms: int
    max_postprocess_pending_per_rpc: int
    max_concurrent_rpcs: int
    max_concurrent_utterances: int
    rpc_max_idle_s: int

    # RTP continuity
    rtp_gap_policy: str
    rtp_max_gap_ms: int
    rtp_out_of_order_policy: str
    rtp_jitter_tolerance_ms: int

    # Translation
    translation_enabled: bool
    translation_model_path: str | None
    translation_tokenizer_path: str | None
    translation_devices: list[str]
    translation_compute_type: str
    translation_inter_threads: int
    translation_intra_threads: int
    translation_queue_size: int
    translation_workers: int
    translation_beam_size: int
    translation_max_decoding_length: int
    translation_batch_size: int
    translation_batch_wait_ms: int
    translation_vi_prefix: str
    translation_en_prefix: str
    translate_emit_source_partials: bool

    # TTS
    tts_enabled: bool
    tts_backend: str
    tts_queue_size: int
    tts_workers: int
    tts_en_model_path: str | None
    tts_en_config_path: str | None
    tts_vi_model_path: str | None
    tts_vi_config_path: str | None
    tts_output_pt: int
    tts_target_sample_rate: int
    tts_max_text_chars: int
    tts_continuation_retry_ms: float
    tts_max_audio_chunk_bytes: int
    tts_length_scale: float
    tts_noise_scale: float
    tts_noise_w_scale: float
    tts_volume: float
    tts_intra_threads: int
    tts_devices: list[str]
    # Magpie TTS backend
    tts_magpie_model_path: str | None
    tts_magpie_codec_model_path: str | None
    tts_speaker_index: int
    tts_cfg_scale: float
    tts_temperature: float
    tts_top_k: int
    tts_magpie_apply_tn: bool

    # Audio pre-processing (default = off; keeps the field optional for direct constructors)
    audio_preprocessing: AudioPreprocessSettings = field(default_factory=AudioPreprocessSettings)
    # Trace pcap (default = off)
    trace_pcap: TracePcapSettings = field(default_factory=TracePcapSettings)
    # Warnings found while reading the config (deprecated YAML paths); main.py logs them
    config_warnings: tuple[str, ...] = ()

    # ── Public constructors ────────────────────────────────────────────────────

    @classmethod
    def from_file_or_env(cls, config_file: str | None = None) -> "Settings":
        """Load config from a YAML file, falling back to environment variables.

        Discovery order (first match wins):
          1. ``config_file`` argument
          2. ``ENGINE_CONFIG_FILE`` environment variable
          3. ``$APP_DIR/etc/nemotron-engine.yaml``
          4. ``./etc/nemotron-engine.yaml`` (relative to cwd)
          5. Environment variables only (fallback, logs a warning after startup)
        """
        path = _discover_config_file(config_file)

        if path is None:
            return cls._build(_Cfg(None, "env"))

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Config file not found: {path!r}. "
                "Check ENGINE_CONFIG_FILE or APP_DIR/etc/ directory."
            )

        data = _load_yaml_file(path)
        return cls._build(_Cfg(data, f"file:{path}"))

    @classmethod
    def from_env(cls) -> "Settings":
        """Load from environment variables only (backward compatibility)."""
        return cls._build(_Cfg(None, "env"))

    # ── Internal builder (all validation lives here) ───────────────────────────

    @classmethod
    def _build(cls, cfg: _Cfg) -> "Settings":  # noqa: C901 (intentionally long)
        # ------------------------------------------------------------------
        # ASR model / lookahead
        # ------------------------------------------------------------------
        lookahead = cfg.int("asr.lookahead_tokens", "MODEL_LOOKAHEAD_TOKENS", 13)
        if lookahead not in _LOOKAHEAD_TO_CHUNK_MS:
            raise ValueError(
                "asr.lookahead_tokens (MODEL_LOOKAHEAD_TOKENS) must be one of "
                f"{sorted(_LOOKAHEAD_TO_CHUNK_MS)}, got {lookahead}"
            )
        model_chunk_ms = _LOOKAHEAD_TO_CHUNK_MS[lookahead]

        model_path = cfg.str("asr.model_path", "MODEL_PATH", "")
        if not model_path:
            raise ValueError(
                "asr.model_path (MODEL_PATH) is required and must point to the "
                "local nemotron-3.5-asr-streaming-0.6b.nemo file"
            )

        # ------------------------------------------------------------------
        # gRPC / transport
        # ------------------------------------------------------------------
        grpc_address = cfg.str("grpc.server_address", "GRPC_SERVER_ADDRESS", "0.0.0.0:50052")
        if not grpc_address.startswith("unix:"):
            _parts = grpc_address.rsplit(":", 1)
            if (
                len(_parts) != 2
                or not _parts[1].isdigit()
                or not (1 <= int(_parts[1]) <= 65535)
            ):
                raise ValueError(
                    f"grpc.server_address (GRPC_SERVER_ADDRESS) must be host:port "
                    f"with port 1–65535, got {grpc_address!r}"
                )

        max_rpc_message_mb = cfg.int("grpc.max_message_mb", "MAX_RPC_MESSAGE_MB", 8)
        if max_rpc_message_mb <= 0:
            raise ValueError(f"grpc.max_message_mb must be > 0, got {max_rpc_message_mb}")

        response_queue_size = cfg.int("grpc.response_queue_size", "RESPONSE_QUEUE_SIZE", 128)
        if response_queue_size < 1:
            raise ValueError(f"grpc.response_queue_size must be >= 1, got {response_queue_size}")

        metrics_port = cfg.int("metrics.port", "METRICS_PORT", 8001)
        if not (1 <= metrics_port <= 65535):
            raise ValueError(f"metrics.port must be in 1–65535, got {metrics_port}")

        grpc_keepalive_time_ms = cfg.int("grpc.keepalive_time_ms", "GRPC_KEEPALIVE_TIME_MS", 60_000)
        if grpc_keepalive_time_ms <= 0:
            raise ValueError(f"grpc.keepalive_time_ms must be > 0, got {grpc_keepalive_time_ms}")

        grpc_keepalive_timeout_ms = cfg.int("grpc.keepalive_timeout_ms", "GRPC_KEEPALIVE_TIMEOUT_MS", 20_000)
        if grpc_keepalive_timeout_ms <= 0:
            raise ValueError(f"grpc.keepalive_timeout_ms must be > 0, got {grpc_keepalive_timeout_ms}")

        grpc_keepalive_permit_without_calls = cfg.bool(
            "grpc.keepalive_permit_without_calls", "GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", True
        )

        grpc_min_ping_interval_without_data_ms = cfg.int(
            "grpc.min_ping_interval_without_data_ms",
            "GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS",
            60_000,
        )
        if grpc_min_ping_interval_without_data_ms < 0:
            raise ValueError(
                f"grpc.min_ping_interval_without_data_ms must be >= 0, "
                f"got {grpc_min_ping_interval_without_data_ms}"
            )

        grpc_max_ping_strikes = cfg.int("grpc.max_ping_strikes", "GRPC_MAX_PING_STRIKES", 2)
        if grpc_max_ping_strikes < 0:
            raise ValueError(f"grpc.max_ping_strikes must be >= 0, got {grpc_max_ping_strikes}")

        grpc_stop_grace_s = cfg.float("grpc.stop_grace_s", "GRPC_STOP_GRACE_S", 5.0)
        if grpc_stop_grace_s < 0.0:
            raise ValueError(f"grpc.stop_grace_s must be >= 0, got {grpc_stop_grace_s}")

        result_queue_maxsize = cfg.int("grpc.result_queue_maxsize", "RESULT_QUEUE_MAXSIZE", 512)
        if result_queue_maxsize < 1:
            raise ValueError(f"grpc.result_queue_maxsize must be >= 1, got {result_queue_maxsize}")

        shutdown_timeout_s = cfg.float("grpc.shutdown_timeout_s", "SHUTDOWN_TIMEOUT_S", 30.0)
        if shutdown_timeout_s <= 0.0:
            raise ValueError(f"grpc.shutdown_timeout_s must be > 0, got {shutdown_timeout_s}")

        stats_log_interval_s = cfg.int("stats.log_interval_s", "STATS_LOG_INTERVAL_S", 10)
        if stats_log_interval_s < 0:
            raise ValueError(f"stats.log_interval_s must be >= 0 (0 to disable), got {stats_log_interval_s}")

        gpu_monitor_interval_s = cfg.int("stats.gpu_monitor_interval_s", "GPU_MONITOR_INTERVAL_S", 5)
        if gpu_monitor_interval_s < 0:
            raise ValueError(f"stats.gpu_monitor_interval_s must be >= 0 (0 to disable), got {gpu_monitor_interval_s}")

        # ------------------------------------------------------------------
        # Tuning constants
        # ------------------------------------------------------------------
        model_loading_timeout_s = cfg.float("tuning.model_loading_timeout_s", "MODEL_LOADING_TIMEOUT_S", 300.0)
        if model_loading_timeout_s <= 0.0:
            raise ValueError(f"tuning.model_loading_timeout_s must be > 0, got {model_loading_timeout_s}")

        asr_backlog_soft_ms = cfg.float("tuning.asr_backlog_soft_ms", "ASR_BACKLOG_SOFT_MS", 320.0)
        if asr_backlog_soft_ms <= 0.0:
            raise ValueError(f"tuning.asr_backlog_soft_ms must be > 0, got {asr_backlog_soft_ms}")

        asr_backlog_hard_ms = cfg.float("tuning.asr_backlog_hard_ms", "ASR_BACKLOG_HARD_MS", 800.0)
        if asr_backlog_hard_ms <= 0.0:
            raise ValueError(f"tuning.asr_backlog_hard_ms must be > 0, got {asr_backlog_hard_ms}")
        if asr_backlog_hard_ms <= asr_backlog_soft_ms:
            raise ValueError(
                f"tuning.asr_backlog_hard_ms ({asr_backlog_hard_ms}) must be > "
                f"tuning.asr_backlog_soft_ms ({asr_backlog_soft_ms})"
            )

        short_utterance_ms = cfg.float("tuning.short_utterance_ms", "SHORT_UTTERANCE_MS", 300.0)
        if short_utterance_ms <= 0.0:
            raise ValueError(f"tuning.short_utterance_ms must be > 0, got {short_utterance_ms}")

        # ------------------------------------------------------------------
        # Log levels
        # ------------------------------------------------------------------
        log_level = cfg.str("logging.level", "LOG_LEVEL", "INFO").upper()
        console_log_level = cfg.str("logging.console_level", "CONSOLE_LOG_LEVEL", "ERROR").upper()
        third_party_log_level = cfg.str("logging.third_party_level", "THIRD_PARTY_LOG_LEVEL", "ERROR").upper()
        pipeline_trace_level = cfg.str(
            "logging.pipeline_trace_level", "LOG_PIPELINE_TRACE_LEVEL", "DEBUG"
        ).upper()
        for _var_name, _var_val in (
            ("logging.level", log_level),
            ("logging.console_level", console_log_level),
            ("logging.third_party_level", third_party_log_level),
            ("logging.pipeline_trace_level", pipeline_trace_level),
        ):
            if not isinstance(_logging.getLevelName(_var_val), int):
                raise ValueError(f"{_var_name}={_var_val!r} is not a valid log level name")

        # ------------------------------------------------------------------
        # ASR workers / beam / queue / CPU
        # ------------------------------------------------------------------
        asr_workers = cfg.int("asr.workers", "ASR_WORKERS", 2)
        if asr_workers < 1:
            raise ValueError(f"asr.workers must be >= 1, got {asr_workers}")

        asr_beam_size = cfg.int("asr.beam_size", "ASR_BEAM_SIZE", 1)
        if asr_beam_size < 1:
            raise ValueError(f"asr.beam_size must be >= 1, got {asr_beam_size}")

        runtime_queue_size = cfg.int("asr.runtime_queue_size", "RUNTIME_QUEUE_SIZE", 256)
        if runtime_queue_size < 1:
            raise ValueError(f"asr.runtime_queue_size must be >= 1, got {runtime_queue_size}")

        warmup_seconds = cfg.float("asr.warmup_seconds", "WARMUP_SECONDS", 2.0)
        if warmup_seconds < 0.0:
            raise ValueError(f"asr.warmup_seconds must be >= 0, got {warmup_seconds}")

        cpu_threads = cfg.int("asr.cpu_threads", "CPU_THREADS", 4)
        if cpu_threads < 1:
            raise ValueError(f"asr.cpu_threads must be >= 1, got {cpu_threads}")

        cpu_interop_threads = cfg.int("asr.cpu_interop_threads", "CPU_INTEROP_THREADS", 1)
        if cpu_interop_threads < 1:
            raise ValueError(f"asr.cpu_interop_threads must be >= 1, got {cpu_interop_threads}")

        # ------------------------------------------------------------------
        # GPU memory
        # ------------------------------------------------------------------
        cuda_memory_fraction = cfg.float("asr.cuda_memory_fraction", "CUDA_MEMORY_FRACTION", 0.0)
        if cuda_memory_fraction != 0.0 and not (0.0 < cuda_memory_fraction <= 1.0):
            raise ValueError("asr.cuda_memory_fraction must be in (0, 1] or 0 to disable")

        # ------------------------------------------------------------------
        # VAD
        # ------------------------------------------------------------------
        vad_backend = cfg.str("vad.backend", "VAD_BACKEND", "energy").lower()
        if vad_backend not in {"energy", "silero"}:
            raise ValueError(f"vad.backend must be 'energy' or 'silero', got {vad_backend!r}")

        vad_threshold = cfg.float("vad.threshold", "VAD_THRESHOLD", 0.5)
        if not (0.0 < vad_threshold < 1.0):
            raise ValueError(f"vad.threshold must be in (0, 1), got {vad_threshold}")

        # Same default as Silero's VADIterator / get_speech_timestamps.
        vad_neg_threshold = cfg.float(
            "vad.neg_threshold", "VAD_NEG_THRESHOLD", max(vad_threshold - 0.15, 0.01)
        )
        if not (0.0 < vad_neg_threshold <= vad_threshold):
            raise ValueError(
                f"vad.neg_threshold must be in (0, vad.threshold={vad_threshold}], "
                f"got {vad_neg_threshold}"
            )

        max_utterance_ms = cfg.int("vad.max_utterance_ms", "MAX_UTTERANCE_MS", 20_000)
        if max_utterance_ms < model_chunk_ms:
            raise ValueError(
                f"vad.max_utterance_ms={max_utterance_ms} must be >= "
                f"model_chunk_ms={model_chunk_ms} "
                f"(set by asr.lookahead_tokens={lookahead})"
            )

        # ------------------------------------------------------------------
        # Sentence segmentation
        # ------------------------------------------------------------------
        # YAML 1.1 reads an unquoted on/off as a boolean.
        raw_mode = cfg._get("segmentation.mode")
        if isinstance(raw_mode, bool):
            segmentation_mode = "on" if raw_mode else "off"
        else:
            segmentation_mode = cfg.str("segmentation.mode", "SEGMENTATION_MODE", "off").lower()
        if segmentation_mode not in {"off", "shadow", "on"}:
            raise ValueError(
                f"segmentation.mode must be 'off', 'shadow' or 'on', got {segmentation_mode!r}"
            )
        segmentation_stable_n = cfg.int("segmentation.stable_n", "SEGMENTATION_STABLE_N", 2)
        if segmentation_stable_n < 1:
            raise ValueError(f"segmentation.stable_n must be >= 1, got {segmentation_stable_n}")
        segmentation_stable_ms = cfg.float("segmentation.stable_ms", "SEGMENTATION_STABLE_MS", 0.0)
        if segmentation_stable_ms < 0:
            raise ValueError(f"segmentation.stable_ms must be >= 0, got {segmentation_stable_ms}")
        segmentation_punct_stable_n = cfg.int(
            "segmentation.punct_stable_n", "SEGMENTATION_PUNCT_STABLE_N", segmentation_stable_n
        )
        if segmentation_punct_stable_n < 1:
            raise ValueError(
                f"segmentation.punct_stable_n must be >= 1, got {segmentation_punct_stable_n}"
            )
        segmentation_punct_stable_ms = cfg.float(
            "segmentation.punct_stable_ms", "SEGMENTATION_PUNCT_STABLE_MS", segmentation_stable_ms
        )
        if segmentation_punct_stable_ms < 0:
            raise ValueError(
                f"segmentation.punct_stable_ms must be >= 0, got {segmentation_punct_stable_ms}"
            )
        segmentation_min_tokens = cfg.int("segmentation.min_tokens", "SEGMENTATION_MIN_TOKENS", 4)
        if segmentation_min_tokens < 1:
            raise ValueError(f"segmentation.min_tokens must be >= 1, got {segmentation_min_tokens}")
        segmentation_max_segment_ms = cfg.float(
            "segmentation.max_segment_ms", "SEGMENTATION_MAX_SEGMENT_MS", 8000.0
        )
        if segmentation_max_segment_ms < 0:
            raise ValueError(
                f"segmentation.max_segment_ms must be >= 0, got {segmentation_max_segment_ms}"
            )
        segmentation_comma_after_ms = cfg.float(
            "segmentation.comma_after_ms", "SEGMENTATION_COMMA_AFTER_MS", 0.0
        )
        if segmentation_comma_after_ms < 0:
            raise ValueError(
                f"segmentation.comma_after_ms must be >= 0, got {segmentation_comma_after_ms}"
            )
        segmentation_speech_to_text = cfg.bool(
            "segmentation.speech_to_text", "SEGMENTATION_SPEECH_TO_TEXT", False
        )
        segmentation_stt_punct_stable_n = cfg.int(
            "segmentation.stt_punct_stable_n",
            "SEGMENTATION_STT_PUNCT_STABLE_N",
            segmentation_punct_stable_n,
        )
        if segmentation_stt_punct_stable_n < 1:
            raise ValueError(
                f"segmentation.stt_punct_stable_n must be >= 1, got {segmentation_stt_punct_stable_n}"
            )
        segmentation_stt_punct_stable_ms = cfg.float(
            "segmentation.stt_punct_stable_ms",
            "SEGMENTATION_STT_PUNCT_STABLE_MS",
            segmentation_punct_stable_ms,
        )
        if segmentation_stt_punct_stable_ms < 0:
            raise ValueError(
                f"segmentation.stt_punct_stable_ms must be >= 0, got {segmentation_stt_punct_stable_ms}"
            )

        # ------------------------------------------------------------------
        # Backpressure
        # ------------------------------------------------------------------
        rpc_max_idle_s =cfg.int("backpressure.rpc_max_idle_s", "RPC_MAX_IDLE_S", 300)
        if rpc_max_idle_s < 0:
            raise ValueError(f"backpressure.rpc_max_idle_s must be >= 0, got {rpc_max_idle_s}")

        max_ingress_audio_ms = cfg.int("backpressure.max_ingress_audio_ms", "MAX_INGRESS_AUDIO_MS", 120_000)
        if max_ingress_audio_ms <= 0:
            raise ValueError(f"backpressure.max_ingress_audio_ms must be > 0, got {max_ingress_audio_ms}")

        max_postprocess_pending_per_rpc = cfg.int(
            "backpressure.max_postprocess_pending_per_rpc",
            "MAX_POSTPROCESS_PENDING_PER_RPC",
            # Counted per segment when segmentation is on, so the default doubles.
            16 if segmentation_mode == "on" else 8,
        )
        if max_postprocess_pending_per_rpc < 1:
            raise ValueError(
                f"backpressure.max_postprocess_pending_per_rpc must be >= 1, "
                f"got {max_postprocess_pending_per_rpc}"
            )

        # ------------------------------------------------------------------
        # RTP
        # ------------------------------------------------------------------
        rtp_gap_policy = cfg.str("rtp.gap_policy", "RTP_GAP_POLICY", "insert_silence").lower()
        if rtp_gap_policy not in {"warn", "insert_silence", "error"}:
            raise ValueError("rtp.gap_policy must be warn|insert_silence|error")

        rtp_ooo_policy = cfg.str("rtp.out_of_order_policy", "RTP_OUT_OF_ORDER_POLICY", "drop").lower()
        if rtp_ooo_policy not in {"drop", "error", "accept"}:
            raise ValueError("rtp.out_of_order_policy must be drop|error|accept")

        # ------------------------------------------------------------------
        # Translation
        # ------------------------------------------------------------------
        translation_enabled = cfg.bool("translation.enabled", "TRANSLATION_ENABLED", True)
        translation_model_path = (
            cfg.str("translation.model_path", "TRANSLATION_MODEL_PATH", "models/envit5_ctranslate2") or None
        )
        translation_tokenizer_path = (
            cfg.str("translation.tokenizer_path", "TRANSLATION_TOKENIZER_PATH", "models/envit5_ctranslate2") or None
        )

        if translation_enabled:
            if not translation_model_path:
                raise ValueError(
                    "translation.model_path (TRANSLATION_MODEL_PATH) is required "
                    "when translation.enabled=true"
                )
            if not translation_tokenizer_path:
                raise ValueError(
                    "translation.tokenizer_path (TRANSLATION_TOKENIZER_PATH) is required "
                    "when translation.enabled=true"
                )

        translation_compute_type = cfg.str("translation.compute_type", "TRANSLATION_COMPUTE_TYPE", "int8")
        if translation_compute_type not in _VALID_CT2_COMPUTE:
            raise ValueError(
                f"translation.compute_type must be one of {sorted(_VALID_CT2_COMPUTE)}, "
                f"got {translation_compute_type!r}"
            )

        translation_workers = cfg.int("translation.workers", "TRANSLATION_WORKERS", 2)
        if translation_workers < 1:
            raise ValueError(f"translation.workers must be >= 1, got {translation_workers}")

        translation_beam_size = cfg.int("translation.beam_size", "TRANSLATION_BEAM_SIZE", 1)
        if translation_beam_size < 1:
            raise ValueError(f"translation.beam_size must be >= 1, got {translation_beam_size}")

        translation_queue_size = cfg.int("translation.queue_size", "TRANSLATION_QUEUE_SIZE", 64)
        if translation_queue_size < 1:
            raise ValueError(f"translation.queue_size must be >= 1, got {translation_queue_size}")

        translation_max_decoding_length = cfg.int(
            "translation.max_decoding_length", "TRANSLATION_MAX_DECODING_LENGTH", 256
        )
        if translation_max_decoding_length < 1:
            raise ValueError(
                f"translation.max_decoding_length must be >= 1, got {translation_max_decoding_length}"
            )

        translation_batch_size = cfg.int("translation.batch_size", "TRANSLATION_BATCH_SIZE", 8)
        if translation_batch_size < 1:
            raise ValueError(f"translation.batch_size must be >= 1, got {translation_batch_size}")

        translation_batch_wait_ms = cfg.int("translation.batch_wait_ms", "TRANSLATION_BATCH_WAIT_MS", 3)
        if translation_batch_wait_ms < 0:
            raise ValueError(f"translation.batch_wait_ms must be >= 0, got {translation_batch_wait_ms}")

        translation_inter_threads = cfg.int("translation.inter_threads", "TRANSLATION_INTER_THREADS", 1)
        if translation_inter_threads < 1:
            raise ValueError(f"translation.inter_threads must be >= 1, got {translation_inter_threads}")

        translation_intra_threads = cfg.int("translation.intra_threads", "TRANSLATION_INTRA_THREADS", 2)
        if translation_intra_threads < 1:
            raise ValueError(f"translation.intra_threads must be >= 1, got {translation_intra_threads}")

        # ------------------------------------------------------------------
        # TTS
        # ------------------------------------------------------------------
        tts_enabled = cfg.bool("tts.enabled", "TTS_ENABLED", True)
        tts_en_model_path = (
            cfg.str("tts.piper.en_model_path", "TTS_EN_MODEL_PATH", "models/piper/en_US-lessac-medium.onnx") or None
        )
        tts_en_config_path = (
            cfg.str("tts.piper.en_config_path", "TTS_EN_CONFIG_PATH", "models/piper/en_US-lessac-medium.onnx.json") or None
        )
        tts_vi_model_path = (
            cfg.str("tts.piper.vi_model_path", "TTS_VI_MODEL_PATH", "models/piper/vi_VN-vais1000-medium.onnx") or None
        )
        tts_vi_config_path = (
            cfg.str("tts.piper.vi_config_path", "TTS_VI_CONFIG_PATH", "models/piper/vi_VN-vais1000-medium.onnx.json") or None
        )

        tts_output_pt = cfg.int("tts.output_pt", "TTS_OUTPUT_PT", 0)

        tts_backend = cfg.str("tts.backend", "TTS_BACKEND", "piper").lower()
        if tts_backend not in {"piper", "magpie"}:
            raise ValueError("tts.backend must be 'piper' or 'magpie'")

        tts_magpie_model_path = cfg.str("tts.magpie.model_path", "TTS_MAGPIE_MODEL_PATH", "") or None
        tts_magpie_codec_model_path = cfg.str("tts.magpie.codec_model_path", "TTS_MAGPIE_CODEC_MODEL_PATH", "") or None
        tts_speaker_index = cfg.int("tts.magpie.speaker_index", "TTS_SPEAKER_INDEX", 0)
        tts_cfg_scale = cfg.float("tts.magpie.cfg_scale", "TTS_CFG_SCALE", 2.5)
        tts_temperature = cfg.float("tts.magpie.temperature", "TTS_TEMPERATURE", 0.6)
        tts_top_k = cfg.int("tts.magpie.top_k", "TTS_TOP_K", 80)
        tts_magpie_apply_tn = cfg.bool("tts.magpie.apply_tn", "TTS_MAGPIE_APPLY_TN", False)

        if tts_enabled:
            if tts_output_pt not in (0, 8):
                raise ValueError("tts.output_pt must be 0 (PCMU) or 8 (PCMA)")
            if tts_backend == "magpie":
                if not tts_magpie_model_path:
                    raise ValueError(
                        "tts.magpie.model_path (TTS_MAGPIE_MODEL_PATH) is required "
                        "when tts.backend=magpie"
                    )
                if not (0 <= tts_speaker_index <= 4):
                    raise ValueError(
                        f"tts.magpie.speaker_index must be 0–4, got {tts_speaker_index}"
                    )
            else:  # piper
                if not tts_en_model_path:
                    raise ValueError(
                        "tts.piper.en_model_path (TTS_EN_MODEL_PATH) is required "
                        "when tts.enabled=true and tts.backend=piper"
                    )
                if not tts_vi_model_path:
                    raise ValueError(
                        "tts.piper.vi_model_path (TTS_VI_MODEL_PATH) is required "
                        "when tts.enabled=true and tts.backend=piper"
                    )
                if not tts_en_config_path:
                    raise ValueError(
                        "tts.piper.en_config_path (TTS_EN_CONFIG_PATH) is required "
                        "when tts.enabled=true and tts.backend=piper"
                    )
                if not tts_vi_config_path:
                    raise ValueError(
                        "tts.piper.vi_config_path (TTS_VI_CONFIG_PATH) is required "
                        "when tts.enabled=true and tts.backend=piper"
                    )

        tts_workers = cfg.int("tts.workers", "TTS_WORKERS", 2)
        if tts_workers < 1:
            raise ValueError(f"tts.workers must be >= 1, got {tts_workers}")

        tts_queue_size = cfg.int("tts.queue_size", "TTS_QUEUE_SIZE", 32)
        if tts_queue_size < 1:
            raise ValueError(f"tts.queue_size must be >= 1, got {tts_queue_size}")

        tts_target_sample_rate = cfg.int("tts.target_sample_rate", "TTS_TARGET_SAMPLE_RATE", 8_000)
        if tts_target_sample_rate <= 0:
            raise ValueError(f"tts.target_sample_rate must be > 0, got {tts_target_sample_rate}")

        tts_max_text_chars = cfg.int("tts.max_text_chars", "TTS_MAX_TEXT_CHARS", 1_000)
        if tts_max_text_chars < 1:
            raise ValueError(f"tts.max_text_chars must be >= 1, got {tts_max_text_chars}")

        tts_continuation_retry_ms = cfg.float(
            "tts.continuation_retry_ms", "TTS_CONTINUATION_RETRY_MS", 2_000.0
        )
        if tts_continuation_retry_ms < 0:
            raise ValueError(
                f"tts.continuation_retry_ms must be >= 0, got {tts_continuation_retry_ms}"
            )

        tts_max_audio_chunk_bytes = cfg.int(
            "tts.max_audio_chunk_bytes", "TTS_MAX_AUDIO_CHUNK_BYTES", 0
        )
        if tts_max_audio_chunk_bytes < 0 or tts_max_audio_chunk_bytes % 160 != 0:
            raise ValueError(
                "tts.max_audio_chunk_bytes must be 0 (disabled) or a multiple of 160 "
                f"(one 20 ms G.711 frame), got {tts_max_audio_chunk_bytes}"
            )

        tts_intra_threads = cfg.int("tts.intra_threads", "TTS_INTRA_THREADS", 2)
        if tts_intra_threads < 1:
            raise ValueError(f"tts.intra_threads must be >= 1, got {tts_intra_threads}")

        tts_volume = cfg.float("tts.volume", "TTS_VOLUME", 1.0)
        if tts_volume <= 0.0:
            raise ValueError(f"tts.volume must be > 0, got {tts_volume}")

        tts_length_scale = cfg.float("tts.length_scale", "TTS_LENGTH_SCALE", 1.0)
        if tts_length_scale <= 0.0:
            raise ValueError(f"tts.length_scale must be > 0, got {tts_length_scale}")

        tts_noise_scale = cfg.float("tts.noise_scale", "TTS_NOISE_SCALE", 0.667)
        if tts_noise_scale <= 0.0:
            raise ValueError(f"tts.noise_scale must be > 0, got {tts_noise_scale}")

        tts_noise_w_scale = cfg.float("tts.noise_w_scale", "TTS_NOISE_W_SCALE", 0.8)
        if tts_noise_w_scale <= 0.0:
            raise ValueError(f"tts.noise_w_scale must be > 0, got {tts_noise_w_scale}")

        if tts_cfg_scale <= 0.0:
            raise ValueError(f"tts.magpie.cfg_scale must be > 0, got {tts_cfg_scale}")
        if tts_temperature <= 0.0:
            raise ValueError(f"tts.magpie.temperature must be > 0, got {tts_temperature}")
        if tts_top_k < 1:
            raise ValueError(f"tts.magpie.top_k must be >= 1, got {tts_top_k}")

        # ------------------------------------------------------------------
        # Device strings (read after worker counts so defaults are consistent)
        # ------------------------------------------------------------------
        asr_device = cfg.str("asr.device", "ASR_DEVICE", "cpu")
        translation_device_default = cfg.str("translation.device", "TRANSLATION_DEVICE", "cpu")
        tts_device_default = cfg.str("tts.device", "TTS_DEVICE", "cpu")

        audio_preprocessing = _build_audio_preprocessing(cfg)
        return cls(
            config_source=cfg.source,
            grpc_address=grpc_address,
            max_rpc_message_mb=max_rpc_message_mb,
            response_queue_size=response_queue_size,
            metrics_port=metrics_port,
            log_level=log_level,
            console_log_level=console_log_level,
            third_party_log_level=third_party_log_level,
            log_file=cfg.str("logging.file", "LOG_FILE", "logs/nemotron-engine.log"),
            log_backup_count=cfg.int("logging.backup_count", "LOG_BACKUP_COUNT", 10),
            pipeline_trace=cfg.bool("logging.pipeline_trace", "LOG_PIPELINE_TRACE", True),
            pipeline_trace_text=cfg.bool(
                "logging.pipeline_trace_text", "LOG_PIPELINE_TRACE_TEXT", True
            ),
            pipeline_trace_partial=cfg.bool(
                "logging.pipeline_trace_partial", "LOG_PIPELINE_TRACE_PARTIAL", False
            ),
            pipeline_trace_level=pipeline_trace_level,
            suppress_model_init_output=cfg.bool(
                "logging.suppress_model_init_output", "SUPPRESS_MODEL_INIT_OUTPUT", True
            ),
            grpc_keepalive_time_ms=grpc_keepalive_time_ms,
            grpc_keepalive_timeout_ms=grpc_keepalive_timeout_ms,
            grpc_keepalive_permit_without_calls=grpc_keepalive_permit_without_calls,
            grpc_min_ping_interval_without_data_ms=grpc_min_ping_interval_without_data_ms,
            grpc_max_ping_strikes=grpc_max_ping_strikes,
            grpc_stop_grace_s=grpc_stop_grace_s,
            result_queue_maxsize=result_queue_maxsize,
            shutdown_timeout_s=shutdown_timeout_s,
            stats_log_interval_s=stats_log_interval_s,
            gpu_monitor_interval_s=gpu_monitor_interval_s,
            model_loading_timeout_s=model_loading_timeout_s,
            asr_backlog_soft_ms=asr_backlog_soft_ms,
            asr_backlog_hard_ms=asr_backlog_hard_ms,
            short_utterance_ms=short_utterance_ms,
            model_path=model_path,
            device=asr_device,
            asr_devices=cfg.devices("asr.devices", "ASR_DEVICES", "asr.device", "ASR_DEVICE"),
            language=cfg.str("asr.language", "ASR_LANGUAGE", "vi-VN"),
            use_language_source=cfg.bool(
                "asr.use_language_source", "ASR_USE_LANGUAGE_SOURCE", False
            ),
            left_context=cfg.int("asr.left_context", "MODEL_LEFT_CONTEXT", 56),
            lookahead_tokens=lookahead,
            model_chunk_ms=model_chunk_ms,
            warmup_seconds=warmup_seconds,
            runtime_queue_size=runtime_queue_size,
            asr_workers=asr_workers,
            asr_beam_size=asr_beam_size,
            cpu_threads=cpu_threads,
            cpu_interop_threads=cpu_interop_threads,
            cuda_memory_fraction=cuda_memory_fraction,
            vad_backend=vad_backend,
            vad_enabled=cfg.bool("vad.enabled", "VAD_ENABLED", True),
            vad_threshold=vad_threshold,
            vad_neg_threshold=vad_neg_threshold,
            vad_energy_db_threshold=cfg.float("vad.energy_db_threshold", "VAD_ENERGY_DB_THRESHOLD", -42.0),
            vad_min_speech_ms=cfg.int("vad.min_speech_ms", "VAD_MIN_SPEECH_MS", 250),
            vad_end_silence_ms=cfg.int("vad.end_silence_ms", "VAD_END_SILENCE_MS", 600),
            vad_pre_roll_ms=cfg.int("vad.pre_roll_ms", "VAD_PRE_ROLL_MS", 200),
            vad_no_start_warn_ms=cfg.int("vad.no_start_warn_ms", "VAD_NO_START_WARN_MS", 5_000),
            silero_model_path=cfg.str("vad.silero_model_path", "SILERO_MODEL_PATH") or None,
            vad_device=cfg.str("vad.device", "VAD_DEVICE", "cpu"),
            max_utterance_ms=max_utterance_ms,
            segmentation_mode=segmentation_mode,
            segmentation_stable_n=segmentation_stable_n,
            segmentation_stable_ms=segmentation_stable_ms,
            segmentation_punct_stable_n=segmentation_punct_stable_n,
            segmentation_punct_stable_ms=segmentation_punct_stable_ms,
            segmentation_min_tokens=segmentation_min_tokens,
            segmentation_max_segment_ms=segmentation_max_segment_ms,
            segmentation_comma_after_ms=segmentation_comma_after_ms,
            segmentation_speech_to_text=segmentation_speech_to_text,
            segmentation_stt_punct_stable_n=segmentation_stt_punct_stable_n,
            segmentation_stt_punct_stable_ms=segmentation_stt_punct_stable_ms,
            max_ingress_audio_ms=max_ingress_audio_ms,
            max_postprocess_pending_per_rpc=max_postprocess_pending_per_rpc,
            max_concurrent_rpcs=cfg.int("backpressure.max_concurrent_rpcs", "MAX_CONCURRENT_RPCS", 0),
            max_concurrent_utterances=cfg.int("backpressure.max_concurrent_utterances", "MAX_CONCURRENT_UTTERANCES", 0),
            rpc_max_idle_s=rpc_max_idle_s,
            rtp_gap_policy=rtp_gap_policy,
            rtp_max_gap_ms=cfg.int("rtp.max_gap_ms", "RTP_MAX_GAP_MS", 200),
            rtp_out_of_order_policy=rtp_ooo_policy,
            rtp_jitter_tolerance_ms=cfg.int("rtp.jitter_tolerance_ms", "RTP_JITTER_TOLERANCE_MS", 20),
            translation_enabled=translation_enabled,
            translation_model_path=translation_model_path,
            translation_tokenizer_path=translation_tokenizer_path,
            translation_devices=cfg.devices(
                "translation.devices", "TRANSLATION_DEVICES",
                "translation.device", "TRANSLATION_DEVICE",
            ),
            translation_compute_type=translation_compute_type,
            translation_inter_threads=translation_inter_threads,
            translation_intra_threads=translation_intra_threads,
            translation_queue_size=translation_queue_size,
            translation_workers=translation_workers,
            translation_beam_size=translation_beam_size,
            translation_max_decoding_length=translation_max_decoding_length,
            translation_batch_size=translation_batch_size,
            translation_batch_wait_ms=translation_batch_wait_ms,
            translation_vi_prefix=cfg.str("translation.vi_prefix", "TRANSLATION_VI_PREFIX", "vi: ", strip=False),
            translation_en_prefix=cfg.str("translation.en_prefix", "TRANSLATION_EN_PREFIX", "en: ", strip=False),
            translate_emit_source_partials=cfg.bool(
                "translation.emit_source_partials", "TRANSLATE_EMIT_SOURCE_PARTIALS", False
            ),
            tts_enabled=tts_enabled,
            tts_backend=tts_backend,
            tts_queue_size=tts_queue_size,
            tts_workers=tts_workers,
            tts_en_model_path=tts_en_model_path,
            tts_en_config_path=tts_en_config_path,
            tts_vi_model_path=tts_vi_model_path,
            tts_vi_config_path=tts_vi_config_path,
            tts_output_pt=tts_output_pt,
            tts_target_sample_rate=tts_target_sample_rate,
            tts_max_text_chars=tts_max_text_chars,
            tts_continuation_retry_ms=tts_continuation_retry_ms,
            tts_max_audio_chunk_bytes=tts_max_audio_chunk_bytes,
            tts_length_scale=tts_length_scale,
            tts_noise_scale=tts_noise_scale,
            tts_noise_w_scale=tts_noise_w_scale,
            tts_volume=tts_volume,
            tts_intra_threads=tts_intra_threads,
            tts_devices=cfg.devices("tts.devices", "TTS_DEVICES", "tts.device", "TTS_DEVICE"),
            tts_magpie_model_path=tts_magpie_model_path,
            tts_magpie_codec_model_path=tts_magpie_codec_model_path,
            tts_speaker_index=tts_speaker_index,
            tts_cfg_scale=tts_cfg_scale,
            tts_temperature=tts_temperature,
            tts_top_k=tts_top_k,
            tts_magpie_apply_tn=tts_magpie_apply_tn,
            audio_preprocessing=audio_preprocessing,
            trace_pcap=_build_trace_pcap(cfg),
            config_warnings=tuple(cfg.warnings),
        )
