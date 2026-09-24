from __future__ import annotations

import logging as _logging
import os
import re
from dataclasses import dataclass
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

_BCP47_RE = re.compile(r'^[a-z]{2,3}(-[A-Za-z0-9]{2,8})+$')

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

    def _get(self, path: str) -> Any:
        """Navigate a dot-separated key path. Returns _MISSING when any key absent."""
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return _MISSING
            node = node[key]
        return node  # may be None for explicit YAML null

    def str(self, yaml_path: str, env_name: str, default: str = "") -> str:
        val = self._get(yaml_path)
        if val is not _MISSING:
            return default if val is None else str(val).strip()
        return _env_str(env_name, default)

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
        fallback_yaml: str,
        fallback_env: str,
    ) -> list[str]:
        """Read a device list from YAML (list or comma-string) or env var."""
        fallback = self.str(fallback_yaml, fallback_env, "cpu")
        val = self._get(yaml_path)
        if val is not _MISSING:
            if val is None:
                return [fallback]
            if isinstance(val, list):
                devices = [str(d).strip() for d in val if d is not None and str(d).strip()]
                return devices if devices else [fallback]
            devices = [d.strip() for d in str(val).split(",") if d.strip()]
            return devices if devices else [fallback]
        return _env_devices(env_name, _env_str(fallback_env, "cpu"))


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
    log_max_bytes: int
    log_backup_count: int
    suppress_model_init_output: bool
    grpc_keepalive_time_ms: int
    grpc_keepalive_timeout_ms: int
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
    vad_energy_db_threshold: float
    vad_min_speech_ms: int
    vad_end_silence_ms: int
    vad_pre_roll_ms: int
    silero_model_path: str | None
    vad_device: str
    max_utterance_ms: int

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
    translate_vi_target: str
    translate_en_target: str
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

        grpc_keepalive_time_ms = cfg.int("grpc.keepalive_time_ms", "GRPC_KEEPALIVE_TIME_MS", 30_000)
        if grpc_keepalive_time_ms <= 0:
            raise ValueError(f"grpc.keepalive_time_ms must be > 0, got {grpc_keepalive_time_ms}")

        grpc_keepalive_timeout_ms = cfg.int("grpc.keepalive_timeout_ms", "GRPC_KEEPALIVE_TIMEOUT_MS", 10_000)
        if grpc_keepalive_timeout_ms <= 0:
            raise ValueError(f"grpc.keepalive_timeout_ms must be > 0, got {grpc_keepalive_timeout_ms}")

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
        for _var_name, _var_val in (
            ("logging.level", log_level),
            ("logging.console_level", console_log_level),
            ("logging.third_party_level", third_party_log_level),
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

        max_utterance_ms = cfg.int("vad.max_utterance_ms", "MAX_UTTERANCE_MS", 20_000)
        if max_utterance_ms < model_chunk_ms:
            raise ValueError(
                f"vad.max_utterance_ms={max_utterance_ms} must be >= "
                f"model_chunk_ms={model_chunk_ms} "
                f"(set by asr.lookahead_tokens={lookahead})"
            )

        # ------------------------------------------------------------------
        # Backpressure
        # ------------------------------------------------------------------
        rpc_max_idle_s = cfg.int("backpressure.rpc_max_idle_s", "RPC_MAX_IDLE_S", 300)
        if rpc_max_idle_s < 0:
            raise ValueError(f"backpressure.rpc_max_idle_s must be >= 0, got {rpc_max_idle_s}")

        max_ingress_audio_ms = cfg.int("backpressure.max_ingress_audio_ms", "MAX_INGRESS_AUDIO_MS", 120_000)
        if max_ingress_audio_ms <= 0:
            raise ValueError(f"backpressure.max_ingress_audio_ms must be > 0, got {max_ingress_audio_ms}")

        max_postprocess_pending_per_rpc = cfg.int(
            "backpressure.max_postprocess_pending_per_rpc",
            "MAX_POSTPROCESS_PENDING_PER_RPC",
            8,
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

        translate_vi_target = cfg.str("translation.vi_target", "TRANSLATE_VI_TARGET", "en-US")
        if not _BCP47_RE.match(translate_vi_target):
            raise ValueError(
                f"translation.vi_target must be a BCP-47 tag (e.g. en-US), "
                f"got {translate_vi_target!r}"
            )

        translate_en_target = cfg.str("translation.en_target", "TRANSLATE_EN_TARGET", "vi-VN")
        if not _BCP47_RE.match(translate_en_target):
            raise ValueError(
                f"translation.en_target must be a BCP-47 tag (e.g. vi-VN), "
                f"got {translate_en_target!r}"
            )

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
            log_max_bytes=cfg.int("logging.max_bytes", "LOG_MAX_BYTES", 100 * 1024 * 1024),
            log_backup_count=cfg.int("logging.backup_count", "LOG_BACKUP_COUNT", 10),
            suppress_model_init_output=cfg.bool(
                "logging.suppress_model_init_output", "SUPPRESS_MODEL_INIT_OUTPUT", True
            ),
            grpc_keepalive_time_ms=grpc_keepalive_time_ms,
            grpc_keepalive_timeout_ms=grpc_keepalive_timeout_ms,
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
            vad_energy_db_threshold=cfg.float("vad.energy_db_threshold", "VAD_ENERGY_DB_THRESHOLD", -42.0),
            vad_min_speech_ms=cfg.int("vad.min_speech_ms", "VAD_MIN_SPEECH_MS", 250),
            vad_end_silence_ms=cfg.int("vad.end_silence_ms", "VAD_END_SILENCE_MS", 600),
            vad_pre_roll_ms=cfg.int("vad.pre_roll_ms", "VAD_PRE_ROLL_MS", 200),
            silero_model_path=cfg.str("vad.silero_model_path", "SILERO_MODEL_PATH") or None,
            vad_device=cfg.str("vad.device", "VAD_DEVICE", "cpu"),
            max_utterance_ms=max_utterance_ms,
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
            translation_vi_prefix=cfg.str("translation.vi_prefix", "TRANSLATION_VI_PREFIX", "vi: "),
            translation_en_prefix=cfg.str("translation.en_prefix", "TRANSLATION_EN_PREFIX", "en: "),
            translate_vi_target=translate_vi_target,
            translate_en_target=translate_en_target,
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
        )
