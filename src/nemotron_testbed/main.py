from __future__ import annotations

import asyncio
import logging
import os
import signal

import grpc

from .config import Settings
from .coordinator import EngineCoordinator
from .gpu_monitor import GpuMonitor
from .logging_config import configure_logging, quiet_nemo_logging, quiet_transformers_logging
from .metrics import MetricsServer
from .pb import speech_pb2_grpc
from .service import SpeechStreamService
from .stats_logger import StatsLogger

logger = logging.getLogger(__name__)


def grpc_server_options(settings: Settings) -> tuple[tuple[str, int], ...]:
    max_bytes = settings.max_rpc_message_mb * 1024 * 1024
    return (
        ("grpc.max_receive_message_length", max_bytes),
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.keepalive_time_ms", settings.grpc_keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", settings.grpc_keepalive_timeout_ms),
        (
            "grpc.keepalive_permit_without_calls",
            int(settings.grpc_keepalive_permit_without_calls),
        ),
        (
            "grpc.http2.min_ping_interval_without_data_ms",
            settings.grpc_min_ping_interval_without_data_ms,
        ),
        ("grpc.http2.max_ping_strikes", settings.grpc_max_ping_strikes),
    )


def configure_gpu_memory(settings: Settings) -> None:
    fraction = settings.cuda_memory_fraction
    if fraction <= 0.0:
        return
    import torch

    device_indices: set[int] = set()
    for device_str in settings.asr_devices:
        if device_str.startswith("cuda"):
            parts = device_str.split(":")
            device_indices.add(int(parts[1]) if len(parts) > 1 else 0)

    if settings.translation_enabled:
        for device_str in settings.translation_devices:
            if device_str.startswith("cuda"):
                parts = device_str.split(":")
                device_indices.add(int(parts[1]) if len(parts) > 1 else 0)

    if settings.tts_enabled:
        for device_str in settings.tts_devices:
            if device_str.startswith("cuda"):
                parts = device_str.split(":")
                device_indices.add(int(parts[1]) if len(parts) > 1 else 0)

    for idx in sorted(device_indices):
        torch.cuda.set_per_process_memory_fraction(fraction, idx)
        logger.info(
            "CUDA device %s: memory fraction capped at %.2f", idx, fraction
        )


def configure_cpu_threads(settings: Settings) -> None:
    import torch

    if any(d.startswith("cpu") for d in settings.asr_devices):
        torch.set_num_threads(settings.cpu_threads)
        torch.set_num_interop_threads(settings.cpu_interop_threads)
        # Keep native BLAS/OpenMP pools bounded. Environment values set by the
        # container still take precedence if explicitly supplied.
        os.environ.setdefault("OMP_NUM_THREADS", str(settings.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(settings.cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(settings.cpu_threads))
        logger.info(
            "Configured PyTorch CPU threads: intra=%s inter=%s",
            settings.cpu_threads,
            settings.cpu_interop_threads,
        )
        if settings.asr_workers > 1:
            logger.warning(
                "ASR_WORKERS=%s on CPU loads multiple Nemotron replicas; "
                "benchmark CPU/RAM contention before production",
                settings.asr_workers,
            )


async def serve() -> None:
    settings = Settings.from_file_or_env()
    configure_logging(
        settings.log_level,
        settings.log_file,
        settings.log_backup_count,
        settings.console_log_level,
        settings.third_party_log_level,
        settings.pipeline_trace_level if settings.pipeline_trace else None,
    )
    if settings.config_source == "env":
        logger.warning(
            "No config file found — using environment variables. "
            "Place etc/nemotron-engine.yaml under APP_DIR or set ENGINE_CONFIG_FILE."
        )
    else:
        logger.info("Configuration loaded from: %s", settings.config_source)
    for warning in settings.config_warnings:
        logger.warning("CONFIG_DEPRECATED: %s", warning)
    configure_cpu_threads(settings)
    configure_gpu_memory(settings)
    MetricsServer(settings.metrics_port).start()

    coordinator = EngineCoordinator(settings)
    try:
        await asyncio.wait_for(
            coordinator.start(), timeout=settings.model_loading_timeout_s
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Model loading timed out after {settings.model_loading_timeout_s:.0f} s "
            "— check model files and GPU availability"
        )
    quiet_nemo_logging(settings.third_party_log_level)
    quiet_transformers_logging(settings.third_party_log_level)

    stats_logger = StatsLogger(settings.stats_log_interval_s)
    await stats_logger.start()

    gpu_monitor = GpuMonitor(interval_s=settings.gpu_monitor_interval_s)
    await gpu_monitor.start()

    try:
        server = grpc.aio.server(options=grpc_server_options(settings))
        speech_pb2_grpc.add_SpeechStreamServicer_to_server(
            SpeechStreamService(coordinator), server
        )
        bound_port = server.add_insecure_port(settings.grpc_address)
        if bound_port == 0:
            raise RuntimeError(f"Failed to bind gRPC address {settings.grpc_address}")

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:
                pass

        await server.start()
        logger.info(
            "gRPC server ready on %s; model_chunk_ms=%s device=%s "
            "asr_workers=%s translate=%s translation_workers=%s tts=%s tts_workers=%s",
            settings.grpc_address,
            settings.model_chunk_ms,
            settings.device,
            settings.asr_workers,
            settings.translation_enabled,
            settings.translation_workers if settings.translation_enabled else 0,
            settings.tts_enabled,
            settings.tts_workers if settings.tts_enabled else 0,
        )

        await stop_event.wait()
        logger.info("Shutdown requested")
        await server.stop(grace=settings.grpc_stop_grace_s)
    finally:
        await stats_logger.stop()
        await gpu_monitor.stop()
        await coordinator.shutdown()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
