from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .metrics import GPU_MEMORY_TOTAL_BYTES, GPU_MEMORY_USED_BYTES, GPU_UTILIZATION

logger = logging.getLogger(__name__)


def _build_torch_poller() -> tuple[int, Callable[[], None]] | None:
    """Return (device_count, poll_fn) using torch.cuda, or None if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        count = torch.cuda.device_count()
        if count == 0:
            return None

        def _poll() -> None:
            for i in range(count):
                label = str(i)
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    GPU_MEMORY_USED_BYTES.labels(device_id=label).set(total - free)
                    GPU_MEMORY_TOTAL_BYTES.labels(device_id=label).set(total)
                except Exception:
                    pass
                try:
                    util = torch.cuda.utilization(i)
                    GPU_UTILIZATION.labels(device_id=label).set(util)
                except Exception:
                    # NVML unavailable in some containers; register -1 so the
                    # label exists and memory metrics still appear in stats.
                    GPU_UTILIZATION.labels(device_id=label).set(-1)

        return count, _poll
    except Exception:
        return None


def _build_nvml_poller() -> tuple[int, Callable[[], None], Callable[[], None]] | None:
    """Return (device_count, poll_fn) using pynvml, or None if unavailable.

    Works with both the official nvidia-ml-py package and the legacy pynvml
    package — both expose the same ``import pynvml`` interface.
    """
    try:
        import pynvml  # type: ignore  # nvidia-ml-py or legacy pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

        def _poll() -> None:
            for i, handle in enumerate(handles):
                label = str(i)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    GPU_UTILIZATION.labels(device_id=label).set(util.gpu)
                except Exception:
                    pass
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    GPU_MEMORY_USED_BYTES.labels(device_id=label).set(mem.used)
                    GPU_MEMORY_TOTAL_BYTES.labels(device_id=label).set(mem.total)
                except Exception:
                    pass

        def _shutdown() -> None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        return count, _poll, _shutdown
    except Exception:
        return None


class GpuMonitor:
    """Asyncio task that polls GPU utilization and memory at a fixed interval.

    Uses torch.cuda (primary) or pynvml (fallback) — no extra dependencies
    beyond what torch already requires. Silently skips if no CUDA device found.

    Metrics updated (labeled by device_id = "0", "1", ...):
        gpu_utilization_percent    — compute utilization %
        gpu_memory_used_bytes      — total - free  (all processes on the GPU)
        gpu_memory_total_bytes     — physical VRAM
    """

    def __init__(self, interval_s: int) -> None:
        self._interval = interval_s
        self._task: asyncio.Task[None] | None = None
        self._poll_fn: Callable[[], None] | None = None
        self._shutdown_fn: Callable[[], None] | None = None
        self._device_count = 0

    def _init(self) -> None:
        result = _build_torch_poller() or _build_nvml_poller()
        if result is None:
            return
        self._device_count = result[0]
        self._poll_fn = result[1]
        self._shutdown_fn = result[2] if len(result) > 2 else None

    async def start(self) -> None:
        if self._interval <= 0:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._init)
        if self._poll_fn is None:
            logger.info("GpuMonitor: no CUDA devices detected — GPU metrics disabled")
            return
        self._task = asyncio.create_task(self._loop(), name="gpu-monitor")
        logger.info(
            "GPU monitor started: devices=%d interval=%ds",
            self._device_count,
            self._interval,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._shutdown_fn is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._shutdown_fn)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._poll_fn)
            except Exception:
                logger.exception("gpu-monitor poll failed")
