#!/usr/bin/env python3
"""Probe the audio-preprocessing denoiser on the GPU server with the engine's own loader.

Checks that onnxruntime-gpu really provides CUDAExecutionProvider next to torch (same load order as the
engine: torch/ASR first), then measures per-window latency, throughput and VRAM so that
audio_preprocessing.window.compute_budget_ms and the stream capacity can be set from numbers.
Run it with the engine idle and again while it serves real traffic.

On a CUDA device it re-runs itself once with the NVIDIA wheel libraries (site-packages/nvidia/*/lib) first in
LD_LIBRARY_PATH, as scripts/run-gpu-server.sh does for the engine; without them onnxruntime cannot load
libcudnn_adv.so.9 and falls back to CPU. --no-reexec turns this off.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

import numpy as np

from nemotron_testbed import resemble_models as rm

SR = rm.SAMPLE_RATE

NVIDIA_LIB_SUBDIRS = ("cudnn", "cuda_runtime", "cublas", "curand", "cufft", "cuda_nvrtc")
REEXEC_GUARD = "PROBE_PREPROC_GPU_REEXEC"


def site_dirs() -> list[str]:
    paths = sysconfig.get_paths()
    return list(dict.fromkeys([paths["purelib"], paths["platlib"]]))


def nvidia_lib_dirs(dirs: list[str]) -> list[str]:
    found = (Path(d) / "nvidia" / sub / "lib" for d in dirs for sub in NVIDIA_LIB_SUBDIRS)
    return [str(p) for p in found if p.is_dir()]


def ld_library_path_with_nvidia(current: str, dirs: list[str]) -> str | None:
    """LD_LIBRARY_PATH with the missing NVIDIA wheel dirs in front, or None when there is nothing to add."""
    present = [p for p in current.split(os.pathsep) if p]
    missing = [d for d in nvidia_lib_dirs(dirs) if d not in present]
    return os.pathsep.join(missing + present) if missing else None


def maybe_reexec_with_nvidia_libs(environ=os.environ, execv=os.execv) -> bool:
    if environ.get(REEXEC_GUARD):
        return False
    new = ld_library_path_with_nvidia(environ.get("LD_LIBRARY_PATH", ""), site_dirs())
    if new is None:
        return False
    environ["LD_LIBRARY_PATH"] = new
    environ[REEXEC_GUARD] = "1"
    print("[info] LD_LIBRARY_PATH lacked the NVIDIA wheel libraries; re-running with them first "
          "(as scripts/run-gpu-server.sh does, --no-reexec to skip)", flush=True)
    execv(sys.executable, [sys.executable, *sys.argv])
    return True


def _nvidia_smi(*query: str) -> list[list[str]] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [[cell.strip() for cell in line.split(",")] for line in out.splitlines() if line.strip()]


def gpu_used_mib(index: int) -> int | None:
    rows = _nvidia_smi("--query-gpu=memory.used", f"--id={index}")
    return int(rows[0][0]) if rows and rows[0][0].isdigit() else None


def own_used_mib(index: int) -> int | None:
    rows = _nvidia_smi("--query-compute-apps=pid,used_memory", f"--id={index}")
    for pid, used in rows or []:
        if pid == str(os.getpid()) and used.isdigit():
            return int(used)
    return None


def snapshot(kind: str, index: int) -> dict[str, int | None]:
    if kind != "cuda":
        return {"gpu": None, "own": None}
    return {"gpu": gpu_used_mib(index), "own": own_used_mib(index)}


def init_torch_cuda(device: str) -> None:
    """Load torch's bundled cuDNN into the process first, as the ASR pool does before the TTS/denoiser sessions."""
    import torch

    if not torch.cuda.is_available():
        print(f"[warn] torch.cuda.is_available() is False, skipping torch-first on {device}")
        return
    x = torch.randn(1, 1, 64, 64, device=device)
    conv = torch.nn.Conv2d(1, 1, 3, padding=1).to(device)
    conv(x)
    torch.cuda.synchronize(device)
    print(f"[info] torch {torch.__version__} initialised cuDNN on {device}")


def load_windows(wav: Path | None, window_n: int, hop_n: int) -> list[np.ndarray]:
    if wav is None:
        rng = np.random.default_rng(0)
        return [(rng.standard_normal(window_n) * 0.05).astype(np.float32) for _ in range(8)]
    from scipy.io import wavfile

    rate, data = wavfile.read(str(wav))
    if rate != SR:
        raise SystemExit(f"{wav}: expected {SR} Hz, got {rate}")
    if data.dtype not in (np.int16, np.float32):
        raise SystemExit(f"{wav}: unsupported sample type {data.dtype}")
    if data.ndim > 1:
        data = data[:, 0]
    x = data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)
    windows = [x[i : i + window_n] for i in range(0, len(x) - window_n + 1, hop_n)]
    windows = [w.copy() for w in windows if float(np.abs(w).max()) > 0.0]
    if not windows:
        raise SystemExit(f"{wav}: no non-silent {window_n / SR:.1f} s window")
    return windows


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values) * 1000.0
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def fmt(s: dict[str, float]) -> str:
    return f"p50 {s['p50']:7.1f}  p95 {s['p95']:7.1f}  p99 {s['p99']:7.1f}  max {s['max']:7.1f} ms"


def bench_worker(model, windows, iters: int, barrier, out: dict) -> None:
    run_times: list[float] = []
    totals: list[float] = []
    original = model.session.run

    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            run_times.append(time.perf_counter() - started)

    model.session.run = timed
    barrier.wait()
    try:
        for i in range(iters):
            started = time.perf_counter()
            model(windows[i % len(windows)])
            totals.append(time.perf_counter() - started)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        model.session.run = original
    out["totals"], out["run_times"] = totals, run_times


def print_environment() -> None:
    import onnxruntime as ort

    print(f"python {sys.version.split()[0]}  onnxruntime {ort.__version__}  device={ort.get_device()}")
    print(f"ort available providers: {ort.get_available_providers()}")
    try:
        import torch

        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  threads={torch.get_num_threads()}")
    except ImportError:
        print("torch: not installed")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}  LD_LIBRARY_PATH[0]={ld.split(':')[0] or '<unset>'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assets-dir", default="models/resemble-enhance", help="folder with denoiser.onnx")
    p.add_argument("--device", default="cuda:1", help="denoiser device, as audio_preprocessing.devices")
    p.add_argument("--wav", type=Path, help="16 kHz clip to cut windows from; default synthetic noise")
    p.add_argument("--iters", type=int, default=200, help="timed windows per worker")
    p.add_argument("--workers", type=int, default=1, help="concurrent model instances on the device")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--left-ms", type=int, default=1000, help="denoiser window defaults, see config.py")
    p.add_argument("--hop-ms", type=int, default=500)
    p.add_argument("--right-ms", type=int, default=500)
    p.add_argument("--budget-ms", type=int, default=500, help="compute_budget_ms the engine would use")
    p.add_argument("--torch-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-device", default="cuda:0", help="device torch initialises on when --torch-first")
    p.add_argument("--cudnn-algo-search", choices=("HEURISTIC", "EXHAUSTIVE", "DEFAULT"), help="override the engine setting")
    p.add_argument("--intra-threads", type=int, help="override ORT_CUDA_INTRA_THREADS")
    p.add_argument("--ort-log-severity", type=int, default=2, help="0 verbose .. 3 error; engine uses 3")
    p.add_argument("--reexec", action=argparse.BooleanOptionalAction, default=True,
                   help="on a CUDA device, re-run once with the NVIDIA wheel dirs first in LD_LIBRARY_PATH")
    args = p.parse_args()

    rm.ORT_LOG_SEVERITY = args.ort_log_severity
    if args.cudnn_algo_search:
        rm.ORT_CUDA_PROVIDER_OPTIONS["cudnn_conv_algo_search"] = args.cudnn_algo_search
    if args.intra_threads is not None:
        rm.ORT_CUDA_INTRA_THREADS = args.intra_threads

    kind, index = rm.parse_device(args.device)
    if kind == "cuda" and args.reexec and os.name == "posix":
        maybe_reexec_with_nvidia_libs()
    window_n = (args.left_ms + args.hop_ms + args.right_ms) * SR // 1000
    hop_s = args.hop_ms / 1000.0

    print_environment()
    print(f"[cfg] device={args.device} window={window_n / SR:.2f}s hop={hop_s:.2f}s workers={args.workers} iters={args.iters}")
    print(f"[cfg] provider options={rm.ORT_CUDA_PROVIDER_OPTIONS} intra_threads={rm.ORT_CUDA_INTRA_THREADS}")

    vram = {"baseline": snapshot(kind, index)}
    if args.torch_first and kind == "cuda":
        init_torch_cuda(args.torch_device)
        vram["after torch init"] = snapshot(kind, index)

    windows = load_windows(args.wav, window_n, args.hop_ms * SR // 1000)
    models, loads, colds = [], [], []
    for worker in range(args.workers):
        started = time.perf_counter()
        try:
            model = rm.load_window_model(
                "denoiser",
                args.assets_dir,
                args.device,
                enhancer_nfe=32,
                enhancer_lambd=1.0,
                enhancer_tau=0.5,
                enhancer_align=True,
                enhancer_align_max_ms=40,
            )
        except Exception as exc:
            print(f"\n[FAIL] load worker {worker}: {type(exc).__name__}: {exc}")
            if kind == "cuda":
                print("       If the CUDA EP did not start, look for 'libcudnn' / 'undefined symbol' lines above (severity 2);")
                print("       compare with --no-torch-first to tell a library-order conflict from a missing library.")
                if ld_library_path_with_nvidia(os.environ.get("LD_LIBRARY_PATH", ""), site_dirs()) is not None:
                    print("       LD_LIBRARY_PATH lacks the NVIDIA wheel libraries (site-packages/nvidia/*/lib): run without")
                    print("       --no-reexec, or export it as scripts/run-gpu-server.sh does.")
            return 2
        loads.append(time.perf_counter() - started)
        started = time.perf_counter()
        model(windows[0])
        colds.append(time.perf_counter() - started)
        models.append(model)
        if worker == 0:
            vram["after load + first call"] = snapshot(kind, index)
    for model in models:
        for i in range(args.warmup):
            model(windows[i % len(windows)])
    vram["after warm-up"] = snapshot(kind, index)

    barrier = threading.Barrier(args.workers + 1)
    outs: list[dict] = [{} for _ in models]
    threads = [
        threading.Thread(target=bench_worker, args=(m, windows, args.iters, barrier, o), daemon=True)
        for m, o in zip(models, outs)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    wall_start = time.perf_counter()
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall_start
    vram["after benchmark"] = snapshot(kind, index)

    errors = [o["error"] for o in outs if "error" in o]
    totals = [x for o in outs for x in o["totals"]]
    runs = [x for o in outs for x in o["run_times"]]
    if not totals:
        print(f"\n[FAIL] no window finished: {errors[:1]}")
        return 2

    total, run = stats(totals), stats(runs)
    per_s = len(totals) / wall
    budget = args.budget_ms
    print("\n=== RESULT ===")
    print(f"provider            : {models[0].provider}")
    print(f"load / first call   : {', '.join(f'{a:.2f}s / {b:.2f}s' for a, b in zip(loads, colds))}")
    print(f"window total        : {fmt(total)}")
    print(f"  of which ort.run  : {fmt(run)}")
    print(f"  cpu side (p50)    : {total['p50'] - run['p50']:.1f} ms  (resample + STFT/iSTFT + copies)")
    print(f"RTF per hop (p50)   : {total['p50'] / 1000.0 / hop_s:.2f}")
    print(f"throughput          : {per_s:.1f} windows/s over {wall:.1f}s  -> about {per_s * hop_s:.1f} streams at hop {hop_s:.2f}s")
    print(f"compute_budget_ms   : {budget} now; 1.5 x p99 = {int(math.ceil(1.5 * total['p99'] / 10.0) * 10)} (queue wait not included)")
    print(f"verdict             : p99 {'fits' if total['p99'] <= budget else 'EXCEEDS'} the {budget} ms budget")
    if errors:
        print(f"errors              : {len(errors)} worker(s), first: {errors[0]}")
    print("\nVRAM MiB (whole GPU / this process; includes other engine processes on the device):")
    for name, snap in vram.items():
        print(f"  {name:24s}: {snap['gpu'] if snap['gpu'] is not None else 'n/a':>6} / {snap['own'] if snap['own'] is not None else 'n/a':>6}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
