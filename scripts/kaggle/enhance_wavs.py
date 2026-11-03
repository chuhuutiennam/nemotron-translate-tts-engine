#!/usr/bin/env python3
"""Offline resemble-enhance branches for the Kaggle WER matrix (gate G0 of docs/design-notes/resemble-denoiser-integration-design.md).

Subcommands:
  fetch  download the third-party assets and check their sha256 (source at a pinned commit, checkpoint, ONNX denoiser)
  run    clips of a manifest -> (rtpgw G.711 path) -> denoiser or enhancer -> new WAVs + manifest with the same ids
  bench  time the denoiser and the enhancer (several nfe) per chunk on the current device

Examples (venv from setup_enhance.sh, PYTHONPATH=src):
  python scripts/kaggle/enhance_wavs.py fetch --assets-dir /kaggle/working/resemble_assets
  python scripts/kaggle/enhance_wavs.py run --manifest testdata/vi.tsv --out enh/enhancer --branch enhancer
  python scripts/kaggle/eval_wer.py run --manifest enh/enhancer/vi.tsv --no-narrowband --out out/H_enhancer.jsonl

`run` applies to_narrowband like `eval_wer.py run` does, so the processed manifest must be streamed with --no-narrowband.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import types
import urllib.request
import wave
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_wer  # noqa: E402

SAMPLE_RATE = 16_000
MODEL_RATE = 44_100
RESAMPLE_WINDOW = ("kaiser", 14.769656459379492)
ZERO_RUN_MIN = 64
DITHER_SEED = 12345
SILENT_RMS = 1.0 / 32768.0
ONNX_HOP = 420
ONNX_WIN = 4 * ONNX_HOP
ONNX_PAD = 441

ENHANCER_COMMIT = "8e978149bfe8abab3eb77d965d579a111afdb0ff"
ENHANCER_TARBALL = f"https://github.com/resemble-ai/resemble-enhance/archive/{ENHANCER_COMMIT}.tar.gz"
ENHANCER_TREE_SHA256 = "7a4bd5529946a4e49cfc8326f87e373a7351ab8b99794339c6c10183fb252964"
CHECKPOINT_BASE = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2"
CHECKPOINT_FILES = {
    "hparams.yaml": "80c3f15bc5a5b2cacf2c698699a0f6599d62911c0d53e1d6dee895c0d7cbaeac",
    "ds/G/latest": "37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f",
    "ds/G/default/mp_rank_00_model_states.pt": "f9d035f318de3e6d919bc70cf7ad7d32b4fe92ec5cbe0b30029a27f5db07d9d6",
}
ONNX_COMMIT = "577726152f06c8d8d6a4137371a5bd29a9936a87"
ONNX_URL = f"https://raw.githubusercontent.com/skeskinen/resemble-denoise-onnx-inference/{ONNX_COMMIT}/denoiser.onnx"
ONNX_SHA256 = "490640369540d1b0948352b75f880e215863a0de0b95a4b621ef590ee0e04e77"


# ── assets ────────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "resemble_enhance").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def download(url: str, dest: Path, retries: int = 3) -> str:
    """Stream url to dest and return its sha256; dest is only written on success."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "enhance_wavs/1"})
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
        except (OSError, http.client.HTTPException) as exc:
            last = exc
            print(f"download attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            continue
        part.replace(dest)
        return digest.hexdigest()
    part.unlink(missing_ok=True)
    raise RuntimeError(f"cannot download {url}: {last}")


def fetch_file(url: str, dest: Path, sha256: str) -> bool:
    """True if the file was downloaded, False if a valid copy was already there."""
    if dest.exists() and sha256_file(dest) == sha256:
        return False
    got = download(url, dest)
    if got != sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: sha256 {got} != expected {sha256}")
    return True


def fetch_enhancer_source(assets: Path, url: str = ENHANCER_TARBALL, tree_sha: str = ENHANCER_TREE_SHA256) -> bool:
    target = assets / "resemble-enhance"
    if target.exists() and tree_sha256(target) == tree_sha:
        return False
    assets.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=assets) as tmp:
        archive = Path(tmp) / "source.tar.gz"
        download(url, archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"unsafe member in source archive: {member.name}")
            tar.extractall(Path(tmp) / "x")
        roots = [p for p in (Path(tmp) / "x").iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"expected one top-level directory in the source archive, found {len(roots)}")
        got = tree_sha256(roots[0])
        if got != tree_sha:
            raise RuntimeError(f"source tree sha256 {got} != expected {tree_sha}")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(roots[0]), str(target))
    return True


def cmd_fetch(args: argparse.Namespace) -> None:
    assets: Path = args.assets_dir
    if "enhancer" in args.what:
        print(f"source {ENHANCER_COMMIT[:12]}: {'downloaded' if fetch_enhancer_source(assets) else 'present'}")
        for rel, sha in CHECKPOINT_FILES.items():
            done = fetch_file(f"{CHECKPOINT_BASE}/{rel}?download=true", assets / "enhancer_stage2" / rel, sha)
            print(f"{rel}: {'downloaded' if done else 'present'} sha256 {sha[:12]} ok")
    if "onnx" in args.what:
        done = fetch_file(ONNX_URL, assets / "denoiser.onnx", ONNX_SHA256)
        print(f"denoiser.onnx @ {ONNX_COMMIT[:12]}: {'downloaded' if done else 'present'} sha256 {ONNX_SHA256[:12]} ok")


# ── audio ─────────────────────────────────────────────────────────────────────


def narrowband_pcm16(path: Path, narrowband: bool) -> np.ndarray:
    """int16 16 kHz exactly as eval_wer.py run would send it (G.711 gateway path when narrowband)."""
    from nemotron_testbed.client_audio import pcm16_bytes, read_wav, to_narrowband

    audio = read_wav(path)
    if narrowband:
        audio = to_narrowband(audio)
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").astype(np.int16)


def write_pcm16_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.astype("<i2").tobytes())


def zero_runs(pcm: np.ndarray, min_run: int = ZERO_RUN_MIN) -> list[tuple[int, int]]:
    flags = np.concatenate([[False], pcm == 0, [False]])
    edges = np.diff(flags.astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_run]


def add_dither(pcm: np.ndarray, runs: Sequence[tuple[int, int]], sigma: float, seed: int = DITHER_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = pcm.astype(np.int32)
    for start, end in runs:
        out[start:end] = np.round(rng.normal(0.0, sigma, end - start)).astype(np.int32)
    return np.clip(out, -32768, 32767).astype(np.int16)


def resampled_pcm16(y44: np.ndarray, n: int) -> np.ndarray:
    """44.1 kHz float -> 16 kHz int16 of exactly n samples."""
    from scipy.signal import resample_poly

    y16 = resample_poly(np.asarray(y44, dtype=np.float64), 160, 441, window=RESAMPLE_WINDOW)
    y16 = np.pad(y16, (0, max(0, n - len(y16))))[:n]
    if not np.isfinite(y16).all():
        raise ValueError("non-finite model output")
    return np.clip(np.round(y16 * 32768.0), -32768, 32767).astype(np.int16)


def chunk_starts(n: int, chunk: int, overlap: int) -> list[int]:
    return [s for s in range(0, n, chunk - overlap) if s == 0 or n - s > overlap]


def sliding_crossfade(
    x: np.ndarray, chunk: int, overlap: int, process: Callable[[np.ndarray], np.ndarray], silent_rms: float = SILENT_RMS
) -> np.ndarray:
    """Run process on overlapping chunks and merge them with a linear crossfade; silent chunks give zeros."""
    starts = chunk_starts(len(x), chunk, overlap)
    acc, weight = np.zeros(len(x)), np.zeros(len(x))
    for i, start in enumerate(starts):
        part = x[start : start + chunk]
        w = np.ones(len(part))
        if i > 0:
            w[:overlap] = np.linspace(0.0, 1.0, overlap)
        if i < len(starts) - 1:
            w[-overlap:] = np.linspace(1.0, 0.0, overlap)
        weight[start : start + len(part)] += w
        if np.sqrt(np.mean(part**2)) < silent_rms:
            continue
        y = process(part)
        if y.shape != part.shape or not np.isfinite(y).all():
            raise ValueError(f"chunk at {start}: bad model output {y.shape} for {part.shape}")
        acc[start : start + len(part)] += y * w
    return acc / np.maximum(weight, 1e-12)


# ── processors ────────────────────────────────────────────────────────────────


def stub_deepspeed() -> None:
    """deepspeed is only imported by the training utilities of resemble-enhance."""
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


def shim_fsolve() -> None:
    """cfm.py does float(fsolve(...)), which numpy >= 2.5 rejects for a 1-element array."""
    import scipy.optimize

    original = scipy.optimize.fsolve
    scipy.optimize.fsolve = lambda *a, **k: np.squeeze(original(*a, **k))


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name == "cpu":
        print("WARNING: running on CPU (the enhancer is about 7x slower than real time there)", file=sys.stderr)
    return name


def load_resemble(assets: Path, device: str):
    stub_deepspeed()
    shim_fsolve()
    sys.path.insert(0, str(assets / "resemble-enhance"))
    from resemble_enhance.enhancer.inference import load_enhancer

    return load_enhancer(assets / "enhancer_stage2", device)


def official_processor(model, device: str, chunk_s: float, overlap_s: float, seed: int) -> Callable[[np.ndarray], np.ndarray]:
    import torch
    from resemble_enhance.inference import inference

    def process(pcm: np.ndarray) -> np.ndarray:
        torch.manual_seed(seed)
        x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        y, rate = inference(model=model, dwav=x, sr=SAMPLE_RATE, device=device, chunk_seconds=chunk_s, overlap_seconds=overlap_s)
        if rate != MODEL_RATE:
            raise RuntimeError(f"unexpected model rate {rate}")
        return resampled_pcm16(y.numpy(), len(pcm))

    return process


class OnnxDenoiser:
    """Third-party ONNX export of the resemble-enhance denoiser, official pre/post processing."""

    def __init__(self, onnx_path: Path, chunk_s: float, overlap_s: float) -> None:
        import onnxruntime as ort
        import torch

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        wanted = ("CUDAExecutionProvider", "CPUExecutionProvider")
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers, sess_options=opts)
        self.provider = self.session.get_providers()[0]
        self.window = torch.hann_window(ONNX_WIN)
        self.chunk = int(MODEL_RATE * chunk_s)
        self.overlap = int(MODEL_RATE * overlap_s)

    def run_chunk(self, c: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        length = len(c)
        abs_max = max(float(np.abs(c).max()), 1e-7)
        x = torch.from_numpy(np.pad(c / abs_max, (0, ONNX_PAD)).astype(np.float32))[None]
        x = x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)
        spec = torch.stft(x, n_fft=ONNX_WIN, hop_length=ONNX_HOP, win_length=ONNX_WIN, window=self.window, return_complex=True)[..., :-1]
        phase = spec.angle()
        feeds = {"mag": spec.abs().numpy(), "cos": phase.cos().numpy(), "sin": phase.sin().numpy()}
        mag, cos, sin = (torch.from_numpy(o) for o in self.session.run(None, feeds))
        spec = F.pad(torch.complex(mag * cos, mag * sin), (0, 1), "replicate")
        y = torch.istft(spec, n_fft=ONNX_WIN, hop_length=ONNX_HOP, win_length=ONNX_WIN, window=self.window, return_complex=False)
        y = F.pad(y, (0, x.shape[-1] - y.shape[-1]))[0].numpy().astype(np.float64)
        return y[:length] * abs_max

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        from scipy.signal import resample_poly

        x = resample_poly(pcm.astype(np.float64) / 32768.0, 441, 160, window=RESAMPLE_WINDOW)
        y = sliding_crossfade(x, self.chunk, self.overlap, self.run_chunk)
        return resampled_pcm16(y, len(pcm))


def build_processor(args: argparse.Namespace) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    import torch

    info: dict = {"torch": torch.__version__, "branch": args.branch}
    if args.branch == "denoiser" and args.denoiser_backend == "onnx":
        denoiser = OnnxDenoiser(args.assets_dir / "denoiser.onnx", args.chunk_seconds, args.overlap_seconds)
        info.update(backend="onnx", provider=denoiser.provider, onnx_commit=ONNX_COMMIT, onnx_sha256=ONNX_SHA256)
        return denoiser, info
    device = resolve_device(args.device)
    model = load_resemble(args.assets_dir, device)
    info.update(device=device, enhancer_commit=ENHANCER_COMMIT, checkpoint_sha256=CHECKPOINT_FILES["ds/G/default/mp_rank_00_model_states.pt"])
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(device)
    if args.branch == "enhancer":
        model.configurate_(nfe=args.nfe, solver="midpoint", lambd=args.lambd, tau=args.tau)
        info.update(backend="official-enhancer", nfe=args.nfe, lambd=args.lambd, tau=args.tau)
        target = model
    else:
        info.update(backend="official-torch-denoiser")
        target = model.denoiser
    return official_processor(target, device, args.chunk_seconds, args.overlap_seconds, args.seed), info


# ── run ───────────────────────────────────────────────────────────────────────


def preprocess_clip(
    pcm: np.ndarray, processor: Callable[[np.ndarray], np.ndarray], dither_sigma: float, rezero: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the processed clip and the mask of the digital-zero runs of the input."""
    runs = zero_runs(pcm)
    mask = np.zeros(len(pcm), dtype=bool)
    for start, end in runs:
        mask[start:end] = True
    source = add_dither(pcm, runs, dither_sigma) if dither_sigma > 0 else pcm
    out = processor(source)
    if out.shape != pcm.shape:
        raise ValueError(f"processor changed the length: {len(pcm)} -> {len(out)}")
    if rezero and mask.any():
        out = out.copy()
        out[mask] = 0
    return out, mask


def run_branch(
    rows: Sequence[dict[str, str]],
    out_dir: Path,
    load_pcm: Callable[[Path], np.ndarray],
    processor: Callable[[np.ndarray], np.ndarray],
    dither_sigma: float = 0.0,
    rezero: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    wav_dir = out_dir / "wav"
    new_rows: list[dict[str, str]] = []
    stats: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        pcm = load_pcm(Path(row["wav"]))
        t0 = time.perf_counter()
        out, mask = preprocess_clip(pcm, processor, dither_sigma, rezero)
        elapsed = time.perf_counter() - t0
        path = wav_dir / f"{row['id']}.wav"
        write_pcm16_wav(path, out)
        new_rows.append({**row, "wav": str(path.resolve())})
        audio_s = len(pcm) / SAMPLE_RATE
        energy_in = float(np.sum(pcm.astype(np.float64) ** 2))
        energy_out = float(np.sum(out.astype(np.float64) ** 2))
        stats.append(
            {
                "id": row["id"],
                "audio_s": round(audio_s, 3),
                "proc_s": round(elapsed, 3),
                "rtf": round(elapsed / audio_s, 3) if audio_s else None,
                "energy_db": round(10 * np.log10((energy_out + 1.0) / (energy_in + 1.0)), 2),
                "zero_run_pct": round(100.0 * float(mask.mean()), 2) if len(mask) else 0.0,
            }
        )
        if index % 20 == 0 or index == len(rows):
            print(f"{out_dir.name}: {index}/{len(rows)}", file=sys.stderr, flush=True)
    return new_rows, stats


def summarize(label: str, stats: Sequence[dict[str, object]]) -> str:
    audio = sum(s["audio_s"] for s in stats)
    proc = sum(s["proc_s"] for s in stats)
    rtfs = [s["rtf"] for s in stats if s["rtf"] is not None]
    energy = np.array([s["energy_db"] for s in stats]) if stats else np.zeros(1)
    zero = np.array([s["zero_run_pct"] for s in stats]) if stats else np.zeros(1)
    return (
        f"{label}: {len(stats)} clips, {audio / 60:.1f} min audio, {proc:.0f} s processing (RTF {proc / max(audio, 1e-9):.3f}, "
        f"per clip p50 {np.percentile(rtfs, 50) if rtfs else 0:.3f} p95 {np.percentile(rtfs, 95) if rtfs else 0:.3f}); "
        f"energy change median {np.median(energy):+.2f} dB (min {energy.min():+.2f}); "
        f"digital-zero runs {zero.mean():.2f}% of samples on average, max {zero.max():.2f}%"
    )


def cmd_run(args: argparse.Namespace) -> None:
    if args.dither_sigma < 0:
        raise SystemExit("--dither-sigma must be >= 0")
    rows = eval_wer.load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    processor, info = build_processor(args)
    args.out.mkdir(parents=True, exist_ok=True)
    new_rows, stats = run_branch(
        rows, args.out, lambda p: narrowband_pcm16(p, args.narrowband), processor, args.dither_sigma, args.rezero
    )
    eval_wer.save_manifest(args.out / args.manifest.name, new_rows)
    with (args.out / "stats.jsonl").open("w", encoding="utf-8") as fh:
        for item in stats:
            fh.write(json.dumps(item) + "\n")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    (args.out / "config.json").write_text(json.dumps({"args": settings, "info": info}, indent=2), encoding="utf-8")
    print(summarize(f"{args.branch} ({info['backend']})", stats))
    print(f"wrote {len(new_rows)} clips -> {args.out / args.manifest.name}")


# ── bench ─────────────────────────────────────────────────────────────────────


def percentile_row(times: Sequence[float]) -> dict[str, float]:
    return {
        "p50_s": round(float(np.percentile(times, 50)), 4),
        "p95_s": round(float(np.percentile(times, 95)), 4),
        "max_s": round(float(np.max(times)), 4),
    }


def benchmark_signal(rows: Sequence[dict[str, str]], seconds: float, narrowband: bool) -> np.ndarray:
    need = int(seconds * SAMPLE_RATE)
    parts: list[np.ndarray] = []
    total = 0
    for _ in range(100):
        for row in rows:
            pcm = narrowband_pcm16(Path(row["wav"]), narrowband)
            parts.append(pcm)
            total += len(pcm)
            if total >= need:
                return np.concatenate(parts)[:need]
    raise ValueError("the manifest has no audio")


def cmd_bench(args: argparse.Namespace) -> None:
    import torch
    from scipy.signal import resample_poly

    device = resolve_device(args.device)
    model = load_resemble(args.assets_dir, device)
    from resemble_enhance.inference import inference_chunk, remove_weight_norm_recursively

    remove_weight_norm_recursively(model)
    rows = eval_wer.load_manifest(args.manifest)
    pcm = benchmark_signal(rows, max(args.chunk_seconds) + 1.0, args.narrowband)
    wave44 = resample_poly(pcm.astype(np.float64) / 32768.0, 441, 160, window=RESAMPLE_WINDOW)
    signal = torch.from_numpy(wave44.astype(np.float32))
    cuda = device.startswith("cuda")
    gpu = torch.cuda.get_device_name(device) if cuda else "cpu"
    print(f"device {device} ({gpu}), torch {torch.__version__}, runs {args.runs} after {args.warmup} warm-up")

    targets: list[tuple[str, object, int | None]] = [("denoiser", model.denoiser, None)]
    targets += [(f"enhancer_nfe{nfe}", model, nfe) for nfe in args.nfe]
    results = []
    for name, module, nfe in targets:
        if nfe is not None:
            model.configurate_(nfe=nfe, solver="midpoint", lambd=args.lambd, tau=args.tau)
        for seconds in args.chunk_seconds:
            chunk = signal[: int(MODEL_RATE * seconds)]
            for _ in range(args.warmup):
                inference_chunk(module, chunk, MODEL_RATE, device)
            if cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                inference_chunk(module, chunk, MODEL_RATE, device)
                if cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            row = {"model": name, "chunk_s": seconds, "nfe": nfe, **percentile_row(times)}
            row["rtf_p50"] = round(row["p50_s"] / seconds, 4)
            row["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None
            results.append(row)
            print(
                f"{name:<16} chunk {seconds:>5.1f} s  p50 {row['p50_s']:.3f} s  p95 {row['p95_s']:.3f} s  "
                f"max {row['max_s']:.3f} s  RTF {row['rtf_p50']:.3f}  peak {row['peak_mib']} MiB"
            )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"device": device, "gpu": gpu, "torch": torch.__version__, "runs": args.runs, "results": results}
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


# ── cli ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--assets-dir", type=Path, required=True, help="directory filled by `fetch`")
        p.add_argument("--device", default="auto", help="auto, cpu or cuda:N")
        p.add_argument("--narrowband", action=argparse.BooleanOptionalAction, default=True, help="rtpgw G.711 round trip first")

    fetch = sub.add_parser("fetch", help="download and verify the assets")
    fetch.add_argument("--assets-dir", type=Path, required=True)
    fetch.add_argument("--what", nargs="+", choices=("enhancer", "onnx"), default=["enhancer", "onnx"])
    fetch.set_defaults(func=cmd_fetch)

    run = sub.add_parser("run", help="process a manifest")
    common(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--branch", choices=("denoiser", "enhancer"), required=True)
    run.add_argument("--denoiser-backend", choices=("onnx", "torch"), default="onnx", help="onnx = third-party export, torch = official module")
    run.add_argument("--nfe", type=int, default=32, help="enhancer: CFM network forwards (midpoint = nfe/2 steps of 2)")
    run.add_argument("--lambd", type=float, default=1.0, help="enhancer: denoiser mix (official CLI default 0.5)")
    run.add_argument("--tau", type=float, default=0.5)
    run.add_argument("--chunk-seconds", type=float, default=10.0)
    run.add_argument("--overlap-seconds", type=float, default=1.0)
    run.add_argument("--dither-sigma", type=float, default=0.0, help="Gaussian dither (LSB) in digital-zero runs of >= 64 samples")
    run.add_argument("--rezero", action="store_true", help="force the input's digital-zero runs back to 0 after processing")
    run.add_argument("--seed", type=int, default=0, help="torch seed set before every clip (the enhancer is stochastic)")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="time the models per chunk")
    common(bench)
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--chunk-seconds", type=float, nargs="+", default=[2.0, 10.0])
    bench.add_argument("--nfe", type=int, nargs="+", default=[32, 64])
    bench.add_argument("--lambd", type=float, default=1.0)
    bench.add_argument("--tau", type=float, default=0.5)
    bench.add_argument("--runs", type=int, default=10)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--out", type=Path, default=None)
    bench.set_defaults(func=cmd_bench)
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("TQDM_DISABLE", "1")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
