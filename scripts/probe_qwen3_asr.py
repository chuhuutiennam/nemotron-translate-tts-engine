#!/usr/bin/env python3
"""Gate 0 for Qwen3-ASR-1.7B as an ASR backend (docs/qwen3-asr-design.md).

Runs with the python of the separate Qwen venv (/opt/qwen-asr-venv: qwen-asr 0.0.6, vLLM 0.14.0, torch 2.9.1 cu128),
never the engine venv. It uses the public streaming API of qwen_asr as QwenLM/Qwen3-ASR
examples/example_qwen3_asr_vllm_streaming.py does: init_streaming_state / streaming_transcribe /
finish_streaming_transcribe. Every streaming call encodes the whole utterance so far again, so the cost grows
with the utterance length; this probe measures it on the GPU the engine would use.

Subcommands:
  env      driver (>= 570 for torch cu128), GPU, package versions, gcc / Python.h for the Triton JIT, model folder
  latency  load time, VRAM, time of each streaming call by audio length, finish time, serial stream capacity
  wer      transcribe a manifest (eval_wer.py prepare) streaming or offline -> JSONL that eval_wer.py score reads

Examples (repo root on the GPU server, M=$APP_DIR/models/Qwen3-ASR-1.7B):
  /opt/qwen-asr-venv/bin/python scripts/probe_qwen3_asr.py env --model-path $M
  /opt/qwen-asr-venv/bin/python scripts/probe_qwen3_asr.py latency --model-path $M --wav asr_data/banking_trouble.wav
  /opt/qwen-asr-venv/bin/python scripts/probe_qwen3_asr.py wer --model-path $M --manifest testdata/vi.tsv \\
      --out out/qwen_stream.jsonl
  /opt/nemotron-venv/bin/python scripts/kaggle/eval_wer.py score --base out/nemotron.jsonl out/qwen_stream.jsonl
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts" / "kaggle"):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))

from eval_wer import build_hypothesis, corpus_wer, load_manifest, wer_pairs  # noqa: E402
from nemotron_testbed.languages import short_language  # noqa: E402

SR = 16_000
MIN_DRIVER_MAJOR = 570  # torch cu128 (pulled by vLLM 0.14.0) needs an R570+ driver
EXPECTED_VERSIONS = {"qwen-asr": "0.0.6", "vllm": "0.14.0", "torch": "2.9.1", "transformers": "4.57.6"}
REPORTED_PACKAGES = ("qwen-asr", "vllm", "torch", "transformers", "numpy", "soxr", "triton", "flashinfer-python")
QWEN_LANGUAGE_BY_SHORT = {"vi": "Vietnamese", "en": "English"}


# ── pure helpers (tested without a GPU) ───────────────────────────────────────


def qwen_language(language: str | None) -> str | None:
    """Engine language tag (vi, vi-VN, en-US, auto) or a qwen_asr name -> the name qwen_asr takes; auto -> None.

    qwen_asr wants a capitalised English name ("Vietnamese") and rejects codes; None lets the model detect the
    language. Names other than Vietnamese/English are passed on normalised as qwen_asr does, and it validates them.
    """
    raw = (language or "").strip()
    if not raw or raw.lower() == "auto":
        return None
    mapped = QWEN_LANGUAGE_BY_SHORT.get(short_language(raw))
    if mapped:
        return mapped
    return raw[:1].upper() + raw[1:].lower()


def resolve_language(setting: str, row: dict) -> str | None:
    """--language value for one manifest row: '@lang' takes the row's lang column (as eval_wer.py run)."""
    return qwen_language(row.get("lang", "") if setting == "@lang" else setting)


def parse_driver_version(text: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in text.strip().split("."))
    except ValueError:
        return None


def driver_ok(version: tuple[int, ...] | None, minimum: int = MIN_DRIVER_MAJOR) -> bool:
    return bool(version) and version[0] >= minimum


def parse_cuda_index(device: str) -> int:
    """cuda / cuda:N -> N. vLLM always uses the first visible GPU, so the probe sets CUDA_VISIBLE_DEVICES=N."""
    kind, _, index = device.strip().partition(":")
    if kind != "cuda":
        raise ValueError(f"--device must be cuda or cuda:N (vLLM needs a GPU), got {device!r}")
    if not index:
        return 0
    if not index.isdigit():
        raise ValueError(f"bad CUDA index in --device {device!r}")
    return int(index)


def prepare_environ(device: str, offline: bool, environ=os.environ) -> int:
    """Set the environment before vLLM/torch start CUDA; return the physical GPU index (for nvidia-smi)."""
    index = parse_cuda_index(device)
    environ["CUDA_VISIBLE_DEVICES"] = str(index)
    environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if offline:
        environ.setdefault("HF_HUB_OFFLINE", "1")
        environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return index


def step_pieces(audio: np.ndarray, step_samples: int) -> Iterator[np.ndarray]:
    for start in range(0, audio.size, max(1, step_samples)):
        yield audio[start : start + step_samples]


def split_utterances(audio: np.ndarray, max_samples: int) -> list[np.ndarray]:
    """Hard cut at max_samples, like vad.max_utterance_ms forces an END (the engine's VAD usually cuts earlier)."""
    if audio.size == 0:
        return []
    if max_samples <= 0:
        return [audio]
    return [audio[start : start + max_samples] for start in range(0, audio.size, max_samples)]


def tile_to(audio: np.ndarray, samples: int) -> np.ndarray:
    """audio repeated until it has `samples` samples (the latency test needs speech up to the longest utterance)."""
    if audio.size == 0:
        raise ValueError("no audio to tile")
    return np.tile(audio, math.ceil(samples / audio.size))[:samples]


def summarize_ms(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def serial_streams(compute_s: float, audio_s: float, headroom: float) -> float:
    """Streams one model can serve one call at a time: each stream needs compute_s of GPU per audio_s of speech."""
    if audio_s <= 0:
        return 0.0
    gpu_per_audio_s = compute_s / audio_s
    return headroom / gpu_per_audio_s if gpu_per_audio_s > 0 else math.inf


def call_budget_ms(chunk_s: float, target_streams: int, headroom: float) -> float:
    """Mean time one call may take so target_streams streams (one call per chunk each) fit in headroom."""
    return chunk_s * 1000.0 * headroom / max(1, target_streams)


def fmt_ms(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "-"
    return f"mean {summary['mean']:7.1f}  p50 {summary['p50']:7.1f}  p95 {summary['p95']:7.1f}  max {summary['max']:7.1f} ms"


def result_row(row: dict, texts: list[str], audio_s: float, total_s: float, error: str, **extra) -> dict:
    """One JSONL row in the eval_wer.py run format (score/show/diff read it); extra keys are probe details.

    total_s is processing time (the probe does not pace audio in real time), and first_result_s is not measured.
    """
    return {
        "id": row["id"],
        "lang": row.get("lang", ""),
        "content_type": row.get("content_type", ""),
        "ref": row.get("reference", ""),
        "hyp": build_hypothesis(texts),
        "n_final": sum(1 for text in texts if text and text.strip()),
        "audio_s": round(audio_s, 3),
        "first_result_s": None,
        "total_s": round(total_s, 3),
        "error": error,
        **extra,
    }


# ── streaming one utterance ───────────────────────────────────────────────────


@dataclass
class StreamRun:
    text: str
    language: str
    call_ms: list[float] = field(default_factory=list)  # calls that ran the model (a whole chunk was buffered)
    call_audio_s: list[float] = field(default_factory=list)  # utterance audio the model encoded in each such call
    finish_ms: float | None = None  # None: nothing was left in the buffer, finish did not run the model
    compute_s: float = 0.0  # every call, including buffering-only ones, plus finish


def stream_utterance(
    asr,
    audio: np.ndarray,
    *,
    chunk_size_sec: float,
    language: str | None,
    context: str,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamRun:
    """Feed one utterance in steps of one chunk, as the engine would send STEP commands, then finish it."""
    state = asr.init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )
    run = StreamRun(text="", language="")
    for piece in step_pieces(audio, round(chunk_size_sec * SR)):
        before = state.chunk_id
        started = clock()
        asr.streaming_transcribe(piece, state)
        elapsed = clock() - started
        run.compute_s += elapsed
        if state.chunk_id != before:
            run.call_ms.append(elapsed * 1000.0)
            run.call_audio_s.append(state.audio_accum.size / SR)
    pending = state.buffer.size > 0
    started = clock()
    asr.finish_streaming_transcribe(state)
    elapsed = clock() - started
    run.compute_s += elapsed
    run.finish_ms = elapsed * 1000.0 if pending else None
    run.text = state.text or ""
    run.language = state.language or ""
    return run


def streaming_kwargs(args: argparse.Namespace) -> dict:
    return {
        "chunk_size_sec": args.chunk_size_sec,
        "context": args.context,
        "unfixed_chunk_num": args.unfixed_chunk_num,
        "unfixed_token_num": args.unfixed_token_num,
    }


def transcribe_row(asr, row: dict, args: argparse.Namespace, clock: Callable[[], float] = time.perf_counter) -> dict:
    audio_s = 0.0
    started = clock()
    try:
        audio = load_clip(Path(row["wav"]), args.narrowband)
        audio_s = audio.size / SR
        language = resolve_language(args.language, row)
        if args.mode == "offline":
            out = asr.transcribe(audio=(audio, SR), context=args.context, language=language)[0]
            texts, extra = [out.text or ""], {"mode": "offline", "language_detected": out.language or ""}
        else:
            pieces = split_utterances(audio, round(args.max_utterance_s * SR))
            runs = [
                stream_utterance(asr, piece, language=language, clock=clock, **streaming_kwargs(args))
                for piece in pieces
            ]
            calls = [ms for run in runs for ms in run.call_ms]
            finishes = [run.finish_ms for run in runs if run.finish_ms is not None]
            texts = [run.text for run in runs]
            extra = {
                "mode": "stream",
                "language_detected": ",".join(sorted({run.language for run in runs if run.language})),
                "utterances": len(pieces),
                "calls": len(calls),
                "call_ms_p50": round(float(np.percentile(calls, 50)), 1) if calls else None,
                "call_ms_max": round(max(calls), 1) if calls else None,
                "finish_ms_max": round(max(finishes), 1) if finishes else None,
            }
        error = ""
    except Exception as exc:  # one bad clip must not stop the run; the row carries the error
        texts, extra, error = [], {"mode": args.mode}, f"{type(exc).__name__}: {exc}"
    return result_row(row, texts, audio_s, clock() - started, error, **extra)


# ── GPU / model side (needs the Qwen venv) ────────────────────────────────────


def nvidia_smi(*query: str) -> list[list[str]] | None:
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
    """Whole-GPU memory in use: vLLM runs the model in a child process, so this process alone shows little."""
    rows = nvidia_smi("--query-gpu=memory.used", f"--id={index}")
    return int(rows[0][0]) if rows and rows[0][0].isdigit() else None


def llm_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return kwargs


def load_model(args: argparse.Namespace):
    from qwen_asr import Qwen3ASRModel

    kwargs = llm_kwargs(args)
    print(f"[load] Qwen3ASRModel.LLM(model={args.model_path!r}, {kwargs})", flush=True)
    return Qwen3ASRModel.LLM(model=args.model_path, **kwargs)


def load_clip(path: Path, narrowband: bool) -> np.ndarray:
    from nemotron_testbed.client_audio import read_wav, to_narrowband

    audio = read_wav(path)
    return to_narrowband(audio) if narrowband else audio


def args_for_json(args: argparse.Namespace) -> dict:
    """The command line as plain JSON values (paths as text, the subcommand function left out)."""

    def plain(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value

    return {key: plain(value) for key, value in vars(args).items() if key != "func"}


def write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


# ── subcommands ───────────────────────────────────────────────────────────────


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_model_dir(path: Path) -> list[str]:
    """Problems with a local Qwen3-ASR folder (HF snapshot); empty when it looks complete."""
    if not path.is_dir():
        return [f"{path} is not a folder"]
    problems = [f"missing {name}" for name in ("config.json", "tokenizer_config.json") if not (path / name).is_file()]
    if not list(path.glob("*.safetensors")):
        problems.append("no *.safetensors")
    return problems


def cmd_env(args: argparse.Namespace) -> int:
    index = prepare_environ(args.device, args.offline)
    failures: list[str] = []
    print(f"python {sys.version.split()[0]} ({sys.executable}), {platform.platform()}")
    gpus = nvidia_smi("--query-gpu=index,name,driver_version,memory.total,memory.used,compute_cap")
    if gpus is None:
        print("[FAIL] nvidia-smi did not run")
        return 2
    for gpu in gpus:
        print("gpu " + "  ".join(gpu) + "   (index, name, driver, MiB total, MiB used, compute capability)")
    driver = gpus[0][2] if gpus and len(gpus[0]) > 2 else ""
    if driver_ok(parse_driver_version(driver)):
        print(f"[ok]   driver {driver} >= {MIN_DRIVER_MAJOR} (torch cu128)")
    else:
        failures.append(f"driver {driver or '?'} < {MIN_DRIVER_MAJOR}: torch cu128 / vLLM 0.14 cannot use the GPU")

    for name in REPORTED_PACKAGES:
        version = package_version(name)
        expected = EXPECTED_VERSIONS.get(name)
        note = ""
        if expected and (version is None or not version.startswith(expected)):
            note = f"   <- expected {expected}"
            if name in ("qwen-asr", "vllm"):
                failures.append(f"{name} {version} is not {expected}")
        print(f"pkg  {name:18s} {version or 'not installed'}{note}")

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch cuda={cuda} cuda_version={torch.version.cuda} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        if cuda:
            print(f"torch device 0 = {torch.cuda.get_device_name(0)} (physical index {index})")
        else:
            failures.append("torch.cuda.is_available() is False")
    except ImportError:
        failures.append("torch is not installed in this venv")

    compiler = shutil.which("gcc") or shutil.which("cc")
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    print(f"gcc  {compiler or 'NOT FOUND'}   Python.h {'ok' if header.is_file() else 'NOT FOUND: ' + str(header)}")
    if not compiler or not header.is_file():
        print("[warn] Triton (used by vLLM) compiles a small C launcher at first use: needs gcc and python3.12-dev")

    if args.model_path:
        problems = check_model_dir(Path(args.model_path))
        size_gb = sum(p.stat().st_size for p in Path(args.model_path).glob("*") if p.is_file()) / 1e9 if not problems else 0
        if problems:
            failures.append(f"model {args.model_path}: {', '.join(problems)}")
        else:
            print(f"[ok]   model {args.model_path} ({size_gb:.2f} GB)")

    for failure in failures:
        print(f"[FAIL] {failure}")
    print("env: OK" if not failures else "env: FAILED")
    return 0 if not failures else 2


def cmd_latency(args: argparse.Namespace) -> int:
    if args.reps < 1:
        raise SystemExit("--reps must be >= 1")
    index = prepare_environ(args.device, args.offline)
    longest = max(args.utterance_s)
    speech = tile_to(np.concatenate([load_clip(Path(w), args.narrowband) for w in args.wav]), round(longest * SR))
    language = qwen_language(args.language)

    vram = {"before load": gpu_used_mib(index)}
    started = time.perf_counter()
    asr = load_model(args)
    load_s = time.perf_counter() - started
    vram["after load"] = gpu_used_mib(index)
    print(f"[load] {load_s:.1f} s")

    report: dict = {"args": args_for_json(args), "load_s": load_s, "runs": []}
    for chunk_s in args.chunk_sizes:
        kwargs = streaming_kwargs(args) | {"chunk_size_sec": chunk_s}
        stream_utterance(asr, speech[: round(longest * SR)], language=language, **kwargs)  # warm-up, discarded
        for utterance_s in args.utterance_s:
            audio = speech[: round(utterance_s * SR)]
            by_audio: dict[float, list[float]] = {}
            finishes: list[float] = []
            computes: list[float] = []
            last: StreamRun | None = None
            for _ in range(args.reps):
                last = stream_utterance(asr, audio, language=language, **kwargs)
                for audio_so_far, ms in zip(last.call_audio_s, last.call_ms):
                    by_audio.setdefault(round(audio_so_far, 2), []).append(ms)
                if last.finish_ms is not None:
                    finishes.append(last.finish_ms)
                computes.append(last.compute_s)
            compute = float(np.median(computes))
            streams = serial_streams(compute, utterance_s, args.headroom)
            budget = call_budget_ms(chunk_s, args.target_streams, args.headroom)
            worst = summarize_ms(by_audio[max(by_audio)]) if by_audio else None
            print(f"\n=== chunk {chunk_s:.1f} s, utterance {utterance_s:.1f} s, {args.reps} reps (+1 warm-up) ===")
            print("audio so far   streaming_transcribe call")
            for audio_so_far in sorted(by_audio):
                print(f"  {audio_so_far:5.1f} s     {fmt_ms(summarize_ms(by_audio[audio_so_far]))}")
            print(f"finish {utterance_s:4.1f} s   {fmt_ms(summarize_ms(finishes))}")
            print(f"compute per utterance (median) {compute * 1000:.0f} ms -> {compute / utterance_s:.3f} s GPU per s of"
                  f" speech -> about {streams:.1f} streams served one call at a time ({args.headroom:.0%} of the GPU)")
            if worst is not None:
                verdict = "fits" if worst["mean"] <= budget else "EXCEEDS"
                print(f"target {args.target_streams} streams: mean call at {max(by_audio):.1f} s {worst['mean']:.0f} ms"
                      f" {verdict} the {budget:.0f} ms budget (chunk x {args.headroom:.0%} / streams)")
            print(f"language {last.language!r} text {last.text!r}")
            report["runs"].append({
                "chunk_s": chunk_s,
                "utterance_s": utterance_s,
                "calls_by_audio_s": {str(k): summarize_ms(v) for k, v in sorted(by_audio.items())},
                "finish_ms": summarize_ms(finishes),
                "compute_s_median": compute,
                "serial_streams": streams,
                "call_budget_ms": budget,
                "language": last.language,
                "text": last.text,
            })
    vram["after benchmark"] = gpu_used_mib(index)
    print("\nVRAM MiB of the whole GPU (vLLM runs the model in a child process):")
    for name, used in vram.items():
        print(f"  {name:16s}: {used if used is not None else 'n/a'}")
    report["vram_mib"] = vram
    write_json(args.json_out, report)
    return 0


def cmd_wer(args: argparse.Namespace) -> int:
    prepare_environ(args.device, args.offline)
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    asr = load_model(args)
    for row in rows[: args.warmup]:
        transcribe_row(asr, row, args)
    results = []
    for done, row in enumerate(rows, 1):
        results.append(transcribe_row(asr, row, args))
        if done % 20 == 0 or done == len(rows):
            print(f"{args.out.stem}: {done}/{len(rows)}", file=sys.stderr, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    errors = [r for r in results if r["error"]]
    pairs = wer_pairs(r for r in results if not r["error"])
    audio = sum(r["audio_s"] for r in results)
    compute = sum(r["total_s"] for r in results)
    print(f"wrote {len(results)} results ({len(errors)} with errors) -> {args.out}")
    print(f"mode {args.mode}: WER {100 * corpus_wer(pairs.values()):.2f}% over {len(pairs)} clips with a reference;"
          f" processing {compute:.1f} s for {audio:.1f} s audio (RTF {compute / audio if audio else 0:.3f})")
    if args.mode == "stream":
        split = sum(1 for r in results if (r.get("utterances") or 0) > 1)
        print(f"clips cut at {args.max_utterance_s:.0f} s: {split} (the hard cut can split a word; the engine's VAD cuts at pauses)")
    if errors:
        print(f"first error: {errors[0]['id']}: {errors[0]['error']}")
    return 1 if errors else 0


# ── command line ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", default="models/Qwen3-ASR-1.7B", help="local HF snapshot of Qwen/Qwen3-ASR-1.7B")
    common.add_argument("--device", default="cuda:0", help="GPU as in asr.devices; sets CUDA_VISIBLE_DEVICES")
    common.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True,
                        help="HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (air-gapped server)")

    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="vLLM share of the whole GPU memory")
    model.add_argument("--max-model-len", type=int, default=2048, help="10 s of audio is a few hundred tokens")
    model.add_argument("--max-num-seqs", type=int, default=1, help="streaming sends one request at a time")
    model.add_argument("--max-new-tokens", type=int, default=64,
                       help="per call; the example uses 32, which may cut 4 s of Vietnamese (2 first chunks)")
    model.add_argument("--attention-backend", default="", help="empty = vLLM default; TRITON_ATTN on a T4")
    model.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs / torch.compile (slower calls)")
    model.add_argument("--dtype", default="", help="empty = model default (bfloat16)")
    model.add_argument("--quantization", default="", help="for example fp8 (L4 supports it); empty = none")
    model.add_argument("--chunk-size-sec", type=float, default=2.0, help="init_streaming_state chunk_size_sec")
    model.add_argument("--unfixed-chunk-num", type=int, default=2)
    model.add_argument("--unfixed-token-num", type=int, default=5)
    model.add_argument("--language", default="vi",
                       help="vi | en | auto (model detects) | a qwen_asr name; wer also takes @lang (manifest column)")
    model.add_argument("--context", default="", help="text put in the system prompt (hot words)")
    model.add_argument("--narrowband", action=argparse.BooleanOptionalAction, default=True,
                       help="8 kHz G.711 round trip like rtpgw (as eval_wer.py run)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", parents=[common], help="driver, GPU, packages, compiler, model folder")
    env.set_defaults(func=cmd_env)

    latency = sub.add_parser("latency", parents=[common, model], help="call / finish time, VRAM, stream capacity")
    latency.add_argument("--wav", type=Path, action="append", required=True,
                         help="speech WAV (repeatable); concatenated and repeated up to the longest utterance")
    latency.add_argument("--chunk-sizes", type=float, nargs="+", default=[1.0, 2.0], help="chunk_size_sec values")
    latency.add_argument("--utterance-s", type=float, nargs="+", default=[9.5],
                         help="utterance lengths; 9.5 stays under vad.max_utterance_ms 10000 and makes finish run")
    latency.add_argument("--reps", type=int, default=5)
    latency.add_argument("--target-streams", type=int, default=8)
    latency.add_argument("--headroom", type=float, default=0.7, help="share of the GPU time the streams may use")
    latency.add_argument("--json-out", type=Path, help="write the numbers as JSON")
    latency.set_defaults(func=cmd_latency)

    wer = sub.add_parser("wer", parents=[common, model], help="manifest -> JSONL for eval_wer.py score")
    wer.add_argument("--manifest", type=Path, required=True, help="vi.tsv from eval_wer.py prepare")
    wer.add_argument("--out", type=Path, required=True)
    wer.add_argument("--mode", choices=("stream", "offline"), default="stream",
                     help="stream: chunk by chunk like the engine; offline: transcribe() on the whole clip")
    wer.add_argument("--max-utterance-s", type=float, default=10.0, help="stream mode: hard cut, as vad.max_utterance_ms")
    wer.add_argument("--limit", type=int, default=0)
    wer.add_argument("--warmup", type=int, default=1, help="clips run once first and discarded")
    wer.set_defaults(func=cmd_wer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
