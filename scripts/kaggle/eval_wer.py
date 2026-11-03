#!/usr/bin/env python3
"""WER harness for the Kaggle test matrix: prepare data, stream it to the engine, score the runs.

Subcommands:
  prepare  download DataStudio/Vietnamese_ASR_TestingData -> 16 kHz wavs + vi.tsv manifest
  mix      interleave vi clips with one English wav (prompt-switch latency test)
  run      stream every manifest row to the engine (one gRPC stream per clip), write JSONL
  score    WER per run, paired against a base run, with a bootstrap 95% CI
  show     print the hypotheses of a run (translate runs have no reference)
  diff     clips whose word errors changed most between a base run and another run
  logstats utterances per clip, empty ASR finals and utterance length per run, from engine logs

Examples:
  python scripts/kaggle/eval_wer.py prepare --out /kaggle/working/testdata
  python scripts/kaggle/eval_wer.py run --manifest testdata/vi.tsv --out out/A_auto.jsonl --language ""
  python scripts/kaggle/eval_wer.py run --manifest testdata/vi.tsv --out out/B_src.jsonl --language "" --language-source vi
  python scripts/kaggle/eval_wer.py score --base out/A_auto.jsonl out/B_src.jsonl
  python scripts/kaggle/eval_wer.py diff --base out/B_source_vi.jsonl out/S600_source_vi.jsonl -n 5
  python scripts/kaggle/eval_wer.py logstats --tags B S600 out/logs/*/nemotron-engine.log

`--language ""` leaves AudioChunk.language unset (transcribe: source auto). `--language-source @lang`
sends each row's `lang` column. Needs PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import unicodedata
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ID = "DataStudio/Vietnamese_ASR_TestingData"
PARQUET_FILE = "data/test-00000-of-00001.parquet"
SAMPLE_RATE = 16_000
MANIFEST_FIELDS = ("id", "wav", "duration_s", "content_type", "reference", "lang")
_PUNCT_RE = re.compile(r"[^\w\s]|_")


# ── text and WER ──────────────────────────────────────────────────────────────


def normalize_text(text: str) -> list[str]:
    """NFC, lowercase, punctuation removed; digits stay as written."""
    lowered = unicodedata.normalize("NFC", text or "").lower()
    return _PUNCT_RE.sub(" ", lowered).split()


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1]


def corpus_wer(pairs: Iterable[tuple[int, int]]) -> float:
    items = list(pairs)
    words = sum(n for _, n in items)
    return sum(e for e, _ in items) / words if words else 0.0


def paired_bootstrap(
    base: Sequence[tuple[int, int]],
    other: Sequence[tuple[int, int]],
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """WER(other) - WER(base) over the same clips, with a 95% bootstrap CI (clips resampled)."""
    if len(base) != len(other):
        raise ValueError("base and other must cover the same clips")
    if not base:
        return 0.0, 0.0, 0.0
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    picks = np.random.default_rng(seed).integers(0, len(a), size=(iterations, len(a)))
    base_wer = a[picks, 0].sum(axis=1) / np.maximum(a[picks, 1].sum(axis=1), 1.0)
    other_wer = b[picks, 0].sum(axis=1) / np.maximum(b[picks, 1].sum(axis=1), 1.0)
    deltas = other_wer - base_wer
    return corpus_wer(other) - corpus_wer(base), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_hypothesis(texts: Iterable[str]) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip())


def wer_pairs(rows: Iterable[dict]) -> dict[str, tuple[int, int]]:
    """id -> (word errors, reference words); rows without a reference are skipped."""
    pairs: dict[str, tuple[int, int]] = {}
    for row in rows:
        ref = normalize_text(row.get("ref", ""))
        if ref:
            pairs[row["id"]] = (edit_distance(ref, normalize_text(row.get("hyp", ""))), len(ref))
    return pairs


# ── manifest and audio ────────────────────────────────────────────────────────


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def save_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def interleave_manifest(vi_rows: Sequence[dict], en_row: dict, count: int) -> list[dict]:
    """vi, en, vi, en, ...: with one ASR worker every stream changes the language."""
    mixed: list[dict] = []
    for index, row in enumerate(vi_rows[:count]):
        mixed.append({**row, "lang": "vi"})
        mixed.append({**en_row, "id": f"en-{index:03d}", "lang": "en"})
    return mixed


def resolve_language_source(setting: str, row: dict) -> str:
    return row.get("lang", "") if setting == "@lang" else setting


def decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 and its sample rate from WAV (16-bit) or, failing that, any libsndfile format."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if wav.getsampwidth() == 2:
                channels = wav.getnchannels()
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, channels)
                return pcm.astype(np.float32).mean(axis=1) / 32768.0, wav.getframerate()
    except (wave.Error, EOFError):
        pass
    import soundfile

    data, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return data.mean(axis=1), rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    from nemotron_testbed.client_audio import pcm16_bytes

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16_bytes(audio))


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


# ── prepare / mix ─────────────────────────────────────────────────────────────


def cmd_prepare(args: argparse.Namespace) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(repo_id=REPO_ID, filename=PARQUET_FILE, repo_type="dataset")
    records = pd.read_parquet(parquet).to_dict("records")
    if args.limit:
        records = records[: args.limit]
    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        audio, rate = decode_audio(record["audio"]["bytes"])
        if rate != SAMPLE_RATE:
            import soxr

            audio = soxr.resample(audio, rate, SAMPLE_RATE)
        clip_id = f"vi-{index:03d}"
        path = wav_dir / f"{clip_id}.wav"
        write_wav(path, audio.astype(np.float32))
        rows.append(
            {
                "id": clip_id,
                "wav": str(path.resolve()),
                "duration_s": f"{audio.size / SAMPLE_RATE:.2f}",
                "content_type": record.get("type of content") or "",
                "reference": record["label"],
                "lang": "vi",
            }
        )
    save_manifest(args.out / "vi.tsv", rows)
    total = sum(float(r["duration_s"]) for r in rows)
    print(f"prepared {len(rows)} clips ({total / 60:.1f} min) in {args.out}")


def cmd_mix(args: argparse.Namespace) -> None:
    vi_rows = load_manifest(args.manifest)
    en_row = {
        "id": "en",
        "wav": str(args.en_wav.resolve()),
        "duration_s": f"{wav_duration_s(args.en_wav):.2f}",
        "content_type": "en",
        "reference": "",
        "lang": "en",
    }
    mixed = interleave_manifest(vi_rows, en_row, args.count)
    save_manifest(args.out, mixed)
    print(f"mixed manifest: {len(mixed)} rows -> {args.out}")


# ── run ───────────────────────────────────────────────────────────────────────


def stream_clip(address: str, row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    import grpc

    from nemotron_testbed.client_audio import RtpTimeline, pcm16_bytes, read_wav, to_narrowband
    from nemotron_testbed.pb import speech_pb2, speech_pb2_grpc

    audio = read_wav(Path(row["wav"]))
    if args.narrowband:
        audio = to_narrowband(audio)
    language_source = resolve_language_source(args.language_source, row)
    chunk_samples = max(1, SAMPLE_RATE * args.chunk_ms // 1000)
    session_id = f"{args.tag}-{row['id']}-{uuid.uuid4().hex[:6]}"
    rtp = RtpTimeline(SAMPLE_RATE, args.rtp_clock_rate, 0)

    def requests():
        for chunk_seq, start in enumerate(range(0, audio.size, chunk_samples)):
            chunk = audio[start : start + chunk_samples]
            yield speech_pb2.AudioChunk(
                session_id=session_id,
                stream_id="MO",
                pcm=pcm16_bytes(chunk),
                sample_rate=SAMPLE_RATE,
                channels=1,
                rtp_timestamp=rtp.next_timestamp(chunk.size),
                rtp_clock_rate=args.rtp_clock_rate,
                chunk_seq=chunk_seq,
                duration_ms=round(chunk.size * 1000 / SAMPLE_RATE),
                end_of_stream=start + chunk_samples >= audio.size,
                language=args.language,
                language_source=language_source,
                task=args.task,
            )
            if args.pace:
                time.sleep(chunk.size / SAMPLE_RATE)

    finals: list[str] = []
    first_result: float | None = None
    error = ""
    t0 = time.monotonic()
    channel = grpc.insecure_channel(address)
    try:
        call = speech_pb2_grpc.SpeechStreamStub(channel).Recognize(requests(), timeout=args.timeout_s)
        for result in call:
            if first_result is None:
                first_result = time.monotonic() - t0
            if result.is_final:
                finals.append(result.text)
    except grpc.RpcError as exc:
        error = f"{exc.code()} {exc.details()}"
    finally:
        channel.close()
    return {
        "id": row["id"],
        "lang": row.get("lang", ""),
        "content_type": row.get("content_type", ""),
        "ref": row.get("reference", ""),
        "hyp": build_hypothesis(finals),
        "n_final": len(finals),
        "audio_s": round(audio.size / SAMPLE_RATE, 3),
        "first_result_s": None if first_result is None else round(first_result, 3),
        "total_s": round(time.monotonic() - t0, 3),
        "error": error,
    }


def cmd_run(args: argparse.Namespace) -> None:
    if args.language.lower() == "auto":
        args.language = ""
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    for row in rows[: args.warmup]:
        stream_clip(args.address, row, args)
    if args.warmup:
        print(f"warmup: {min(args.warmup, len(rows))} clip(s) discarded", file=sys.stderr)

    done = 0
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for result in pool.map(lambda r: stream_clip(args.address, r, args), rows):
            results.append(result)
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"{args.out.stem}: {done}/{len(rows)}", file=sys.stderr, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    errors = sum(1 for r in results if r["error"])
    print(f"wrote {len(results)} results ({errors} with errors) -> {args.out}")


# ── score / show ──────────────────────────────────────────────────────────────


def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latency(rows: Sequence[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return "-"
    return f"{np.median(values):.2f}/{np.percentile(values, 95):.2f}"


def summarize(name: str, rows: Sequence[dict], base_name: str, base_pairs: dict) -> dict:
    pairs = wer_pairs(rows)
    summary = {
        "run": name,
        "clips": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "wer_pct": "",
        "delta_pp": "",
        "ci95_pp": "",
        "final_per_clip": f"{np.mean([r['n_final'] for r in rows]):.2f}" if rows else "-",
        "first_result_s_med/p95": _latency(rows, "first_result_s"),
        "total_s_med/p95": _latency(rows, "total_s"),
    }
    if pairs:
        summary["wer_pct"] = f"{100 * corpus_wer(pairs.values()):.2f}"
        if name != base_name:
            ids = sorted(set(pairs) & set(base_pairs))
            delta, low, high = paired_bootstrap([base_pairs[i] for i in ids], [pairs[i] for i in ids])
            summary["delta_pp"] = f"{100 * delta:+.2f}"
            summary["ci95_pp"] = f"[{100 * low:+.2f}, {100 * high:+.2f}]"
    return summary


def _print_table(rows: Sequence[dict]) -> None:
    columns = list(rows[0])
    widths = [max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in columns]
    print("  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(columns, widths)))


def cmd_score(args: argparse.Namespace) -> None:
    base_rows = load_results(args.base)
    base_name = args.base.stem
    base_pairs = wer_pairs(base_rows)
    runs = [(base_name, base_rows)] + [(p.stem, load_results(p)) for p in args.runs]

    print(f"WER vs base '{base_name}' (delta = run - base, percentage points, 95% bootstrap CI over clips)")
    table = [summarize(name, rows, base_name, base_pairs) for name, rows in runs]
    _print_table(table)

    types = sorted({r.get("content_type", "") for _, rows in runs for r in rows if r.get("ref")})
    if len(types) > 1:
        print("\nWER % by content type")
        by_type = []
        for name, rows in runs:
            line: dict[str, object] = {"run": name}
            for content_type in types:
                subset = wer_pairs(r for r in rows if r.get("content_type") == content_type)
                line[content_type or "(none)"] = f"{100 * corpus_wer(subset.values()):.2f}" if subset else "-"
            by_type.append(line)
        _print_table(by_type)

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"\nwrote {args.csv}")


def cmd_show(args: argparse.Namespace) -> None:
    for row in load_results(args.run)[: args.n]:
        print(f"[{row['id']}] error={row['error'] or '-'} final_segments={row['n_final']}")
        if row.get("ref"):
            print(f"  ref: {row['ref']}")
        print(f"  hyp: {row['hyp']}")


def clip_diffs(base_rows: Sequence[dict], other_rows: Sequence[dict]) -> list[dict]:
    """Per clip present in both runs with a reference: word errors in each run, worst change first."""
    base = {r["id"]: r for r in base_rows}
    diffs: list[dict] = []
    for row in other_rows:
        before = base.get(row["id"])
        ref = normalize_text(row.get("ref", ""))
        if before is None or not ref:
            continue
        base_errors = edit_distance(ref, normalize_text(before.get("hyp", "")))
        other_errors = edit_distance(ref, normalize_text(row.get("hyp", "")))
        diffs.append(
            {
                "id": row["id"],
                "ref_words": len(ref),
                "base_errors": base_errors,
                "errors": other_errors,
                "delta": other_errors - base_errors,
                "ref": row.get("ref", ""),
                "base_hyp": before.get("hyp", ""),
                "hyp": row.get("hyp", ""),
            }
        )
    diffs.sort(key=lambda d: (-d["delta"], d["id"]))
    return diffs


def cmd_diff(args: argparse.Namespace) -> None:
    diffs = clip_diffs(load_results(args.base), load_results(args.run))
    worse = [d for d in diffs if d["delta"] > 0]
    better = [d for d in reversed(diffs) if d["delta"] < 0]
    print(
        f"{args.run.stem} vs {args.base.stem}: {len(diffs)} clips, {len(worse)} worse, "
        f"{len(better)} better, {len(diffs) - len(worse) - len(better)} unchanged (word errors per clip)"
    )
    for title, items in (("worse", worse), ("better", better)):
        print(f"\n## {title} (top {min(args.n, len(items))})")
        for d in items[: args.n]:
            print(f"[{d['id']}] errors {d['base_errors']} -> {d['errors']} of {d['ref_words']} words")
            print(f"  ref : {d['ref']}")
            print(f"  base: {d['base_hyp']}")
            print(f"  run : {d['hyp']}")


_LOG_STAGES = ('"VAD_START"', '"VAD_END"', '"ASR_FINAL"')


def log_stats(lines: Iterable[str], tags: Sequence[str]) -> list[dict]:
    """Per run tag, from the engine's JSON trace lines: clips, utterances per clip, empty ASR finals,
    utterances cut by max_utterance_ms and utterance length. Session ids are "<tag>-<clip id>-<hex>"
    (stream_clip), so lines of several runs in one log are told apart by the session prefix."""
    wanted = list(dict.fromkeys(tags))
    stats = {
        tag: {"sessions": set(), "utterances": 0, "finals": 0, "empty": 0, "max_cut": 0, "audio_ms": []}
        for tag in wanted
    }
    for line in lines:
        if not any(stage in line for stage in _LOG_STAGES):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        tag = str(event.get("session_id", "")).split("-", 1)[0]
        bucket = stats.get(tag)
        if bucket is None:
            continue
        stage = event.get("stage")
        if stage == "VAD_START":
            bucket["sessions"].add(event["session_id"])
            bucket["utterances"] += 1
        elif stage == "VAD_END":
            if event.get("reason") == "max_utterance":
                bucket["max_cut"] += 1
        elif stage == "ASR_FINAL":
            bucket["sessions"].add(event["session_id"])
            bucket["finals"] += 1
            text_len = event.get("text_len")
            if (text_len == 0) if text_len is not None else not str(event.get("text", "")).strip():
                bucket["empty"] += 1
            if isinstance(event.get("utterance_audio_ms"), (int, float)):
                bucket["audio_ms"].append(float(event["utterance_audio_ms"]))
    table: list[dict] = []
    for tag in wanted:
        b = stats[tag]
        clips = len(b["sessions"])
        audio = b["audio_ms"]
        table.append(
            {
                "run": tag,
                "clips": clips,
                "utterances": b["utterances"],
                "utt_per_clip": f"{b['utterances'] / clips:.2f}" if clips else "-",
                "empty_asr": b["empty"],
                "empty_pct": f"{100 * b['empty'] / b['finals']:.1f}" if b["finals"] else "-",
                "max_utt_cuts": b["max_cut"],
                "utt_ms_p50/p90": (
                    f"{np.median(audio):.0f}/{np.percentile(audio, 90):.0f}" if audio else "-"
                ),
            }
        )
    return table


def _read_lines(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as fh:
            yield from fh


def cmd_logstats(args: argparse.Namespace) -> None:
    missing = [str(p) for p in args.logs if not p.is_file()]
    if missing:
        print(f"missing log file(s): {', '.join(missing)}", file=sys.stderr)
    table = log_stats(_read_lines(p for p in args.logs if p.is_file()), args.tags)
    print("Utterances per run from engine trace lines (empty_pct = empty ASR finals / ASR finals)")
    _print_table(table)
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"\nwrote {args.csv}")


# ── cli ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download the dataset, write wavs and vi.tsv")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=0, help="0 = all 200 clips")
    prepare.set_defaults(func=cmd_prepare)

    mix = sub.add_parser("mix", help="interleave vi clips with one English wav")
    mix.add_argument("--manifest", type=Path, required=True)
    mix.add_argument("--en-wav", type=Path, required=True)
    mix.add_argument("--count", type=int, default=20, help="number of vi/en pairs")
    mix.add_argument("--out", type=Path, required=True)
    mix.set_defaults(func=cmd_mix)

    run = sub.add_parser("run", help="stream the manifest to the engine")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--address", default="127.0.0.1:50052")
    run.add_argument("--language", default="", help='AudioChunk.language; "" or auto = not sent')
    run.add_argument("--language-source", default="", help='AudioChunk.language_source; "" = not sent, @lang = per row')
    run.add_argument("--task", default="speech_to_text", choices=("speech_to_text", "realtime_translation"))
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--warmup", type=int, default=0, help="clips streamed first and discarded")
    run.add_argument("--chunk-ms", type=int, default=80, help="rtpgw sends 80")
    run.add_argument("--rtp-clock-rate", type=int, default=8000, help="rtpgw uses 8000")
    run.add_argument("--narrowband", action=argparse.BooleanOptionalAction, default=True, help="8 kHz G.711 round trip like rtpgw")
    run.add_argument("--pace", action="store_true", help="real-time pacing (default: as fast as the engine accepts)")
    run.add_argument("--timeout-s", type=float, default=300.0)
    run.add_argument("--tag", default="eval")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="WER of runs, paired against --base")
    score.add_argument("--base", type=Path, required=True)
    score.add_argument("runs", type=Path, nargs="*")
    score.add_argument("--csv", type=Path, default=None)
    score.set_defaults(func=cmd_score)

    show = sub.add_parser("show", help="print hypotheses of a run")
    show.add_argument("run", type=Path)
    show.add_argument("-n", type=int, default=5)
    show.set_defaults(func=cmd_show)

    diff = sub.add_parser("diff", help="clips whose word errors changed most between --base and a run")
    diff.add_argument("--base", type=Path, required=True)
    diff.add_argument("run", type=Path)
    diff.add_argument("-n", type=int, default=5, help="clips listed per direction (worse / better)")
    diff.set_defaults(func=cmd_diff)

    logstats = sub.add_parser("logstats", help="utterances, empty ASR finals and length per run tag, from engine logs")
    logstats.add_argument("logs", type=Path, nargs="+", help="engine log files (JSON lines)")
    logstats.add_argument("--tags", nargs="+", required=True, help="run tags = session id prefixes (run --tag)")
    logstats.add_argument("--csv", type=Path, default=None)
    logstats.set_defaults(func=cmd_logstats)
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
