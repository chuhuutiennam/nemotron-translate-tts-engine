#!/usr/bin/env python3
"""Check rtp_timestamp / rtp_clock_rate / chunk_seq consistency from engine DEBUG logs.

The engine logs one `GRPC_IN ...` line per received AudioChunk when the log level is
DEBUG. This script replays the engine's RTP continuity policy over those lines, per
(rpc, session, stream), and reports where the sender's timeline does not match what
the engine expects (gaps that make it insert silence, out-of-order chunks, a wrong
rtp_clock_rate, chunk_seq problems).

Rule checked for every pair of consecutive chunks:
  rtp_timestamp[n+1] - rtp_timestamp[n] == payload_duration_ms * rtp_clock_rate / 1000

Examples:
  python scripts/check_rtp_log.py logs/nemotron-engine.log
  python scripts/check_rtp_log.py logs/nemotron-engine.log --stream-id MO --max-issues 30
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from typing import Iterable

_FIELD_RE = re.compile(r"(\w+)=(\S*)")
_MASK = 0xFFFFFFFF
_STANDARD_CLOCKS = (8000, 16000, 24000, 32000, 44100, 48000, 90000)
_ERROR_KINDS = ("clock_mismatch", "gap_lost", "out_of_order", "param_change")


def expand_paths(paths: Iterable[str]) -> list[str]:
    """Expand globs and include date-rotated siblings (`<file>.YYYY-MM-DD`), oldest first."""
    found: list[str] = []
    for path in paths:
        base = glob.glob(path)
        rotated = sorted(glob.glob(path + ".*"))
        for candidate in [*rotated, *base]:
            if candidate not in found:
                found.append(candidate)
    return found


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if line.startswith("{"):
        try:
            message = str(json.loads(line).get("message", ""))
        except json.JSONDecodeError:
            return None
    else:
        message = line
    marker = message.find("GRPC_IN ")
    if marker < 0:
        return None
    fields = dict(_FIELD_RE.findall(message[marker + len("GRPC_IN "):]))
    try:
        return {
            "rpc": fields["rpc"],
            "session": fields["session"],
            "stream": fields["stream"],
            "pcm_bytes": int(fields["pcm_bytes"]),
            "sample_rate": int(fields["sample_rate"]),
            "channels": int(fields["channels"]),
            "rtp_timestamp": int(fields["rtp_timestamp"]),
            "rtp_clock_rate": int(fields["rtp_clock_rate"]),
            "chunk_seq": int(fields["chunk_seq"]),
            "duration_ms": float(fields["duration_ms"]),
        }
    except (KeyError, ValueError):
        return None


def scan_chunks(paths: Iterable[str]) -> tuple[list[dict], dict]:
    """Load chunks and return (chunks, stats) so callers can explain an empty result."""
    files = expand_paths(paths)
    stats = {"files": files, "grpc_in_lines": 0, "chunks": 0}
    chunks: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "GRPC_IN" in line:
                    stats["grpc_in_lines"] += 1
                chunk = parse_line(line)
                if chunk is not None:
                    chunks.append(chunk)
    stats["chunks"] = len(chunks)
    return chunks, stats


def load_chunks(paths: Iterable[str]) -> list[dict]:
    return scan_chunks(paths)[0]


def explain_empty(paths: list[str], stats: dict, chunks: list[dict]) -> str:
    """Return the reason nothing was analysed: missing file, no usable GRPC_IN lines, or filters excluded all."""
    if not stats["files"]:
        return (
            f"No log file matches: {' '.join(paths)}\n"
            "Check the path (it is relative to the current directory) or pass a glob such as 'logs/nemotron-engine.log*'."
        )
    if not chunks:
        if stats["grpc_in_lines"]:
            return (
                f"{stats['grpc_in_lines']} line(s) contain 'GRPC_IN' but none has the expected fields "
                "(rpc, session, stream, pcm_bytes, sample_rate, channels, rtp_timestamp, rtp_clock_rate, chunk_seq, duration_ms).\n"
                "The running build may log a different GRPC_IN format."
            )
        return (
            f"No GRPC_IN lines in {len(stats['files'])} file(s). "
            "The engine logs them only at DEBUG level: set logging.level=DEBUG (or LOG_LEVEL=DEBUG) and restart."
        )
    sessions = sorted({c["session"] for c in chunks})
    streams = sorted({c["stream"] for c in chunks})
    return (
        f"{len(chunks)} GRPC_IN chunk(s) found, but none match the filters.\n"
        f"Available session: {', '.join(sessions[:10])}{' ...' if len(sessions) > 10 else ''}\n"
        f"Available stream: {', '.join(streams)}"
    )


def filter_chunks(
    chunks: Iterable[dict],
    *,
    stream_id: str | None = None,
    session_id: str | None = None,
    rpc_id: str | None = None,
) -> list[dict]:
    return [
        c
        for c in chunks
        if (stream_id is None or c["stream"] == stream_id)
        and (session_id is None or c["session"] == session_id)
        and (rpc_id is None or c["rpc"] == rpc_id)
    ]


def group_chunks(chunks: Iterable[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk["rpc"], chunk["session"], chunk["stream"]), []).append(chunk)
    return grouped


def _signed_diff(a: int, b: int) -> int:
    raw = (a - b) & _MASK
    return raw if raw < 0x80000000 else raw - 0x100000000


def _nearest_standard_clock(value: float) -> int | None:
    best = min(_STANDARD_CLOCKS, key=lambda c: abs(c - value))
    return best if abs(best - value) / best <= 0.05 else None


def analyze_stream(
    chunks: list[dict],
    *,
    max_gap_ms: float = 200.0,
    jitter_ms: float = 20.0,
    default_clock: int = 8000,
) -> dict:
    issues: list[dict] = []
    counts: dict[str, int] = {}

    def flag(kind: str, index: int, detail: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        issues.append({"index": index, "kind": kind, "detail": detail})

    declared = chunks[0]["rtp_clock_rate"]
    clock = declared or default_clock
    if not declared:
        counts["clock_unset"] = 1

    expected_ts: int | None = None
    expected_seq: int | None = None
    inserted_ms = 0.0
    lost_ms = 0.0
    ratios: list[float] = []
    durations: list[float] = []
    prev: dict | None = None
    prev_ms = 0.0

    for index, chunk in enumerate(chunks):
        samples = chunk["pcm_bytes"] // (2 * max(1, chunk["channels"]))
        actual_ms = samples * 1000.0 / chunk["sample_rate"] if chunk["sample_rate"] > 0 else 0.0
        durations.append(actual_ms)

        if prev is not None:
            for name in ("sample_rate", "channels", "rtp_clock_rate"):
                if chunk[name] != prev[name]:
                    flag("param_change", index, f"{name} changed {prev[name]} -> {chunk[name]}")
            delta = _signed_diff(chunk["rtp_timestamp"], prev["rtp_timestamp"])
            if delta > 0 and prev_ms > 0:
                ratios.append(delta / (prev_ms / 1000.0))

        provided = chunk["duration_ms"]
        if provided > 0 and abs(provided - actual_ms) > max(5.0, actual_ms * 0.10):
            flag(
                "duration_mismatch",
                index,
                f"duration_ms={provided:g} but pcm_bytes imply {actual_ms:.1f} ms",
            )

        ts = chunk["rtp_timestamp"]
        dur_ticks = int(round(actual_ms * clock / 1000))
        if expected_ts is None:
            expected_ts = (ts + dur_ticks) & _MASK
        else:
            diff = _signed_diff(ts, expected_ts)
            if diff == 0:
                expected_ts = (ts + dur_ticks) & _MASK
            elif diff > 0:
                gap_ms = diff * 1000.0 / clock
                detail = f"ts={ts} expected={expected_ts} diff=+{diff} ticks (+{gap_ms:.1f} ms)"
                if gap_ms <= max_gap_ms:
                    inserted_ms += gap_ms
                    flag("gap_inserted", index, detail + " -> engine inserts silence")
                else:
                    lost_ms += gap_ms
                    flag("gap_lost", index, detail + f" > {max_gap_ms:g} ms -> engine drops it as audio loss")
                expected_ts = (ts + dur_ticks) & _MASK
            else:
                jitter_ticks = int(round(jitter_ms * clock / 1000))
                detail = f"ts={ts} expected={expected_ts} diff={diff} ticks ({diff * 1000.0 / clock:.1f} ms)"
                if jitter_ticks > 0 and abs(diff) <= jitter_ticks:
                    flag("jitter", index, detail + " -> accepted within jitter tolerance")
                else:
                    flag("out_of_order", index, detail + " -> engine warns (policy decides drop/process)")

        seq = chunk["chunk_seq"]
        if seq >= 0:
            if expected_seq is None or seq == expected_seq:
                expected_seq = seq + 1
            elif seq > expected_seq:
                flag("seq_gap", index, f"chunk_seq={seq} expected={expected_seq} ({seq - expected_seq} missing)")
                expected_seq = seq + 1
            else:
                flag("seq_backwards", index, f"chunk_seq={seq} expected={expected_seq}")

        prev = chunk
        prev_ms = actual_ms

    seqs = [c["chunk_seq"] for c in chunks]
    if len(seqs) > 1 and all(s == 0 for s in seqs):
        flag("seq_constant_zero", 0, "chunk_seq is 0 on every chunk (field unset; send -1 or 0,1,2,...)")

    inferred = statistics.median(ratios) if len(ratios) >= 3 else None
    if inferred is not None and abs(inferred - clock) / clock > 0.05:
        suggested = _nearest_standard_clock(inferred)
        hint = f"rtp_clock_rate should be {suggested}" if suggested else f"timestamps advance ~{inferred:.0f} ticks/s"
        flag(
            "clock_mismatch",
            0,
            f"declared/effective rtp_clock_rate={clock}"
            + (" (unset, engine default)" if not declared else "")
            + f" but timestamps advance ~{inferred:.0f} ticks per second of audio: {hint}",
        )

    return {
        "chunks": len(chunks),
        "sample_rate": chunks[0]["sample_rate"],
        "channels": chunks[0]["channels"],
        "declared_clock": declared,
        "effective_clock": clock,
        "inferred_clock": inferred,
        "median_chunk_ms": statistics.median(durations) if durations else 0.0,
        "inserted_silence_ms": inserted_ms,
        "lost_ms": lost_ms,
        "counts": counts,
        "issues": issues,
    }


def verdict(analysis: dict) -> str:
    counts = analysis["counts"]
    if not counts:
        return "OK"
    errors = [k for k in _ERROR_KINDS if k in counts]
    label = "PROBLEMS" if errors else "WARN"
    return f"{label}: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def render(grouped: dict[tuple, list[dict]], *, max_issues: int = 10, **policy) -> str:
    lines: list[str] = []
    for (rpc, session, stream), chunks in grouped.items():
        analysis = analyze_stream(chunks, **policy)
        inferred = analysis["inferred_clock"]
        lines.append(f"== session={session} stream={stream} rpc={rpc} ==")
        lines.append(
            f"chunks={analysis['chunks']} sample_rate={analysis['sample_rate']} "
            f"channels={analysis['channels']} median_chunk_ms={analysis['median_chunk_ms']:.1f} "
            f"declared_rtp_clock={analysis['declared_clock']} "
            f"effective_rtp_clock={analysis['effective_clock']} "
            f"inferred_rtp_clock={'n/a' if inferred is None else f'{inferred:.0f}'}"
        )
        lines.append(f"verdict: {verdict(analysis)}")
        if analysis["inserted_silence_ms"] or analysis["lost_ms"]:
            lines.append(
                f"engine impact: silence inserted {analysis['inserted_silence_ms']:.0f} ms, "
                f"audio-loss gaps {analysis['lost_ms']:.0f} ms"
            )
        issues = analysis["issues"]
        for issue in issues[:max_issues]:
            lines.append(f"  chunk #{issue['index']} [{issue['kind']}] {issue['detail']}")
        if len(issues) > max_issues:
            lines.append(f"  ... {len(issues) - max_issues} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--stream-id")
    parser.add_argument("--session-id")
    parser.add_argument("--rpc-id")
    parser.add_argument("--max-gap-ms", type=float, default=200.0, help="engine rtp.max_gap_ms")
    parser.add_argument("--jitter-ms", type=float, default=20.0, help="engine rtp.jitter_tolerance_ms")
    parser.add_argument("--default-clock", type=int, default=8000, help="clock the engine assumes when rtp_clock_rate=0")
    parser.add_argument("--max-issues", type=int, default=10, help="issues printed per stream")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    all_chunks, stats = scan_chunks(args.logfiles)
    chunks = filter_chunks(
        all_chunks,
        stream_id=args.stream_id,
        session_id=args.session_id,
        rpc_id=args.rpc_id,
    )
    if not chunks:
        print(explain_empty(args.logfiles, stats, all_chunks), file=sys.stderr)
        return 2
    grouped = group_chunks(chunks)
    print(
        render(
            grouped,
            max_issues=args.max_issues,
            max_gap_ms=args.max_gap_ms,
            jitter_ms=args.jitter_ms,
            default_clock=args.default_clock,
        )
    )
    failed = any(
        any(k in analyze_stream(c, max_gap_ms=args.max_gap_ms, jitter_ms=args.jitter_ms, default_clock=args.default_clock)["counts"] for k in _ERROR_KINDS)
        for c in grouped.values()
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
