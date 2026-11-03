#!/usr/bin/env python3
"""Sum the audio_preprocessing, VAD and ASR blocks of the engine's periodic STATS lines over a time range.

Prometheus counters restart at 0 with the engine and only give totals since then. The STATS log line (every
stats.log_interval_s, 10 s in the shipped yaml) holds the deltas of its interval, so the log covers any period:
the hour after turning model audx on, a night, or two windows to compare (off vs on). Read-only.

Reported for the selected STATS lines:
  * lines read, first and last timestamp, how many carried an audio_preprocessing block (pool running)
  * audio_preprocessing: regions by outcome and the share played raw (late + shed + error + bypass), fallback
    reasons (AUDX_* for model audx, DENOISER_* for resemble), state resets by reason (audx), stream bypasses,
    streams shed by max_streams, peak and mean preprocessed streams
  * job / queue wait / slack / late / flush times: STATS only keeps a mean per interval, so they are averaged
    weighted by the interval's region count (an estimate; percentiles need Prometheus)
  * VAD utterances, empty ASR finals and short utterances with their share; ASR finals (ingress block) and mean
    ASR confidence weighted by finals; loudest ingress sample (indirect signals: not split by preprocessed or not)
  * GPU memory peak and mean utilisation per device

Exit code: 0 ok; 1 no log file, no STATS line in the range, or no audio_preprocessing block (pool not running);
2 the raw share is above --max-raw-pct or there were more state resets than --max-resets.

Examples:
  python scripts/check_preproc_stats.py /u01/ims/ai-engine/logs/nemotron-engine.log --last-minutes 60
  python scripts/check_preproc_stats.py 'logs/nemotron-engine.log*' --since 2026-10-31T08:00 --until 2026-10-31T09:00
  python scripts/check_preproc_stats.py logs/nemotron-engine.log --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_log import expand_paths, parse_time, read_stats, select  # noqa: E402,F401  (shared with stats_phases.py)

RAW_OUTCOMES = ("late", "shed", "error", "bypass")
TIMES = ("window_mean_ms", "queue_wait_mean_ms", "slack_mean_ms", "late_mean_ms", "flush_mean_ms")


@dataclass
class Summary:
    lines: int = 0
    first: datetime | None = None
    last: datetime | None = None
    preproc_lines: int = 0
    regions: Counter = field(default_factory=Counter)
    fallback: Counter = field(default_factory=Counter)
    resets: Counter = field(default_factory=Counter)
    bypass: int = 0
    shed_streams: int = 0
    streams_peak: int = 0
    streams_sum: int = 0
    times: dict = field(default_factory=dict)  # name -> [weighted sum, weight]
    vad_utterances: int = 0
    vad_empty: int = 0
    vad_short: int = 0
    finals: int = 0
    confidence: list = field(default_factory=lambda: [0.0, 0])
    peak_dbfs: float | None = None  # loudest ingress sample over the range
    gpu: dict = field(default_factory=dict)  # device -> {"mem_peak": , "util_sum": , "n": }

    @property
    def regions_total(self) -> int:
        return sum(self.regions.values())

    @property
    def raw_regions(self) -> int:
        return sum(self.regions[o] for o in RAW_OUTCOMES)

    @property
    def raw_pct(self) -> float | None:
        return 100.0 * self.raw_regions / self.regions_total if self.regions_total else None

    def mean_time(self, name: str) -> float | None:
        total, weight = self.times.get(name, (0.0, 0))
        return total / weight if weight else None


def summarise(rows: list[tuple[datetime, dict]]) -> Summary:
    out = Summary(lines=len(rows))
    if rows:
        out.first, out.last = rows[0][0], rows[-1][0]
    for _, stats in rows:
        pre = stats.get("audio_preprocessing")
        if isinstance(pre, dict):
            out.preproc_lines += 1
            regions = {k: int(v) for k, v in (pre.get("regions") or {}).items()}
            out.regions.update(regions)
            out.fallback.update({k: int(v) for k, v in (pre.get("fallback") or {}).items()})
            out.resets.update({k: int(v) for k, v in (pre.get("state_resets") or {}).items()})
            out.bypass += int(pre.get("bypass") or 0)
            out.shed_streams += int(pre.get("shed_streams") or 0)
            streams = int(pre.get("streams") or 0)
            out.streams_peak = max(out.streams_peak, streams)
            out.streams_sum += streams
            weight = sum(regions.values()) or 1
            for name in TIMES:
                value = pre.get(name)
                if value is not None:
                    total, w = out.times.get(name, (0.0, 0))
                    out.times[name] = (total + float(value) * weight, w + weight)
        vad = stats.get("vad") or {}
        out.vad_utterances += int(vad.get("utterances") or 0)
        out.vad_empty += int(vad.get("empty_asr") or 0)
        out.vad_short += int(vad.get("short") or 0)
        ingress = stats.get("ingress") or {}
        finals = int(ingress.get("finals") or 0)  # ASR finals are counted in the ingress block
        out.finals += finals
        if ingress.get("peak_dbfs") is not None:
            peak = float(ingress["peak_dbfs"])
            out.peak_dbfs = peak if out.peak_dbfs is None else max(out.peak_dbfs, peak)
        confidence = (stats.get("asr") or {}).get("confidence_mean")
        if confidence is not None:
            weight = finals or 1
            out.confidence[0] += float(confidence) * weight
            out.confidence[1] += weight
        for device, g in (stats.get("gpu") or {}).items():
            if not isinstance(g, dict):
                continue
            slot = out.gpu.setdefault(device, {"mem_peak": 0.0, "util_sum": 0.0, "n": 0})
            slot["mem_peak"] = max(slot["mem_peak"], float(g.get("mem_used_mb") or 0.0))
            util = g.get("util_pct")
            if util is not None and float(util) >= 0:
                slot["util_sum"] += float(util)
                slot["n"] += 1
    return out


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} ms"


def to_dict(s: Summary) -> dict:
    return {
        "lines": s.lines,
        "first": s.first.isoformat() if s.first else None,
        "last": s.last.isoformat() if s.last else None,
        "preprocessing_lines": s.preproc_lines,
        "regions": dict(s.regions),
        "raw_regions": s.raw_regions,
        "raw_pct": s.raw_pct,
        "fallback": dict(s.fallback),
        "state_resets": dict(s.resets),
        "bypass": s.bypass,
        "shed_streams": s.shed_streams,
        "streams_peak": s.streams_peak,
        "streams_mean": s.streams_sum / s.preproc_lines if s.preproc_lines else None,
        "times_ms": {name: s.mean_time(name) for name in TIMES},
        "vad": {"utterances": s.vad_utterances, "empty_asr": s.vad_empty, "short": s.vad_short},
        "asr": {"finals": s.finals, "confidence_mean": s.confidence[0] / s.confidence[1] if s.confidence[1] else None},
        "ingress_peak_dbfs": s.peak_dbfs,
        "gpu": {d: {"mem_peak_mb": g["mem_peak"], "util_mean_pct": g["util_sum"] / g["n"] if g["n"] else None}
                for d, g in s.gpu.items()},
    }


def render(s: Summary, files: list[str]) -> str:
    lines = [
        f"STATS lines: {s.lines} from {len(files)} file(s), {s.first.isoformat() if s.first else '-'} .. "
        f"{s.last.isoformat() if s.last else '-'}",
        f"with an audio_preprocessing block (pool running): {s.preproc_lines}",
    ]
    if s.preproc_lines:
        regions = ", ".join(f"{k} {v}" for k, v in sorted(s.regions.items(), key=lambda kv: -kv[1])) or "none"
        lines += [
            "",
            "audio_preprocessing",
            f"  regions: {regions}",
            f"  played raw (late+shed+error+bypass): {s.raw_regions} of {s.regions_total}"
            f" ({'-' if s.raw_pct is None else f'{s.raw_pct:.2f}%'})",
            f"  fallback reasons: {dict(s.fallback) or 'none'}",
            f"  state resets (audx): {dict(s.resets) or 'none'}",
            f"  stream bypasses: {s.bypass}; streams shed by max_streams: {s.shed_streams}",
            f"  preprocessed streams: peak {s.streams_peak}, mean {s.streams_sum / s.preproc_lines:.1f}",
            "  times (interval means weighted by regions): "
            + ", ".join(f"{name.replace('_mean_ms', '')} {fmt_ms(s.mean_time(name))}" for name in TIMES),
        ]
    conf = s.confidence[0] / s.confidence[1] if s.confidence[1] else None
    lines += [
        "",
        "VAD / ASR (all streams, not split by preprocessing)",
        f"  utterances {s.vad_utterances}; empty ASR finals {s.vad_empty} ({pct(s.vad_empty, s.vad_utterances)});"
        f" short {s.vad_short} ({pct(s.vad_short, s.vad_utterances)})",
        f"  ASR finals {s.finals}; confidence mean {'-' if conf is None else f'{conf:.3f}'} (weighted by finals)",
        f"  ingress peak {'-' if s.peak_dbfs is None else f'{s.peak_dbfs:.1f} dBFS'} (loudest sample, not the speech level)",
    ]
    for device, g in sorted(s.gpu.items()):
        util = g["util_sum"] / g["n"] if g["n"] else None
        lines.append(f"  gpu {device}: memory peak {g['mem_peak']:.0f} MB, util mean "
                     f"{'-' if util is None else f'{util:.0f}%'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--since", type=parse_time, default=None, help="ISO time, UTC when no zone is given")
    parser.add_argument("--until", type=parse_time, default=None, help="ISO time, UTC when no zone is given")
    parser.add_argument("--last-minutes", type=float, default=None, help="only the last N minutes of STATS lines")
    parser.add_argument("--max-raw-pct", type=float, default=1.0, help="exit 2 above this share of raw regions")
    parser.add_argument("--max-resets", type=int, default=0, help="exit 2 above this many state resets")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = expand_paths(args.logfiles)
    if not files:
        print(f"No log file matches: {' '.join(args.logfiles)}", file=sys.stderr)
        return 1
    rows = select(read_stats(files), args.since, args.until, args.last_minutes)
    if not rows:
        print("No STATS line in the range (stats.log_interval_s = 0 disables them).", file=sys.stderr)
        return 1
    summary = summarise(rows)
    print(json.dumps(to_dict(summary), indent=1, ensure_ascii=False) if args.json else render(summary, files))
    if not summary.preproc_lines:
        print("\nNo audio_preprocessing block: the preprocessing pool was not running in this range.", file=sys.stderr)
        return 1
    problems = []
    if summary.raw_pct is not None and summary.raw_pct > args.max_raw_pct:
        problems.append(f"raw share {summary.raw_pct:.2f}% > --max-raw-pct {args.max_raw_pct}")
    if sum(summary.resets.values()) > args.max_resets:
        problems.append(f"state resets {sum(summary.resets.values())} > --max-resets {args.max_resets}")
    if problems:
        print("\nCHECK FAILED: " + "; ".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
