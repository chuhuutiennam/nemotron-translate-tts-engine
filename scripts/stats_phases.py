#!/usr/bin/env python3
"""Analyse the engine's periodic STATS lines phase by phase (one STATS block = one phase of the pipeline).

Phases, in pipeline order: active, queues, ingress, audio_preprocessing, vad, asr, pipeline, rtp, errors, gpu
(a block the engine adds later is shown after them). Each STATS line holds the deltas of its interval, so the
log covers any period. Read-only.

How a field is combined over the selected lines, from its kind:
  counter      (deltas: chunks, finals, regions.*, ...)             sum
  *_mean*      (STATS keeps one mean per interval)                   mean of the intervals that have one, with
                                                                      min and max; not weighted (the counts are
                                                                      not in STATS), so a few slow intervals weigh
                                                                      as much as busy ones
  {mean_ms, n} (asr.model_step.*)                                    mean weighted by n, and the sum of n
  gauge        (active.*, queues.*, gpu.*, audio_preprocessing.streams / queue)   max and mean
  peak_dbfs                                                          max
A few ratios are derived per phase (realtime audio load, empty ASR share, degraded share, raw share...).

Modes:
  summary (default)  every phase (or --phase a,b) over the range
  --series PHASE     one row per --bucket-minutes bucket (clock aligned) for one phase, to see trends;
                     gauges show the bucket max; --fields picks columns (default: the phase's direct fields)
  --list             phases and fields found in the selected lines

Exit code: 0 ok; 1 no log file or no STATS line in the range; 2 bad arguments (unknown phase or field).

Examples:
  python scripts/stats_phases.py /u01/ims/ai-engine/logs/nemotron-engine.log --last-minutes 60
  python scripts/stats_phases.py 'logs/nemotron-engine.log*' --since 2026-10-31T08:00 --phase vad,asr
  python scripts/stats_phases.py logs/nemotron-engine.log --series asr --bucket-minutes 5 \\
      --fields first_partial_mean_ms,finalization_mean_ms,backlog_soft
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_log import expand_paths, parse_time, read_stats, select  # noqa: E402

PHASE_ORDER = ("active", "queues", "ingress", "audio_preprocessing", "vad", "asr", "pipeline", "rtp", "errors", "gpu")
GAUGE_PHASES = {"active", "queues", "gpu"}
GAUGE_FIELDS = {("audio_preprocessing", "streams"), ("audio_preprocessing", "queue")}
MAX_FIELDS = {"peak_dbfs"}


def is_mean(name: str) -> bool:
    return "_mean" in name or name.startswith("mean_")


@dataclass
class Field:
    kind: str  # sum | mean | gauge | max | wmean
    total: float = 0.0
    values: list = field(default_factory=list)
    weight: float = 0.0

    def add(self, value: float, weight: float = 1.0) -> None:
        if self.kind == "sum":
            self.total += value
        elif self.kind == "wmean":
            self.total += value * weight
            self.weight += weight
        else:
            self.values.append(value)

    def value(self) -> float | None:
        """One number for the field: sum, mean, gauge max, max, or weighted mean."""
        if self.kind == "sum":
            return self.total
        if self.kind == "wmean":
            return self.total / self.weight if self.weight else None
        if not self.values:
            return None
        if self.kind == "mean":
            return sum(self.values) / len(self.values)
        return max(self.values)

    def describe(self) -> str:
        if self.kind == "sum":
            return f"sum {num(self.total)}"
        if self.kind == "wmean":
            return f"mean {num(self.value())} (weighted by n, n = {num(self.weight)})"
        if not self.values:
            return "-"
        if self.kind == "mean":
            return (f"mean {num(self.value())} (min {num(min(self.values))}, max {num(max(self.values))}, "
                    f"{len(self.values)} intervals)")
        if self.kind == "gauge":
            return f"max {num(max(self.values))}, mean {num(sum(self.values) / len(self.values))}"
        return f"max {num(max(self.values))}"


def num(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"


def kind_of(phase: str, path: tuple[str, ...]) -> str:
    if phase in GAUGE_PHASES or (phase, path[0]) in GAUGE_FIELDS:
        return "gauge"
    if path[-1] in MAX_FIELDS:
        return "max"
    if any(is_mean(part) for part in path):
        return "mean"
    return "sum"


def walk(phase: str, node: dict, path: tuple[str, ...], out: dict[tuple[str, ...], Field]) -> None:
    for key, value in node.items():
        sub = path + (str(key),)
        if isinstance(value, dict):
            if set(value) == {"mean_ms", "n"}:  # asr.model_step.<command>
                f = out.setdefault(sub, Field("wmean"))
                if value.get("mean_ms") is not None and value.get("n"):
                    f.add(float(value["mean_ms"]), float(value["n"]))
            else:
                walk(phase, value, sub, out)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            f = out.setdefault(sub, Field(kind_of(phase, sub)))
            if not (isinstance(value, float) and math.isnan(value)):
                if f.kind == "gauge" and phase == "gpu" and sub[-1] == "util_pct" and value < 0:
                    continue  # -1 = utilisation unknown (NVML blocked)
                f.add(float(value))
        elif value is None:
            out.setdefault(sub, Field(kind_of(phase, sub)))


def aggregate(rows: list[tuple[datetime, dict]]) -> tuple[dict[str, dict[tuple[str, ...], Field]], dict[str, int]]:
    """(phase -> field path -> Field, phase -> number of lines that carried it)."""
    phases: dict[str, dict[tuple[str, ...], Field]] = {}
    present: dict[str, int] = {}
    for _, stats in rows:
        for phase, block in stats.items():
            if not isinstance(block, dict):
                continue
            present[phase] = present.get(phase, 0) + 1
            walk(phase, block, (), phases.setdefault(phase, {}))
    return phases, present


def ordered(phases) -> list[str]:
    return [p for p in PHASE_ORDER if p in phases] + sorted(p for p in phases if p not in PHASE_ORDER)


def derived(phase: str, fields: dict[tuple[str, ...], Field], wall_s: float) -> list[str]:
    get = lambda *path: fields[path].value() if path in fields else None  # noqa: E731
    share = lambda part, whole: f"{100.0 * part / whole:.1f}%" if part is not None and whole else "-"  # noqa: E731
    lines = []
    if phase == "ingress":
        audio = get("audio_s")
        if audio is not None and wall_s:
            lines.append(f"audio received / wall time: {audio / wall_s:.2f} (= streams of realtime audio on average)")
        partials, finals, stale = get("partials"), get("finals"), get("stale")
        if partials is not None and finals is not None:
            lines.append(f"stale results: {share(stale, (partials or 0) + (finals or 0))} of partials + finals")
    elif phase == "vad":
        utt = get("utterances")
        lines.append(f"empty ASR finals: {share(get('empty_asr'), utt)} of utterances; short: {share(get('short'), utt)}")
        if get("speech_s") is not None and wall_s:
            lines.append(f"speech / wall time: {get('speech_s') / wall_s:.2f}")
    elif phase == "pipeline":
        lines.append(f"degraded: {share(get('degraded'), get('results'))} of results")
    elif phase == "audio_preprocessing":
        regions = {p[1]: f.value() or 0 for p, f in fields.items() if p[0] == "regions"}
        total = sum(regions.values())
        raw = sum(regions.get(o, 0) for o in ("late", "shed", "error", "bypass"))
        lines.append(f"played raw (late+shed+error+bypass): {num(raw)} of {num(total)} regions ({share(raw, total)})")
    elif phase == "errors":
        lines.append(f"all errors: {num(sum(f.value() or 0 for f in fields.values()))}")
    return lines


def render_summary(phases, present, wanted, rows, files) -> str:
    wall_s = sum(float(s.get("interval_s") or 0) for _, s in rows)
    out = [f"STATS lines: {len(rows)} from {len(files)} file(s), {rows[0][0].isoformat()} .. {rows[-1][0].isoformat()}"
           f" (wall {wall_s / 60:.1f} min)"]
    for phase in wanted:
        fields = phases.get(phase, {})
        out.append(f"\n== {phase} ({present.get(phase, 0)} of {len(rows)} lines)")
        if not fields:
            out.append("  (not in the selected lines)")
            continue
        width = max(len(".".join(p)) for p in fields)
        for path in sorted(fields, key=lambda p: (len(p) > 1, p)):
            out.append(f"  {'.'.join(path):<{width}}  {fields[path].describe()}")
        for line in derived(phase, fields, wall_s):
            out.append(f"  -> {line}")
    return "\n".join(out)


def bucket_of(t: datetime, minutes: float) -> datetime:
    size = minutes * 60.0
    return datetime.fromtimestamp(math.floor(t.timestamp() / size) * size, tz=timezone.utc)


def render_series(rows, phase: str, minutes: float, wanted_fields: list[str] | None) -> str:
    buckets: dict[datetime, list] = {}
    for t, s in rows:
        buckets.setdefault(bucket_of(t, minutes), []).append((t, s))
    per_bucket = {b: aggregate(r)[0].get(phase, {}) for b, r in buckets.items()}
    all_paths = sorted({p for fields in per_bucket.values() for p in fields}, key=lambda p: (len(p) > 1, p))
    if wanted_fields:
        names = {".".join(p): p for p in all_paths}
        unknown = [f for f in wanted_fields if f not in names]
        if unknown:
            raise SystemExit(f"unknown field(s) for {phase}: {unknown}; known: {sorted(names)}")
        paths = [names[f] for f in wanted_fields]
    else:
        # direct fields of the phase; a phase made only of nested ones (gpu.<device>.*, queues.<pool>.*) shows them all
        paths = [p for p in all_paths if len(p) == 1] or all_paths
    header = ["bucket (UTC)", "lines"] + [".".join(p) for p in paths]
    table = [header]
    for b in sorted(per_bucket):
        fields = per_bucket[b]
        table.append([b.strftime("%Y-%m-%d %H:%M"), str(len(buckets[b]))]
                     + [num(fields[p].value()) if p in fields else "-" for p in paths])
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = [f"== {phase}, buckets of {num(minutes)} min (counters: sum; means: mean of intervals; gauges: max)"]
    for row in table:
        lines.append("  ".join(cell.rjust(w) if i else cell.ljust(w) for i, (cell, w) in enumerate(zip(row, widths))))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--since", type=parse_time, default=None, help="ISO time, UTC when no zone is given")
    parser.add_argument("--until", type=parse_time, default=None, help="ISO time, UTC when no zone is given")
    parser.add_argument("--last-minutes", type=float, default=None, help="only the last N minutes of STATS lines")
    parser.add_argument("--phase", default=None, help="comma-separated phases for the summary (default: all)")
    parser.add_argument("--series", metavar="PHASE", default=None, help="time series of one phase instead")
    parser.add_argument("--bucket-minutes", type=float, default=5.0, help="series bucket size (default 5)")
    parser.add_argument("--fields", default=None, help="series columns, comma-separated (see --list)")
    parser.add_argument("--list", action="store_true", help="list phases and fields found, then exit")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.bucket_minutes <= 0:
        parser.error("--bucket-minutes must be > 0")

    files = expand_paths(args.logfiles)
    if not files:
        print(f"No log file matches: {' '.join(args.logfiles)}", file=sys.stderr)
        return 1
    rows = select(read_stats(files), args.since, args.until, args.last_minutes)
    if not rows:
        print("No STATS line in the range (stats.log_interval_s = 0 disables them).", file=sys.stderr)
        return 1
    phases, present = aggregate(rows)

    if args.list:
        for phase in ordered(phases):
            fields = ", ".join(sorted(".".join(p) for p in phases[phase]))
            print(f"{phase} ({present[phase]} lines): {fields}")
        return 0
    if args.series:
        if args.series not in phases:
            print(f"unknown phase {args.series!r}; found: {', '.join(ordered(phases))}", file=sys.stderr)
            return 2
        wanted_fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else None
        try:
            print(render_series(rows, args.series, args.bucket_minutes, wanted_fields))
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 2
        return 0
    wanted = ordered(phases)
    if args.phase:
        wanted = [p.strip() for p in args.phase.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in phases and p not in PHASE_ORDER]
        if unknown:
            print(f"unknown phase(s) {unknown}; found: {', '.join(ordered(phases))}", file=sys.stderr)
            return 2
    print(render_summary(phases, present, wanted, rows, files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
