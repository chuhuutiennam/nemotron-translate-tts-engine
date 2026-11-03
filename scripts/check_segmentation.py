#!/usr/bin/env python3
"""Replay ASR_PARTIAL / ASR_FINAL trace lines through the Stable Punctuation Sentence
Segmentation design and report how often an early commit would have been wrong.

Needs logging.pipeline_trace=true, logging.pipeline_trace_partial=true and
logging.pipeline_trace_text=true on the engine. The script changes nothing in the
engine; it only simulates the commit rule on recorded data.

The commit rule is the engine's own (nemotron_testbed.segmenter), so results match what
segmentation.mode=shadow logs live. A sentence is committed when, in the text after the
already committed part:
  * it ends with . ? ! followed by whitespace or end of text,
  * it was seen unchanged in >= --punct-stable-n consecutive model steps, or for
    >= --punct-stable-ms (0 = off); both default to --stable-n / --stable-ms, which
    always govern the comma fallback,
  * it has >= --min-tokens words and does not end with an unfinished connector word
    (shorter or guarded sentences are merged with the next one).
Whatever is left is committed by fallback: --comma-after-ms (last comma, 0 = off),
--max-segment-ms without a commit, or the utterance end (ASR_FINAL).

Metrics:
  premature  committed sentence whose words are not found in the ASR_FINAL text
  revised    a later partial changed text inside the already committed region
  gain       ASR_FINAL time minus commit time (how much earlier translation could start)

Examples:
  python scripts/check_segmentation.py logs/nemotron-engine.log --session-id wav-9b18aa84
  python scripts/check_segmentation.py logs/nemotron-engine.log --sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from nemotron_testbed.segmenter import (
        SegmenterParams,
        StableSentenceSegmenter,
        find_premature,
        normalize_words,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nemotron_testbed.segmenter import (
        SegmenterParams,
        StableSentenceSegmenter,
        find_premature,
        normalize_words,
    )

Params = SegmenterParams


@dataclass
class Observation:
    step: int
    t_ms: float
    model_step: bool
    text: str


@dataclass
class Commit:
    text: str
    t_ms: float
    step: int
    kind: str  # punct | comma | max_segment | utterance_end
    premature: bool = False
    revised: bool = False


@dataclass
class Utterance:
    key: tuple
    observations: list[Observation] = field(default_factory=list)
    final: dict | None = None


def expand_paths(paths: Iterable[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        base = glob.glob(path)
        rotated = sorted(glob.glob(path + ".*"))
        for candidate in [*rotated, *base]:
            if candidate not in found:
                found.append(candidate)
    return found


def load_utterances(
    paths: Iterable[str],
    *,
    stream_id: str | None = None,
    session_id: str | None = None,
    rpc_id: str | None = None,
) -> tuple[list[Utterance], int, int]:
    """Return (utterances, partial_events_seen, partial_events_without_text)."""
    partials: dict[tuple, list[dict]] = {}
    finals: dict[tuple, dict] = {}
    for path in expand_paths(paths):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"ASR_PARTIAL"' not in line and '"ASR_FINAL"' not in line:
                    continue
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = ev.get("stage")
                if stage not in ("ASR_PARTIAL", "ASR_FINAL"):
                    continue
                if stream_id is not None and ev.get("stream_id") != stream_id:
                    continue
                if session_id is not None and ev.get("session_id") != session_id:
                    continue
                if rpc_id is not None and ev.get("rpc_id") != rpc_id:
                    continue
                key = (ev.get("rpc_id"), ev.get("session_id"), ev.get("stream_id"), ev.get("utterance_id"))
                if stage == "ASR_FINAL":
                    finals[key] = ev
                else:
                    partials.setdefault(key, []).append(ev)

    seen = 0
    no_text = 0
    utterances: list[Utterance] = []
    for key, events in partials.items():
        events.sort(key=lambda e: e.get("step", 0))
        utt = Utterance(key=key, final=finals.get(key))
        text = ""
        for ev in events:
            seen += 1
            if ev.get("changed"):
                if "text" not in ev:
                    no_text += 1
                else:
                    text = ev["text"]
            utt.observations.append(
                Observation(
                    step=int(ev.get("step", 0)),
                    t_ms=float(ev.get("t_ms", 0.0)),
                    model_step=bool(ev.get("model_step", True)),
                    text=text,
                )
            )
        utterances.append(utt)
    utterances.sort(key=lambda u: tuple(str(part) for part in u.key))
    return utterances, seen, no_text


def simulate(utt: Utterance, params: Params) -> list[Commit]:
    """Replay the observations through the engine's StableSentenceSegmenter."""
    final = utt.final or {}
    segmenter = StableSentenceSegmenter(params, str(final.get("source_language", "")))
    commits: list[Commit] = []

    def add(found) -> None:
        commits.extend(Commit(c.text, c.t_ms, c.step, c.kind) for c in found)

    for obs in utt.observations:
        if obs.text:
            add(segmenter.update(obs.text, obs.t_ms, model_step=obs.model_step, step=obs.step).commits)
    if utt.final is not None:
        last = utt.observations[-1] if utt.observations else None
        final_t = float(final.get("t_ms", last.t_ms if last else 0.0))
        add(segmenter.finish(str(final.get("text", "")), final_t, step=last.step if last else 0).commits)
    return commits


def evaluate(utt: Utterance, commits: list[Commit]) -> None:
    """Mark premature / revised on commits (utterance_end commits come from the final text)."""
    judged = [c for c in commits if c.kind != "utterance_end"]
    premature = find_premature([c.text for c in judged], str((utt.final or {}).get("text", "")))
    committed_words: list[str] = []
    ends: list[int] = []
    for commit in judged:
        committed_words.extend(normalize_words(commit.text))
        ends.append(len(committed_words))

    for commit, is_premature, end in zip(judged, premature, ends):
        commit.premature = is_premature
        expected = committed_words[:end]
        for obs in utt.observations:
            if obs.step <= commit.step:
                continue
            if normalize_words(obs.text)[:end] != expected:
                commit.revised = True
                break


def analyse(utterances: list[Utterance], params: Params) -> tuple[list[tuple[Utterance, list[Commit]]], int]:
    results: list[tuple[Utterance, list[Commit]]] = []
    skipped = 0
    for utt in utterances:
        if utt.final is None or not utt.observations:
            skipped += 1
            continue
        commits = simulate(utt, params)
        evaluate(utt, commits)
        results.append((utt, commits))
    return results, skipped


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def summarize(results: list[tuple[Utterance, list[Commit]]]) -> dict:
    punct = [c for _, commits in results for c in commits if c.kind == "punct"]
    fallback = [c for _, commits in results for c in commits if c.kind in ("comma", "max_segment")]
    gains: list[float] = []
    first_gains: list[float] = []
    for utt, commits in results:
        final_t = float((utt.final or {}).get("t_ms", utt.observations[-1].t_ms))
        utt_gains = [final_t - c.t_ms for c in commits if c.kind == "punct"]
        gains.extend(utt_gains)
        if utt_gains:
            first_gains.append(utt_gains[0])
    total = len(punct)
    return {
        "utterances": len(results),
        "punct_commits": total,
        "fallback_commits": len(fallback),
        "premature": sum(c.premature for c in punct),
        "revised": sum(c.revised for c in punct),
        "premature_rate": (sum(c.premature for c in punct) / total) if total else 0.0,
        "revised_rate": (sum(c.revised for c in punct) / total) if total else 0.0,
        "gain_mean_ms": statistics.fmean(gains) if gains else 0.0,
        "gain_p50_ms": _percentile(gains, 50),
        "gain_p95_ms": _percentile(gains, 95),
        "first_gain_mean_ms": statistics.fmean(first_gains) if first_gains else 0.0,
    }


def render_summary(summary: dict, params: Params) -> str:
    return "\n".join(
        [
            f"params: punct_stable_n={params.sentence_stable_n} punct_stable_ms={params.sentence_stable_ms:g} "
            f"stable_n={params.stable_n} stable_ms={params.stable_ms:g} (comma) "
            f"min_tokens={params.min_tokens} max_segment_ms={params.max_segment_ms:g} "
            f"comma_after_ms={params.comma_after_ms:g}",
            f"utterances analysed: {summary['utterances']}",
            f"punctuation commits: {summary['punct_commits']}  fallback (comma/max_segment): {summary['fallback_commits']}",
            f"premature commit rate: {summary['premature_rate'] * 100:.1f}% ({summary['premature']}/{summary['punct_commits']})",
            f"revised after commit : {summary['revised_rate'] * 100:.1f}% ({summary['revised']}/{summary['punct_commits']})",
            f"gain vs ASR_FINAL (ms): mean={summary['gain_mean_ms']:.0f} p50={summary['gain_p50_ms']:.0f} "
            f"p95={summary['gain_p95_ms']:.0f}; first commit per utterance mean={summary['first_gain_mean_ms']:.0f}",
        ]
    )


def render_verbose(results: list[tuple[Utterance, list[Commit]]]) -> str:
    lines: list[str] = []
    for utt, commits in results:
        rpc_id, session_id, stream_id, utterance_id = utt.key
        final_t = float((utt.final or {}).get("t_ms", utt.observations[-1].t_ms))
        lines.append(
            f"== session={session_id} stream={stream_id} rpc={rpc_id} utterance={utterance_id} "
            f"steps={len(utt.observations)} final_t={final_t:.0f}ms =="
        )
        for commit in commits:
            flags = []
            if commit.premature:
                flags.append("PREMATURE")
            if commit.revised:
                flags.append("REVISED")
            status = ",".join(flags) if flags else "ok"
            gain = final_t - commit.t_ms
            lines.append(
                f"  t={commit.t_ms:8.0f}ms step={commit.step:4d} {commit.kind:<13} "
                f"gain={gain:7.0f}ms {status:<18} {commit.text!r}"
            )
    return "\n".join(lines)


_SWEEP_STABLE = (
    ("n=1", {"punct_stable_n": 1, "punct_stable_ms": 0.0}),
    ("n=2", {"punct_stable_n": 2, "punct_stable_ms": 0.0}),
    ("n=3", {"punct_stable_n": 3, "punct_stable_ms": 0.0}),
    ("ms=400", {"punct_stable_n": 10**9, "punct_stable_ms": 400.0}),
    ("ms=800", {"punct_stable_n": 10**9, "punct_stable_ms": 800.0}),
    ("ms=1200", {"punct_stable_n": 10**9, "punct_stable_ms": 1200.0}),
)


def render_sweep(utterances: list[Utterance], base: Params) -> str:
    header = f"{'stable':>8} {'min_tok':>7} {'commits':>7} {'premature%':>10} {'revised%':>9} {'gain_mean':>9} {'gain_p50':>8} {'fallback':>8}"
    lines = [header]
    for label, stable in _SWEEP_STABLE:
        for min_tokens in (1, 4, 6):
            params = dataclasses.replace(base, min_tokens=min_tokens, **stable)
            results, _ = analyse(utterances, params)
            summary = summarize(results)
            lines.append(
                f"{label:>8} {min_tokens:>7} {summary['punct_commits']:>7} "
                f"{summary['premature_rate'] * 100:>9.1f}% {summary['revised_rate'] * 100:>8.1f}% "
                f"{summary['gain_mean_ms']:>9.0f} {summary['gain_p50_ms']:>8.0f} {summary['fallback_commits']:>8}"
            )
    return "\n".join(lines)


def explain_no_partials(paths: list[str], *, filtered: bool) -> str:
    """Explain why no ASR_PARTIAL events were analysed: missing file, none logged, or filters excluded all."""
    files = expand_paths(paths)
    if not files:
        return (
            f"No log file matches: {' '.join(paths)}\n"
            "Check the path (it is relative to the current directory) or pass a glob such as 'logs/nemotron-engine.log*'."
        )
    partial_lines = 0
    pipeline_lines = 0
    sessions: set[str] = set()
    streams: set[str] = set()
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "PIPELINE" in line:
                    pipeline_lines += 1
                if '"ASR_PARTIAL"' not in line:
                    continue
                partial_lines += 1
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("session_id") is not None:
                    sessions.add(str(ev["session_id"]))
                if ev.get("stream_id") is not None:
                    streams.add(str(ev["stream_id"]))
    if partial_lines and filtered:
        return (
            f"{partial_lines} ASR_PARTIAL event(s) found, but none match the filters.\n"
            f"Available session_id: {', '.join(sorted(sessions)[:10]) or '-'}\n"
            f"Available stream_id: {', '.join(sorted(streams)) or '-'}"
        )
    return (
        f"No ASR_PARTIAL events in {len(files)} file(s) ({pipeline_lines} line(s) contain 'PIPELINE').\n"
        "Enable logging.pipeline_trace and logging.pipeline_trace_partial (LOG_PIPELINE_TRACE_PARTIAL=true) and restart. "
        "Builds older than 0.3.23 do not emit ASR_PARTIAL."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--stream-id")
    parser.add_argument("--session-id")
    parser.add_argument("--rpc-id")
    defaults = Params()
    parser.add_argument("--stable-n", type=int, default=defaults.stable_n, help="stable steps; also the default for --punct-stable-n")
    parser.add_argument("--stable-ms", type=float, default=defaults.stable_ms, help="stable ms; also the default for --punct-stable-ms")
    parser.add_argument("--punct-stable-n", type=int, default=None, help="stable steps for . ? ! commits (default: --stable-n)")
    parser.add_argument("--punct-stable-ms", type=float, default=None, help="stable ms for . ? ! commits (default: --stable-ms)")
    parser.add_argument("--min-tokens", type=int, default=defaults.min_tokens)
    parser.add_argument("--max-segment-ms", type=float, default=defaults.max_segment_ms)
    parser.add_argument("--comma-after-ms", type=float, default=defaults.comma_after_ms)
    parser.add_argument("--verbose", action="store_true", help="list every commit")
    parser.add_argument("--sweep", action="store_true", help="compare the punctuation stable rule (n=1..3, ms=400..1200) x min_tokens; keeps --stable-n/--stable-ms (comma), --max-segment-ms, --comma-after-ms")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    utterances, seen, no_text = load_utterances(
        args.logfiles, stream_id=args.stream_id, session_id=args.session_id, rpc_id=args.rpc_id
    )
    if not seen:
        filtered = any(v is not None for v in (args.stream_id, args.session_id, args.rpc_id))
        print(explain_no_partials(args.logfiles, filtered=filtered), file=sys.stderr)
        return 1
    if no_text:
        print(
            "ASR_PARTIAL events have no text (enable logging.pipeline_trace_text).",
            file=sys.stderr,
        )
        return 2

    params = Params(
        stable_n=args.stable_n,
        stable_ms=args.stable_ms,
        min_tokens=args.min_tokens,
        max_segment_ms=args.max_segment_ms,
        comma_after_ms=args.comma_after_ms,
        punct_stable_n=args.punct_stable_n,
        punct_stable_ms=args.punct_stable_ms,
    )
    if args.sweep:
        print(render_sweep(utterances, params))
        return 0
    results, skipped = analyse(utterances, params)
    if not results:
        print(f"No utterance has both ASR_PARTIAL and ASR_FINAL events (skipped {skipped}).", file=sys.stderr)
        return 1
    if args.verbose:
        print(render_verbose(results))
        print()
    print(render_summary(summarize(results), params))
    if skipped:
        print(f"skipped utterances without ASR_FINAL: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
