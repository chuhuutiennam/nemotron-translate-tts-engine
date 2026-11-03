#!/usr/bin/env python3
"""Check that the text sent for segmented utterances adds up to the ASR final: no word repeated or
lost where an utterance was cut into segments.

With segmentation.mode=on an utterance can be sent in several segments before ASR FINAL:
  * speech_to_text: each segment is an EMIT with is_final=true (the last one is the remainder),
  * realtime_translation: each segment is a TRANSLATE_SUBMIT (source_text sent to MT).
This script joins the segments of every segmented utterance (ASR_FINAL committed_tokens > 0) and
aligns their words with the ASR_FINAL text. Differences are classified by where they are:

  CUT_LOSS     words of the final that were never sent, at a cut between segments or after the
               last one (the remainder was cut too far)
  CUT_DUP      words sent twice at a cut (the remainder was cut too early)
  CUT_CHANGED  other differences right at a cut (ASR changed the last committed words, or a cut
               error that is not a clean repeat or loss): read the example
  REVISED      ASR changed already committed text inside a segment (a sent is_final is never re-sent;
               not a cut error)
  OK           the segments add up to the final

Words are compared lower-case, \\w+ only (punctuation and case changes are ignored). Needs
logging.pipeline_trace=true and logging.pipeline_trace_text=true; pipeline_trace_partial is not needed.
Read-only: the engine is not touched.

Examples:
  python scripts/check_segment_output.py logs/nemotron-engine.log
  python scripts/check_segment_output.py 'logs/nemotron-engine.log*' --examples 20
  python scripts/check_segment_output.py logs/nemotron-engine.log --session-id call-1 --verbose --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from nemotron_testbed.segmenter import normalize_words
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nemotron_testbed.segmenter import normalize_words

TASK_STT = "speech_to_text"
TASK_TRANSLATION = "realtime_translation"
STAGES = ("STREAM_OPEN", "ASR_FINAL", "EMIT", "TRANSLATE_SUBMIT", "SEGMENT_ALIGN")

OK = "OK"
REVISED = "REVISED"
CUT_CHANGED = "CUT_CHANGED"
CUT_DUP = "CUT_DUP"
CUT_LOSS = "CUT_LOSS"
STATUS_ORDER = (CUT_LOSS, CUT_DUP, CUT_CHANGED, REVISED, OK)  # worst first
STATUS_HELP = {
    OK: "segments add up to the ASR final",
    REVISED: "ASR changed committed text inside a segment (not a cut error)",
    CUT_CHANGED: "words differ right at a cut: read the example",
    CUT_DUP: "words sent twice at a cut",
    CUT_LOSS: "words of the final never sent",
}


@dataclass
class Utterance:
    key: tuple  # (rpc_id, session_id, stream_id, utterance_id)
    final_text: str | None = None
    committed_tokens: int | None = None
    has_final: bool = False
    stt_finals: list[tuple[int, int | None, str]] = field(default_factory=list)  # (seq, segment_idx, text)
    submits: list[tuple[int, str]] = field(default_factory=list)  # (segment_idx, source_text)
    align: list[str] = field(default_factory=list)  # SEGMENT_ALIGN "phase:how"
    missing_text: bool = False


@dataclass
class Comparison:
    cut_loss: int = 0
    cut_dup: int = 0
    cut_changed: int = 0
    revised: int = 0
    diff: str = ""

    @property
    def status(self) -> str:
        for status, count in (
            (CUT_LOSS, self.cut_loss),
            (CUT_DUP, self.cut_dup),
            (CUT_CHANGED, self.cut_changed),
            (REVISED, self.revised),
        ):
            if count:
                return status
        return OK


@dataclass
class Result:
    utt: Utterance
    task: str
    segments: list[str]
    comparison: Comparison
    duplicate_submits: int = 0


def expand_paths(paths: Iterable[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        base = glob.glob(path)
        rotated = sorted(glob.glob(path + ".*"))
        for candidate in [*rotated, *base]:
            if candidate not in found:
                found.append(candidate)
    return found


def load(
    paths: Iterable[str],
    *,
    stream_id: str | None = None,
    session_id: str | None = None,
    rpc_id: str | None = None,
) -> tuple[dict[tuple, str], dict[tuple, Utterance], int]:
    """(task per stream, utterances, files read) from the engine's JSON trace lines."""
    tasks: dict[tuple, str] = {}
    utterances: dict[tuple, Utterance] = {}
    files = expand_paths(paths)
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not any(f'"{stage}"' in line for stage in STAGES):
                    continue
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = ev.get("stage")
                if stage not in STAGES:
                    continue
                if stream_id is not None and ev.get("stream_id") != stream_id:
                    continue
                if session_id is not None and ev.get("session_id") != session_id:
                    continue
                if rpc_id is not None and ev.get("rpc_id") != rpc_id:
                    continue
                stream = (ev.get("rpc_id"), ev.get("session_id"), ev.get("stream_id"))
                if stage == "STREAM_OPEN":
                    if ev.get("task"):
                        tasks[stream] = str(ev["task"])
                    continue
                key = (*stream, ev.get("utterance_id"))
                utt = utterances.setdefault(key, Utterance(key=key))
                if stage == "ASR_FINAL":
                    utt.has_final = True
                    if "text" in ev:
                        utt.final_text = str(ev["text"])
                    else:
                        utt.missing_text = True
                    if isinstance(ev.get("committed_tokens"), int):
                        utt.committed_tokens = ev["committed_tokens"]
                elif stage == "EMIT":
                    if not ev.get("is_final") or ev.get("skipped"):
                        continue
                    if "text" not in ev:
                        utt.missing_text = True
                        continue
                    if str(ev["text"]).strip():
                        segment_idx = ev.get("segment_idx")
                        utt.stt_finals.append(
                            (int(ev.get("seq", 0)), segment_idx if isinstance(segment_idx, int) else None, str(ev["text"]))
                        )
                elif stage == "TRANSLATE_SUBMIT":
                    if "source_text" not in ev:
                        utt.missing_text = True
                        continue
                    utt.submits.append((int(ev.get("segment_idx") or 0), str(ev["source_text"])))
                elif stage == "SEGMENT_ALIGN":
                    utt.align.append(f"{ev.get('phase', '?')}:{ev.get('how', '?')}")
    return tasks, utterances, len(files)


def utterance_task(utt: Utterance, tasks: dict[tuple, str]) -> str:
    task = tasks.get(utt.key[:3], "")
    if task:
        return task
    # STREAM_OPEN rotated out of the log: an MT submit means translation (its EMITs carry target text)
    return TASK_TRANSLATION if utt.submits else TASK_STT


def is_segmented(utt: Utterance, task: str) -> bool:
    """Text was sent before ASR FINAL: ASR_FINAL committed_tokens > 0 (segmentation.mode=on)."""
    if utt.committed_tokens is not None:
        return utt.committed_tokens > 0
    if task == TASK_TRANSLATION:
        return len({idx for idx, _ in utt.submits}) > 1
    return any(idx is not None for _, idx, _ in utt.stt_finals)


def sent_segments(utt: Utterance, task: str) -> tuple[list[str], int]:
    """(segment texts in order, translation segments submitted more than once)."""
    if task == TASK_TRANSLATION:
        first: dict[int, str] = {}
        for idx, text in utt.submits:
            first.setdefault(idx, text)
        return [first[idx] for idx in sorted(first)], len(utt.submits) - len(first)
    return [text for _, _, text in sorted(utt.stt_finals, key=lambda item: item[0])], 0


def compare(segments: list[str], final_text: str) -> Comparison:
    """Align the words sent with the words of the final and classify each difference by its place."""
    seg_words = [normalize_words(text) for text in segments]
    sent = [word for words in seg_words for word in words]
    final = normalize_words(final_text)
    cuts: set[int] = set()
    position = 0
    for words in seg_words[:-1]:
        position += len(words)
        cuts.add(position)
    cuts.add(len(sent))  # final words after everything sent were never sent either
    result = Comparison()
    parts: list[str] = []
    matcher = difflib.SequenceMatcher(None, sent, final, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.extend(sent[i1:i2])
            continue
        at_cut = any(i1 <= cut <= i2 for cut in cuts)
        if tag == "insert":
            if at_cut:
                result.cut_loss += j2 - j1
            else:
                result.revised += j2 - j1
            parts.append("[-" + " ".join(final[j1:j2]) + "]")
        elif tag == "delete":
            removed = sent[i1:i2]
            n = i2 - i1
            repeat = sent[max(0, i1 - n) : i1] == removed or sent[i2 : i2 + n] == removed
            if at_cut and repeat:
                result.cut_dup += n
            elif at_cut:
                result.cut_changed += n
            else:
                result.revised += n
            parts.append("[+" + " ".join(removed) + "]")
        else:
            n = max(i2 - i1, j2 - j1)
            if at_cut:
                result.cut_changed += n
            else:
                result.revised += n
            parts.append("[" + " ".join(sent[i1:i2]) + " -> " + " ".join(final[j1:j2]) + "]")
    result.diff = " ".join(parts)
    return result


def analyse(tasks: dict[tuple, str], utterances: dict[tuple, Utterance]) -> tuple[list[Result], Counter]:
    counts: Counter = Counter()
    results: list[Result] = []
    for key in sorted(utterances, key=lambda k: tuple(str(part) for part in k)):
        utt = utterances[key]
        if not utt.has_final:
            counts["no_final"] += 1
            continue
        task = utterance_task(utt, tasks)
        counts[f"final_{task}"] += 1
        if not is_segmented(utt, task):
            counts["unsegmented"] += 1
            continue
        if utt.final_text is None:
            counts["no_text"] += 1
            continue
        segments, duplicates = sent_segments(utt, task)
        if not segments:
            counts["nothing_sent"] += 1
            continue
        results.append(Result(utt, task, segments, compare(segments, utt.final_text), duplicates))
    return results, counts


def render_summary(results: list[Result], counts: Counter, files: int) -> str:
    by_task = Counter(r.task for r in results)
    finals = counts[f"final_{TASK_STT}"] + counts[f"final_{TASK_TRANSLATION}"]
    lines = [
        f"files: {files}, utterances with ASR_FINAL: {finals} "
        f"(speech_to_text {counts[f'final_{TASK_STT}']}, realtime_translation {counts[f'final_{TASK_TRANSLATION}']})",
        f"segmented (text sent before ASR_FINAL): {len(results)} "
        f"(speech_to_text {by_task[TASK_STT]}, realtime_translation {by_task[TASK_TRANSLATION]}); "
        f"not segmented {counts['unsegmented']}",
    ]
    skipped = [
        f"{label} {counts[name]}"
        for name, label in (
            ("no_final", "without ASR_FINAL"),
            ("no_text", "without ASR_FINAL text"),
            ("nothing_sent", "segmented but no sent text found"),
        )
        if counts[name]
    ]
    if skipped:
        lines.append("skipped: " + ", ".join(skipped))
    if not results:
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{'status':<12} {'utterances':>10} {'words':>6}  meaning")
    statuses = Counter(r.comparison.status for r in results)
    words = {
        CUT_LOSS: sum(r.comparison.cut_loss for r in results),
        CUT_DUP: sum(r.comparison.cut_dup for r in results),
        CUT_CHANGED: sum(r.comparison.cut_changed for r in results),
        REVISED: sum(r.comparison.revised for r in results),
        OK: 0,
    }
    for status in reversed(STATUS_ORDER):
        word_text = "-" if status == OK else str(words[status])
        lines.append(f"{status:<12} {statuses[status]:>10} {word_text:>6}  {STATUS_HELP[status]}")
    lines.append("(utterances: each counted once, under its worst status; words: every difference of that kind)")
    cut_errors = sum(1 for r in results if r.comparison.cut_loss or r.comparison.cut_dup)
    lines.append("")
    lines.append(
        f"cut errors (any CUT_LOSS or CUT_DUP word): {cut_errors} of {len(results)} segmented utterances "
        f"({100.0 * cut_errors / len(results):.1f}%)"
    )
    aligned = [r for r in results if r.utt.align]
    hows = Counter(entry.split(":", 1)[1] for r in aligned for entry in r.utt.align)
    lines.append(
        f"SEGMENT_ALIGN seen in {len(aligned)} utterance(s)"
        + (f" ({', '.join(f'{how} {n}' for how, n in sorted(hows.items()))})" if hows else "")
        + "; builds before the alignment fix never log it"
    )
    duplicates = sum(r.duplicate_submits for r in results)
    lines.append(f"translation segments submitted more than once: {duplicates}")
    return "\n".join(lines)


def render_result(result: Result) -> str:
    rpc, session, stream, utterance_id = result.utt.key
    lines = [
        f"[{result.comparison.status}] session={session} stream={stream} utterance={utterance_id} "
        f"task={result.task} rpc={rpc} align={','.join(result.utt.align) or '-'}"
    ]
    for index, text in enumerate(result.segments):
        lines.append(f"  sent #{index}: {text}")
    lines.append(f"  final   : {result.utt.final_text}")
    lines.append(f"  diff    : {result.comparison.diff}   ([+x] sent not in final, [-x] in final never sent)")
    return "\n".join(lines)


def render_examples(results: list[Result], limit: int, *, verbose: bool) -> str:
    rank = {status: index for index, status in enumerate(STATUS_ORDER)}
    chosen = results if verbose else [r for r in results if r.comparison.status != OK]
    chosen = sorted(chosen, key=lambda r: rank[r.comparison.status])
    if not verbose:
        chosen = chosen[:limit]
    return "\n\n".join(render_result(r) for r in chosen)


def write_csv(path: Path, results: list[Result]) -> None:
    fields = [
        "rpc_id", "session_id", "stream_id", "utterance_id", "task", "status", "segments",
        "cut_loss", "cut_dup", "cut_changed", "revised", "align", "sent", "final", "diff",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            rpc, session, stream, utterance_id = r.utt.key
            writer.writerow(
                {
                    "rpc_id": rpc, "session_id": session, "stream_id": stream, "utterance_id": utterance_id,
                    "task": r.task, "status": r.comparison.status, "segments": len(r.segments),
                    "cut_loss": r.comparison.cut_loss, "cut_dup": r.comparison.cut_dup,
                    "cut_changed": r.comparison.cut_changed, "revised": r.comparison.revised,
                    "align": ",".join(r.utt.align), "sent": " | ".join(r.segments),
                    "final": r.utt.final_text, "diff": r.comparison.diff,
                }
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--stream-id")
    parser.add_argument("--session-id")
    parser.add_argument("--rpc-id")
    parser.add_argument("--examples", type=int, default=10, help="utterances listed that are not OK, worst first")
    parser.add_argument("--verbose", action="store_true", help="list every segmented utterance, OK included")
    parser.add_argument("--csv", type=Path, default=None, help="write one row per segmented utterance")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tasks, utterances, files = load(
        args.logfiles, stream_id=args.stream_id, session_id=args.session_id, rpc_id=args.rpc_id
    )
    if not files:
        print(f"No log file matches: {' '.join(args.logfiles)}", file=sys.stderr)
        return 1
    if not any(u.has_final for u in utterances.values()):
        filtered = any(v is not None for v in (args.stream_id, args.session_id, args.rpc_id))
        print(
            "No ASR_FINAL events"
            + (" match the filters." if filtered else " (enable logging.pipeline_trace and restart)."),
            file=sys.stderr,
        )
        return 1
    if any(u.missing_text for u in utterances.values()):
        print(
            "Some ASR_FINAL / EMIT / TRANSLATE_SUBMIT lines have no text (enable logging.pipeline_trace_text).",
            file=sys.stderr,
        )
        return 2

    results, counts = analyse(tasks, utterances)
    print(render_summary(results, counts, files))
    if not results:
        print(
            "\nNo segmented utterance: nothing to check (segmentation.mode must be on; for speech_to_text also "
            "segmentation.speech_to_text: true).",
            file=sys.stderr,
        )
        return 1
    examples = render_examples(results, args.examples, verbose=args.verbose)
    if examples:
        print("\n" + examples)
    if args.csv:
        write_csv(args.csv, results)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
