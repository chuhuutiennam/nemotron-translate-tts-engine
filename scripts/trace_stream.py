#!/usr/bin/env python3
"""Print a per-utterance timeline (ASR_FINAL -> TRANSLATE_SUBMIT -> TRANSLATE -> TTS_SUBMIT -> TTS_PHRASE -> SEGMENT_LATENCY -> EMIT) from engine logs.

The engine writes one JSON log line per pipeline stage when logging.pipeline_trace
is enabled. This script filters those lines by stream/session/rpc and shows, for
every utterance, which stages ran, how long each took, and where it stopped.

Examples:
  python scripts/trace_stream.py logs/nemotron-engine.log --stream-id MO
  python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --only-problems
  python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --partials

--partials also lists the ASR_PARTIAL lines (needs logging.pipeline_trace_partial). Their offsets are
negative: milliseconds before the ASR_FINAL of the same utterance.

The whole call is shown, from gRPC open to close: GRPC_OPEN, STREAM_OPEN, per-utterance VAD_START / VAD_END /
GRPC_SENT, STREAM_CLOSE, GRPC_CANCEL, GRPC_CLOSE. Each stream starts with "first audio to rtpgw" (time of the
first GRPC_SENT carrying audio, after GRPC_OPEN / STREAM_OPEN) and each utterance ends with its own
first-audio line (after VAD_END, speech end, ASR_FINAL, GRPC_OPEN). GRPC_SENT is the moment the engine handed
the message to gRPC; the rtpgw receive time is only in the gateway log.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import re
import sys
from datetime import datetime
from typing import Iterable

STAGE_ORDER = (
    "ASR_PARTIAL",
    "ASR_FINAL",
    "TRANSLATE_SUBMIT",
    "TRANSLATE",
    "ORDER_HOLD",
    "TTS_SUBMIT",
    "TTS_PHRASE",
    "SEGMENT_LATENCY",
    "EMIT",
    "DEGRADE",
    "PASSTHROUGH",
    "LANG_FALLBACK",
    "STALE",
    "GRPC_OPEN",
    "STREAM_OPEN",
    "VAD_START",
    "VAD_END",
    "GRPC_SENT",
    "STREAM_CLOSE",
    "GRPC_CANCEL",
    "GRPC_CLOSE",
)
RPC_STAGES = ("GRPC_OPEN", "GRPC_CANCEL", "GRPC_CLOSE")
STREAM_STAGES = ("STREAM_OPEN", "STREAM_CLOSE")
# Informational per-utterance lines: never change the diagnosis or the timeline baseline.
_INFO_STAGES = ("ASR_PARTIAL", "VAD_START", "VAD_END", "GRPC_SENT")
_CHAIN = ("ASR_FINAL", "TRANSLATE_SUBMIT", "TRANSLATE", "ORDER_HOLD", "TTS_SUBMIT", "TTS_PHRASE", "EMIT")
_DETAIL_KEYS = (
    "step",
    "model_step",
    "changed",
    "audio_ms",
    "common_prefix_words",
    "revised_words",
    "sentence_ends",
    "segment_idx",
    "text",
    "source_text",
    "translated_text",
    "committed_tokens",
    "remainder",
    "source_language",
    "target_language",
    "detected_language",
    "fallback",
    "language",
    "phrase_id",
    "is_last_phrase",
    "seq",
    "is_final",
    "audio_bytes",
    "confidence",
    "utterance_audio_ms",
    "latency_ms",
    "t_ms",
    "infer_ms",
    "queue_ms",
    "asr_to_submit_ms",
    "seen_to_submit_ms",
    "asr_ms",
    "translate_queue_ms",
    "translate_ms",
    "tts_queue_ms",
    "tts_ms",
    "order_hold_ms",
    "total_ms",
    "reason",
    "error",
    "queued",
    "skipped",
    "origin",
    "status",
    "queue_depth",
    "pending_postprocess",
    "phrases",
    "phrase_texts",
    "waiting_for",
    "waiting_segment",
    "held_count",
    "pre_roll_ms",
    "vad_silence_ms",
    "send_ms",
    "buffer_wait_ms",
    "first_audio",
    "speech_end_to_sent_ms",
    "speech_end_to_sent_preproc_ms",
)


G711_BYTES_PER_SEC = 8000.0
DEFAULT_MAX_CPS = 35.0
_MIN_CPS_CHARS = 10


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


def _legacy_stage(record: dict) -> str | None:
    """Older builds put the stage in `message` (`PIPELINE <STAGE>` or `<STAGE>`) and have no `stage` key."""
    message = str(record.get("message", "")).strip()
    if message.startswith("PIPELINE "):
        message = message[len("PIPELINE "):].strip()
    return message if message in STAGE_ORDER else None


_KV_RE = re.compile(r"(\w+)=(\S+)")
_SENT_INT = {"seq": "seq", "ts_start": "ts_start", "ts_end": "ts_end", "text_len": "text_len", "audio_bytes": "audio_bytes"}
_SENT_FLOAT = {
    "send_ms": "send_ms",
    "speech_end_to_sent_ms": "speech_end_to_sent_ms",
    "speech_end_to_sent_preproc_ms": "speech_end_to_sent_preproc_ms",
}


def _message_stage(record: dict) -> str | None:
    """Builds without a `stage` key log lifecycle and GRPC_SENT lines as plain messages; recover them."""
    message = str(record.get("message", ""))
    if message == "RPC opened":
        return "GRPC_OPEN"
    if message == "RPC closed":
        return "GRPC_CLOSE"
    if message.startswith("RPC cancelled"):
        record.setdefault("reason", message.partition(": ")[2])
        return "GRPC_CANCEL"
    if message == "Logical stream opened":
        return "STREAM_OPEN"
    if message == "Logical stream closed":
        return "STREAM_CLOSE"
    if message.startswith("GRPC_SENT "):
        fields = dict(_KV_RE.findall(message))
        record.setdefault("rpc_id", fields.get("rpc"))
        record.setdefault("session_id", fields.get("session"))
        record.setdefault("stream_id", fields.get("stream"))
        if "final" in fields:
            record.setdefault("is_final", fields["final"] == "True")
        for src, dst in _SENT_INT.items():
            if src in fields and fields[src].lstrip("-").isdigit():
                record.setdefault(dst, int(fields[src]))
        for src, dst in _SENT_FLOAT.items():
            try:
                record.setdefault(dst, float(fields[src]))
            except (KeyError, ValueError):
                pass
        return "GRPC_SENT"
    return None


def _attach_sent_utterances(events: list[dict]) -> None:
    """Older GRPC_SENT lines carry no utterance_id; borrow it from the EMIT line with the same seq."""
    by_seq = {
        (e.get("rpc_id"), e.get("session_id"), e.get("stream_id"), e.get("seq")): e["utterance_id"]
        for e in events
        if e["stage"] == "EMIT" and "seq" in e and "utterance_id" in e
    }
    for e in events:
        if e["stage"] == "GRPC_SENT" and "utterance_id" not in e:
            utt = by_seq.get((e.get("rpc_id"), e.get("session_id"), e.get("stream_id"), e.get("seq")))
            if utt is not None:
                e["utterance_id"] = utt


def scan_events(paths: Iterable[str]) -> tuple[list[dict], dict]:
    """Load trace events and return (events, stats) so callers can explain an empty result."""
    files = expand_paths(paths)
    stats = {"files": files, "json_lines": 0, "pipeline_lines": 0, "events": 0, "legacy_events": 0}
    events: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if "PIPELINE" in line:
                    stats["pipeline_lines"] += 1
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stats["json_lines"] += 1
                if record.get("stage") in STAGE_ORDER:
                    events.append(record)
                    continue
                if "stage" in record:
                    continue
                legacy = _legacy_stage(record)
                if legacy is not None:
                    record["stage"] = legacy
                    events.append(record)
                    stats["legacy_events"] += 1
                    continue
                recovered = _message_stage(record)
                if recovered is not None:
                    record["stage"] = recovered
                    events.append(record)
    events.sort(key=lambda r: r.get("timestamp", ""))
    _attach_sent_utterances(events)
    stats["events"] = len(events)
    return events, stats


def load_events(paths: Iterable[str]) -> list[dict]:
    return scan_events(paths)[0]


def explain_empty(paths: list[str], stats: dict, events: list[dict], hidden_partials: int = 0) -> str:
    """Return the reason no events were printed: missing file, no trace lines, or filters excluded all."""
    if not stats["files"]:
        return (
            f"No log file matches: {' '.join(paths)}\n"
            "Check the path (it is relative to the current directory) or pass a glob such as 'logs/nemotron-engine.log*'."
        )
    if not events and hidden_partials:
        return (
            f"Only ASR_PARTIAL trace lines found ({hidden_partials}); they are hidden by default.\n"
            "Re-run with --partials to list them. No ASR_FINAL/TRANSLATE/TTS_PHRASE/EMIT lines exist yet "
            "(call still in progress, or the utterance never reached ASR_FINAL)."
        )
    if not events:
        return (
            f"No pipeline trace lines in {len(stats['files'])} file(s): read {stats['json_lines']} JSON line(s), "
            f"{stats['pipeline_lines']} line(s) contain 'PIPELINE', 0 recognised as a stage.\n"
            "Check logging.pipeline_trace / LOG_PIPELINE_TRACE and that logging.level is INFO or lower. "
            "If 'PIPELINE' lines exist but are not recognised, the running build may emit a different log format."
        )
    sessions = sorted({str(e.get("session_id")) for e in events if e.get("session_id") is not None})
    streams = sorted({str(e.get("stream_id")) for e in events if e.get("stream_id") is not None})
    return (
        f"{len(events)} trace event(s) found, but none match the filters.\n"
        f"Available session_id: {', '.join(sessions[:10]) or '-'}{' ...' if len(sessions) > 10 else ''}\n"
        f"Available stream_id: {', '.join(streams) or '-'}"
    )


def filter_events(
    events: Iterable[dict],
    *,
    stream_id: str | None = None,
    session_id: str | None = None,
    rpc_id: str | None = None,
    utterance_id: int | None = None,
) -> list[dict]:
    events = list(events)
    out = []
    matched_rpcs = set()
    for ev in events:
        if ev.get("stage") in RPC_STAGES:
            continue
        if stream_id is not None and ev.get("stream_id") != stream_id:
            continue
        if session_id is not None and ev.get("session_id") != session_id:
            continue
        if rpc_id is not None and ev.get("rpc_id") != rpc_id:
            continue
        if utterance_id is not None and (
            ev.get("stage") in STREAM_STAGES or ev.get("utterance_id") != utterance_id
        ):
            continue
        out.append(ev)
        matched_rpcs.add(ev.get("rpc_id"))
    if utterance_id is None:
        scoped = stream_id is not None or session_id is not None
        for ev in events:
            if ev.get("stage") not in RPC_STAGES:
                continue
            if rpc_id is not None and ev.get("rpc_id") != rpc_id:
                continue
            if scoped and ev.get("rpc_id") not in matched_rpcs:
                continue
            out.append(ev)
        out.sort(key=lambda r: r.get("timestamp", ""))
    return out


def _is_stream_level(ev: dict) -> bool:
    return ev["stage"] in STREAM_STAGES or (ev["stage"] == "GRPC_SENT" and "utterance_id" not in ev)


def group_events(events: Iterable[dict]) -> dict[tuple, dict[int, list[dict]]]:
    """Group as {(rpc_id, session_id, stream_id): {utterance_id: [events...]}} (per-utterance stages only)."""
    grouped: dict[tuple, dict[int, list[dict]]] = {}
    for ev in events:
        if ev["stage"] in RPC_STAGES or _is_stream_level(ev):
            continue
        stream = (ev.get("rpc_id"), ev.get("session_id"), ev.get("stream_id"))
        utt = ev.get("utterance_id", -1)
        grouped.setdefault(stream, {}).setdefault(utt, []).append(ev)
    return grouped


def group_stream_events(events: Iterable[dict], known_streams: Iterable[tuple] = ()) -> dict[tuple, list[dict]]:
    """Group lifecycle events: {(rpc_id, session_id, stream_id): [GRPC_OPEN, STREAM_OPEN, ..., GRPC_CLOSE]}.

    RPC-level events (GRPC_OPEN/GRPC_CANCEL/GRPC_CLOSE) are attached to every stream of that RPC;
    an RPC with no stream at all is keyed (rpc_id, None, None).
    """
    per_stream: dict[tuple, list[dict]] = {}
    per_rpc: dict[str, list[dict]] = {}
    for ev in events:
        if ev["stage"] in RPC_STAGES:
            per_rpc.setdefault(ev.get("rpc_id"), []).append(ev)
        elif _is_stream_level(ev):
            per_stream.setdefault((ev.get("rpc_id"), ev.get("session_id"), ev.get("stream_id")), []).append(ev)
    for key in known_streams:
        per_stream.setdefault(key, [])
    out: dict[tuple, list[dict]] = {}
    for key, evs in per_stream.items():
        out[key] = sorted([*evs, *per_rpc.get(key[0], [])], key=lambda r: r.get("timestamp", ""))
    for rpc, evs in per_rpc.items():
        if not any(key[0] == rpc for key in per_stream):
            out[(rpc, None, None)] = evs
    return out


def _text_mismatch(translated: str, phrase_texts: list[str]) -> str | None:
    """Describe words lost or changed between the translation and the text actually sent to TTS."""
    expected = translated.split()
    actual = " ".join(phrase_texts).split()
    if expected == actual:
        return None
    matcher = difflib.SequenceMatcher(a=expected, b=actual, autojunk=False)
    missing: list[str] = []
    extra: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            missing.extend(expected[i1:i2])
        if tag in ("insert", "replace"):
            extra.extend(actual[j1:j2])
    parts = []
    if missing:
        parts.append(f"missing {' '.join(missing)!r}")
    if extra:
        parts.append(f"changed/extra {' '.join(extra)!r}")
    return ", ".join(parts)


def _text_chars(ev: dict) -> int | None:
    if "text" in ev:
        return len(str(ev["text"]))
    value = ev.get("text_len")
    return int(value) if value is not None else None


def _tts_consistency_problems(stages: dict[str, list[dict]], max_cps: float) -> list[str]:
    """Checks that the audio the client receives covers the translated text."""
    phrase_events = sorted(
        (e for e in stages.get("TTS_PHRASE", []) if not e.get("error")),
        key=lambda e: e.get("phrase_id", 0),
    )
    if not phrase_events:
        return []
    problems: list[str] = []

    submits = stages.get("TTS_SUBMIT", [])
    expected_phrases = max((s.get("phrases", 0) for s in submits), default=0)
    if expected_phrases and len(phrase_events) < expected_phrases and not stages.get("DEGRADE"):
        problems.append(f"INCOMPLETE: only {len(phrase_events)} of {expected_phrases} phrases synthesized")

    translated = next((e["translated_text"] for e in stages.get("TRANSLATE", []) if "translated_text" in e), None)
    if translated is not None and all("text" in e for e in phrase_events):
        mismatch = _text_mismatch(translated, [str(e["text"]) for e in phrase_events])
        if mismatch:
            problems.append(f"WARN: TTS text differs from translation ({mismatch})")

    for e in phrase_events:
        chars = _text_chars(e)
        audio_bytes = e.get("audio_bytes", 0)
        if chars is None or chars < _MIN_CPS_CHARS or audio_bytes <= 0:
            continue
        cps = chars / (audio_bytes / G711_BYTES_PER_SEC)
        if cps > max_cps:
            problems.append(
                f"WARN: phrase {e.get('phrase_id')} audio too short for its text "
                f"({chars} chars in {audio_bytes / G711_BYTES_PER_SEC:.2f}s = {cps:.0f} chars/s > {max_cps:.0f})"
            )

    emits = [e for e in stages.get("EMIT", []) if not e.get("skipped") and e.get("queued") is not False]
    emitted = sum(e.get("audio_bytes", 0) for e in emits)
    synthesized = sum(e.get("audio_bytes", 0) for e in phrase_events)
    if emits and emitted < synthesized:
        problems.append(f"WARN: emitted audio {emitted} bytes < synthesized {synthesized} bytes")

    seqs = [e["seq"] for e in stages.get("EMIT", []) if "seq" in e]
    if seqs != sorted(seqs):
        problems.append("WARN: EMIT seq not increasing in emission order")
    return problems


def diagnose(events: list[dict], max_cps: float = DEFAULT_MAX_CPS) -> str:
    """Return 'OK' or a short description of where the utterance stopped or degraded.

    With segmentation.mode=on an utterance carries several playback segments
    (segment_idx); each one is checked on its own against the shared ASR_FINAL.
    ASR_PARTIAL lines are informational and never change the status.
    """
    events = [e for e in events if e["stage"] not in _INFO_STAGES]
    segments = sorted({e["segment_idx"] for e in events if e.get("segment_idx") is not None})
    if not segments:
        return _diagnose_one(events, max_cps)
    shared = [e for e in events if e["stage"] == "ASR_FINAL"]
    bad = []
    for idx in segments:
        own = [e for e in events if e.get("segment_idx") == idx]
        status = _diagnose_one(shared + own, max_cps)
        if status != "OK":
            bad.append(f"segment {idx}: {status}")
    if not shared:
        bad.insert(0, "INCOMPLETE: no ASR_FINAL")
    return "; ".join(bad) if bad else "OK"


def _diagnose_one(events: list[dict], max_cps: float) -> str:
    stages: dict[str, list[dict]] = {}
    for ev in events:
        stages.setdefault(ev["stage"], []).append(ev)

    if "ASR_FINAL" not in stages:
        return "INCOMPLETE: no ASR_FINAL"
    asr = stages["ASR_FINAL"][0]
    if asr.get("text_len", 0) == 0:
        return "EMPTY_ASR: ASR returned no text"

    problems: list[str] = []
    degrade = stages.get("DEGRADE", [])
    if degrade:
        reasons = ", ".join(str(d.get("reason")) for d in degrade)
        problems.append(f"DEGRADED ({reasons})")
    elif stages.get("PASSTHROUGH"):
        pass  # replayed untranslated by design: no TRANSLATE / TTS chain expected
    elif asr.get("target_language"):
        for stage in ("TRANSLATE", "TTS_PHRASE", "EMIT"):
            if stage in stages:
                continue
            before = _CHAIN[: _CHAIN.index(stage)]
            reached = [s for s in before if s in stages][-1]
            problems.append(f"INCOMPLETE: no {stage} (stopped after {reached})")
            break
    if not any(e.get("is_final") for e in stages.get("EMIT", [])):
        if not any(p.startswith("INCOMPLETE") for p in problems):
            problems.append("INCOMPLETE: no final EMIT")
    for stale in stages.get("STALE", []):
        problems.append(f"WARN: stale {stale.get('origin')} result ({stale.get('reason')})")
    if any(e.get("queued") is False for e in stages.get("EMIT", [])):
        problems.append("WARN: response buffer rejected a result")
    if any(e.get("skipped") for e in stages.get("EMIT", [])):
        problems.append("NOTE: skipped empty result (no text, no audio) - not sent to client")
    if (
        asr.get("target_language")
        and not degrade
        and not any(e.get("audio_bytes", 0) > 0 for e in stages.get("EMIT", []))
        and "EMIT" in stages
    ):
        problems.append("NO_AUDIO: translated result emitted without audio")
    problems.extend(_tts_consistency_problems(stages, max_cps))
    return "; ".join(problems) if problems else "OK"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_details(ev: dict) -> str:
    parts = []
    for key in _DETAIL_KEYS:
        if key in ev:
            parts.append(f"{key}={ev[key]!r}")
            if key == "audio_bytes" and ev[key]:
                parts.append(f"audio_s={ev[key] / G711_BYTES_PER_SEC:.2f}")
    for key in ("text_len", "source_text_len", "translated_text_len"):
        base = key[: -len("_len")]
        if key in ev and base not in ev:
            parts.append(f"{key}={ev[key]}")
    return " ".join(parts)


def _delta_ms(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000.0


def _fmt_offset(delta_ms: float | None) -> str:
    if delta_ms is None:
        return " " * 11
    return f"{'-' if delta_ms < 0 else '+'}{abs(delta_ms):8.1f}ms"


def _first_stage_ts(events: Iterable[dict], stage: str) -> datetime | None:
    ev = next((e for e in events if e["stage"] == stage), None)
    return _parse_ts(ev.get("timestamp")) if ev else None


def _first_audio_sent(events: Iterable[dict]) -> dict | None:
    """First GRPC_SENT carrying audio: the moment the first audio of the call/utterance left the engine."""
    sent = [e for e in events if e["stage"] == "GRPC_SENT" and e.get("audio_bytes", 0) > 0]
    return next((e for e in sent if e.get("first_audio")), sent[0] if sent else None)


def _utterance_first_audio_line(events: list[dict], open_ts: datetime | None, has_sent_logs: bool) -> str | None:
    first = _first_audio_sent(events)
    if first is None:
        emitted_audio = any(
            e["stage"] == "EMIT" and e.get("audio_bytes", 0) > 0 and not e.get("skipped") for e in events
        )
        if has_sent_logs and emitted_audio:
            return "  >> first audio to rtpgw: NONE (EMIT had audio but no GRPC_SENT with audio)"
        return None
    sent_ts = _parse_ts(first.get("timestamp"))
    parts: list[str] = []
    for label, stage in (("VAD_END", "VAD_END"), ("ASR_FINAL", "ASR_FINAL")):
        delta = _delta_ms(_first_stage_ts(events, stage), sent_ts)
        if delta is not None:
            parts.append(f"{delta:+.1f}ms after {label}")
    if first.get("speech_end_to_sent_ms") is not None:
        parts.append(f"{first['speech_end_to_sent_ms']:.1f}ms after speech end")
        if first.get("speech_end_to_sent_preproc_ms") is not None:
            parts.append(f"{first['speech_end_to_sent_preproc_ms']:.1f}ms after speech end incl. denoiser delay")
    elif any(e.get("speech_end_to_sent_ms") is not None for e in events):
        parts.append("before speech end (segmented)")
    delta = _delta_ms(open_ts, sent_ts)
    if delta is not None:
        parts.append(f"{delta:+.1f}ms after GRPC_OPEN")
    return f"  >> first audio to rtpgw (seq={first.get('seq')}): " + (", ".join(parts) or "time unknown")


def _stream_first_audio_line(
    utterances: dict[int, list[dict]], stream_events: list[dict], open_ts: datetime | None, stream_open_ts: datetime | None
) -> str | None:
    if not stream_events:
        return None
    sent_events = [e for evs in utterances.values() for e in evs if e["stage"] == "GRPC_SENT"]
    sent_events += [e for e in stream_events if e["stage"] == "GRPC_SENT"]
    if not sent_events:
        return "first audio to rtpgw: unknown (no GRPC_SENT lines; needs logging.pipeline_trace)"
    sent_events.sort(key=lambda e: e.get("timestamp", ""))
    first = _first_audio_sent(sent_events)
    if first is None:
        return "first audio to rtpgw: NONE (no GRPC_SENT carried audio)"
    sent_ts = _parse_ts(first.get("timestamp"))
    parts = []
    delta = _delta_ms(open_ts, sent_ts)
    if delta is not None:
        parts.append(f"{delta:.1f}ms after GRPC_OPEN")
    delta = _delta_ms(stream_open_ts, sent_ts)
    if delta is not None:
        parts.append(f"{delta:.1f}ms after STREAM_OPEN")
    where = f"utterance {first['utterance_id']}, " if "utterance_id" in first else ""
    return f"first audio to rtpgw: {', '.join(parts) or 'time unknown'} ({where}seq={first.get('seq')})"


def render(
    grouped: dict[tuple, dict[int, list[dict]]],
    only_problems: bool = False,
    max_cps: float = DEFAULT_MAX_CPS,
    stream_events: dict[tuple, list[dict]] | None = None,
) -> str:
    stream_events = stream_events or {}
    keys = list(grouped)
    keys += [key for key in stream_events if key not in grouped]
    lines: list[str] = []
    for key in keys:
        rpc_id, session_id, stream_id = key
        utterances = grouped.get(key, {})
        lifecycle = stream_events.get(key, [])
        statuses = {utt_id: diagnose(utterances[utt_id], max_cps) for utt_id in utterances}
        shown = [u for u in sorted(utterances) if not (only_problems and statuses[u] == "OK")]
        cancelled = any(e["stage"] == "GRPC_CANCEL" for e in lifecycle)
        if only_problems and not shown and not cancelled:
            continue

        lines.append(f"== session={session_id} stream={stream_id} rpc={rpc_id} ==")
        all_events = [*lifecycle, *(e for evs in utterances.values() for e in evs)]
        open_ts = _first_stage_ts(lifecycle, "GRPC_OPEN")
        stream_open_ts = _first_stage_ts(lifecycle, "STREAM_OPEN")
        has_sent_logs = any(e["stage"] == "GRPC_SENT" for e in all_events)
        summary = _stream_first_audio_line(utterances, lifecycle, open_ts, stream_open_ts)
        if summary:
            lines.append(summary)
        if lifecycle:
            base_ts = open_ts or min(
                (t for t in (_parse_ts(e.get("timestamp")) for e in lifecycle) if t is not None), default=None
            )
            lines.append("stream lifecycle:")
            for ev in lifecycle:
                delta = _delta_ms(base_ts, _parse_ts(ev.get("timestamp")))
                lines.append(f"  {_fmt_offset(delta)}  {ev['stage']:<16} {_format_details(ev)}")
            if not any(e["stage"] in ("STREAM_CLOSE", "GRPC_CLOSE", "GRPC_CANCEL") for e in lifecycle):
                lines.append("  (log ends before the stream/RPC was closed)")

        for utt_id in shown:
            events = utterances[utt_id]
            lines.append(f"utterance {utt_id}: {statuses[utt_id]}")
            baseline = next((e for e in events if e["stage"] not in _INFO_STAGES), events[0])
            first_ts = _parse_ts(baseline.get("timestamp"))
            for ev in events:
                delta = _delta_ms(first_ts, _parse_ts(ev.get("timestamp")))
                lines.append(f"  {_fmt_offset(delta)}  {ev['stage']:<16} {_format_details(ev)}")
            audio_line = _utterance_first_audio_line(events, open_ts, has_sent_logs)
            if audio_line:
                lines.append(audio_line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logfiles", nargs="+", help="log file(s); globs and rotated <file>.YYYY-MM-DD are included")
    parser.add_argument("--stream-id")
    parser.add_argument("--session-id")
    parser.add_argument("--rpc-id")
    parser.add_argument("--utterance-id", type=int)
    parser.add_argument(
        "--max-cps",
        type=float,
        default=DEFAULT_MAX_CPS,
        help="flag a phrase whose audio is shorter than text length / this many chars per second (default %(default)s)",
    )
    parser.add_argument("--only-problems", action="store_true", help="hide utterances that completed OK")
    parser.add_argument(
        "--partials",
        action="store_true",
        help="also list ASR_PARTIAL lines (needs logging.pipeline_trace_partial; about 2 lines/s per stream)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    all_events, stats = scan_events(args.logfiles)
    hidden_partials = 0
    if not args.partials:
        shown = [e for e in all_events if e["stage"] != "ASR_PARTIAL"]
        hidden_partials = len(all_events) - len(shown)
        all_events = shown
    events = filter_events(
        all_events,
        stream_id=args.stream_id,
        session_id=args.session_id,
        rpc_id=args.rpc_id,
        utterance_id=args.utterance_id,
    )
    if not events:
        print(explain_empty(args.logfiles, stats, all_events, hidden_partials), file=sys.stderr)
        return 2 if not stats["files"] else 1
    if stats["legacy_events"]:
        print(
            f"Note: {stats['legacy_events']} event(s) came from an older log format (stage taken from 'message'); "
            "newer fields such as t_ms may be missing. Deploy the current build for full traces.",
            file=sys.stderr,
        )
    grouped = group_events(events)
    output = render(
        grouped,
        only_problems=args.only_problems,
        max_cps=args.max_cps,
        stream_events=group_stream_events(events, grouped),
    )
    print(output if output else "All utterances OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
