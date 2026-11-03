"""Read the engine's periodic STATS log lines (shared by check_preproc_stats.py and stats_phases.py).

A STATS line is a JSON log record {"timestamp": ISO UTC, "message": "STATS", "stats": {...}} written every
stats.log_interval_s; its numbers are the deltas of that interval (gauges are the value at the end of it).
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timedelta, timezone
from typing import Iterable


def expand_paths(paths: Iterable[str]) -> list[str]:
    """Globs, plus rotated <file>.* next to each match (oldest first by name)."""
    found: list[str] = []
    for path in paths:
        base = glob.glob(path)
        rotated = sorted(glob.glob(path + ".*"))
        for candidate in [*rotated, *base]:
            if candidate not in found:
                found.append(candidate)
    return found


def parse_time(text: str) -> datetime:
    """ISO 8601; a time without a zone is UTC, like the log timestamps."""
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def read_stats(paths: Iterable[str]) -> list[tuple[datetime, dict]]:
    """(timestamp, stats) of every STATS line, sorted by time."""
    rows: list[tuple[datetime, dict]] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"STATS"' not in line:
                    continue
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                    if record.get("message") != "STATS" or not isinstance(record.get("stats"), dict):
                        continue
                    rows.append((parse_time(record["timestamp"]), record["stats"]))
                except (ValueError, KeyError, TypeError):
                    continue
    rows.sort(key=lambda row: row[0])
    return rows


def select(rows: list[tuple[datetime, dict]], since=None, until=None, last_minutes=None):
    """Rows in [since, until]; last_minutes counts back from the newest STATS line (not from now)."""
    if last_minutes is not None and rows:
        cut = rows[-1][0] - timedelta(minutes=last_minutes)
        since = cut if since is None else max(since, cut)
    return [(t, s) for t, s in rows if (since is None or t >= since) and (until is None or t <= until)]
