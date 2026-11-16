TASK_SPEECH_TO_TEXT = "speech_to_text"
TASK_REALTIME_TRANSLATION = "realtime_translation"
TASK_KEYWORD = "keyword"

VALID_TASKS = frozenset(
    {
        TASK_SPEECH_TO_TEXT,
        TASK_REALTIME_TRANSLATION,
        TASK_KEYWORD,
    }
)

# Proto / gateway task names that map to canonical internal names.
_PROTO_TASK_MAP: dict[str, str] = {
    "transcribe": TASK_SPEECH_TO_TEXT,
    "speech_to_text": TASK_SPEECH_TO_TEXT,
    "translate": TASK_REALTIME_TRANSLATION,
    "realtime_translation": TASK_REALTIME_TRANSLATION,
    "keyword": TASK_KEYWORD,
}


def normalize_task(task: str | None) -> str:
    value = (task or "").strip().lower()
    canonical = _PROTO_TASK_MAP.get(value)
    if canonical is None:
        if not value:
            return TASK_SPEECH_TO_TEXT
        raise ValueError(
            "task must be "
            '"speech_to_text" / "transcribe", "realtime_translation" / "translate", or "keyword"'
        )
    return canonical