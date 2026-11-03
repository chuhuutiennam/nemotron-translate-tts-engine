from __future__ import annotations


_NORMALIZED = {
    "vi": "vi-VN",
    "vi-vn": "vi-VN",
    "en": "en-US",
    "en-us": "en-US",
}

_CANONICAL_BY_SHORT = {"vi": "vi-VN", "en": "en-US"}


def normalize_language(language: str, default: str = "vi-VN") -> str:
    raw = (language or default).strip()
    if not raw:
        raw = default
    return _NORMALIZED.get(raw.lower(), raw)


def short_language(language: str) -> str:
    normalized = normalize_language(language)
    lowered = normalized.lower()
    if lowered.startswith("vi"):
        return "vi"
    if lowered.startswith("en"):
        return "en"
    return lowered.split("-", 1)[0]


def is_translatable_language(language: str) -> bool:
    return bool(language) and short_language(language) in ("vi", "en")


def canonical_translatable_language(language: str) -> str:
    """Return vi-VN / en-US for a vi or en language tag, else "" (empty input included)."""
    if not language or not language.strip():
        return ""
    return _CANONICAL_BY_SHORT.get(short_language(language), "")


def envit5_prefix(source_language: str, vi_prefix: str, en_prefix: str) -> str:
    short = short_language(source_language)
    if short == "vi":
        return vi_prefix
    if short == "en":
        return en_prefix
    raise ValueError(f"unsupported EnViT5 source language: {source_language!r}")


def envit5_output_prefix(target_language: str) -> str:
    short = short_language(target_language)
    return f"{short}:"
