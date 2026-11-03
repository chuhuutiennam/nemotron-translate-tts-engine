"""Device strings of audio_preprocessing (`cpu`, `cuda`, `cuda:N`), shared by the resemble-enhance and AudX models."""

from __future__ import annotations


def parse_device(device: str) -> tuple[str, int]:
    name = device.strip().lower()
    if name == "cpu":
        return "cpu", 0
    if name == "cuda":
        return "cuda", 0
    if name.startswith("cuda:") and name[5:].isdigit():
        return "cuda", int(name[5:])
    raise ValueError(f"unsupported audio preprocessing device: {device!r}")
