#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
from pathlib import Path

from nemotron_testbed.domain import StreamKey, TtsJob
from nemotron_testbed.tts_runtime import TtsRuntime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--en-model", required=True)
    p.add_argument("--en-config")
    p.add_argument("--vi-model", required=True)
    p.add_argument("--vi-config")
    p.add_argument("--language", default="en-US")
    p.add_argument("--text", default="Hello, this is a Piper test.")
    p.add_argument("--pt", type=int, choices=(0, 8), default=0)
    p.add_argument("--out", type=Path, default=Path("tts_probe.g711"))
    args = p.parse_args()

    done = threading.Event()
    holder = {}

    def callback(result):
        holder["result"] = result
        done.set()

    runtime = TtsRuntime(
        en_model_path=args.en_model,
        en_config_path=args.en_config,
        vi_model_path=args.vi_model,
        vi_config_path=args.vi_config,
        target_sample_rate=8000,
        queue_size=4,
        max_text_chars=1000,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
        volume=1.0,
        result_callback=callback,
    )
    runtime.start()
    runtime.submit_nowait(
        TtsJob(
            key=StreamKey("probe", "probe", "MO"),
            utterance_id=1,
            text=args.text,
            target_language=args.language,
            ts_start=0,
            ts_end=0,
            audio_pt=args.pt,
        )
    )
    if not done.wait(30):
        raise TimeoutError("TTS probe timed out")
    result = holder["result"]
    if result.error:
        raise RuntimeError(result.error)
    args.out.write_bytes(result.audio_payload)
    print(f"wrote {len(result.audio_payload)} raw G.711 bytes to {args.out}")
    print(f"pt={result.audio_pt} latency_ms={result.latency_ms:.1f}")
    runtime.shutdown()


if __name__ == "__main__":
    main()
