#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading

from nemotron_testbed.domain import StreamKey, TranslationJob
from nemotron_testbed.translation_runtime import TranslationRuntime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("tokenizer_path")
    p.add_argument("--text", default="Xin chào, tôi muốn kiểm tra tài khoản.")
    p.add_argument("--source", default="vi-VN")
    p.add_argument("--target", default="en-US")
    p.add_argument("--compute-type", default="int8")
    args = p.parse_args()

    done = threading.Event()
    holder = {}

    def callback(result):
        holder["result"] = result
        done.set()

    runtime = TranslationRuntime(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device="cpu",
        compute_type=args.compute_type,
        inter_threads=1,
        intra_threads=2,
        queue_size=4,
        beam_size=1,
        max_decoding_length=256,
        vi_prefix="vi: ",
        en_prefix="en: ",
        result_callback=callback,
    )
    runtime.start()
    runtime.submit_nowait(
        TranslationJob(
            key=StreamKey("probe", "probe", "MO"),
            utterance_id=1,
            source_text=args.text,
            source_language=args.source,
            target_language=args.target,
            ts_start=0,
            ts_end=0,
        )
    )
    if not done.wait(30):
        raise TimeoutError("translation probe timed out")
    result = holder["result"]
    if result.error:
        raise RuntimeError(result.error)
    print(result.translated_text)
    print(f"latency_ms={result.latency_ms:.1f}")
    runtime.shutdown()


if __name__ == "__main__":
    main()
