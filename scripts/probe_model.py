#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from nemotron_testbed.runtime import NemotronRuntime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load/warm a local Nemotron .nemo checkpoint and verify streaming APIs"
    )
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="vi-VN")
    parser.add_argument("--left-context", type=int, default=56)
    parser.add_argument("--lookahead", type=int, default=6)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    args = parser.parse_args()

    results = []
    runtime = NemotronRuntime(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        left_context=args.left_context,
        lookahead_tokens=args.lookahead,
        queue_size=8,
        warmup_seconds=args.warmup_seconds,
        result_callback=results.append,
    )

    started = time.perf_counter()
    runtime.start(timeout=600)
    print(f"Model loaded and warmed in {time.perf_counter() - started:.2f} s")
    print(f"Sample rate: {runtime.sample_rate}")
    print(f"Compute dtype: {runtime.compute_dtype}")
    print(f"Streaming config: {runtime.streaming_cfg}")
    runtime.shutdown()
    print("Probe completed successfully")


if __name__ == "__main__":
    main()
