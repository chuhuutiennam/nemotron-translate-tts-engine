"""Pure-Python parts of scripts/probe_qwen3_asr.py (Gate 0, docs/qwen3-asr-design.md).

The model side needs the Qwen venv and a GPU; here a fake model with the same state fields as
qwen_asr's ASRStreamingState (buffer, audio_accum, chunk_id, text, language) drives the streaming loop.
"""
import argparse
import importlib.util
import json
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_qwen3_asr.py"
_spec = importlib.util.spec_from_file_location("probe_qwen3_asr", _SCRIPT)
probe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = probe  # @dataclass looks its module up in sys.modules
_spec.loader.exec_module(probe)

SR = 16_000


class FakeState:
    def __init__(self, chunk_samples: int):
        self.chunk_size_samples = chunk_samples
        self.buffer = np.zeros(0, dtype=np.float32)
        self.audio_accum = np.zeros(0, dtype=np.float32)
        self.chunk_id = 0
        self.text = ""
        self.language = ""


class FakeAsr:
    """Buffers and re-feeds like qwen_asr's streaming_transcribe; records the audio length of each model run."""

    def __init__(self, language: str = "Vietnamese"):
        self.language = language
        self.init_kwargs: list[dict] = []
        self.generated: list[int] = []
        self.transcribed: list[dict] = []

    def _generate(self, state: FakeState) -> None:
        self.generated.append(state.audio_accum.size)
        state.text = f"text{state.audio_accum.size // SR}"
        state.language = self.language

    def init_streaming_state(self, **kwargs) -> FakeState:
        self.init_kwargs.append(kwargs)
        return FakeState(round(kwargs["chunk_size_sec"] * SR))

    def streaming_transcribe(self, pcm, state):
        state.buffer = np.concatenate([state.buffer, pcm])
        while state.buffer.size >= state.chunk_size_samples:
            chunk, state.buffer = state.buffer[: state.chunk_size_samples], state.buffer[state.chunk_size_samples :]
            state.audio_accum = np.concatenate([state.audio_accum, chunk])
            state.chunk_id += 1
            self._generate(state)
        return state

    def finish_streaming_transcribe(self, state):
        if state.buffer.size == 0:
            return state
        state.audio_accum = np.concatenate([state.audio_accum, state.buffer])
        state.buffer = np.zeros(0, dtype=np.float32)
        self._generate(state)
        return state

    def transcribe(self, audio, context="", language=None):
        self.transcribed.append({"audio": audio, "context": context, "language": language})
        return [SimpleNamespace(text="offline text", language="Vietnamese")]


def ticking_clock(step: float = 0.1):
    """Each call moves time by `step`, so every timed call lasts exactly `step` seconds."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def write_wav(path: Path, seconds: float, rate: int = SR) -> None:
    samples = (0.1 * np.sin(np.arange(round(seconds * rate)) / 10.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())


class LanguageTest(unittest.TestCase):
    def test_engine_tags_map_to_qwen_names(self):
        for tag in ("vi", "vi-VN", "VI-vn", "Vietnamese", "vietnamese"):
            self.assertEqual(probe.qwen_language(tag), "Vietnamese", tag)
        for tag in ("en", "en-US", "English"):
            self.assertEqual(probe.qwen_language(tag), "English", tag)

    def test_auto_and_empty_let_the_model_detect(self):
        for tag in ("", "auto", "AUTO", "  ", None):
            self.assertIsNone(probe.qwen_language(tag), repr(tag))

    def test_other_names_are_normalised_like_qwen_asr(self):
        self.assertEqual(probe.qwen_language("japanese"), "Japanese")
        self.assertEqual(probe.qwen_language("CHINESE"), "Chinese")

    def test_at_lang_takes_the_manifest_column(self):
        self.assertEqual(probe.resolve_language("@lang", {"lang": "en"}), "English")
        self.assertIsNone(probe.resolve_language("@lang", {"lang": ""}))
        self.assertEqual(probe.resolve_language("vi", {"lang": "en"}), "Vietnamese")


class DriverAndDeviceTest(unittest.TestCase):
    def test_driver_version_parsing(self):
        self.assertEqual(probe.parse_driver_version("570.86.10"), (570, 86, 10))
        self.assertEqual(probe.parse_driver_version(" 550.54 "), (550, 54))
        self.assertIsNone(probe.parse_driver_version("N/A"))
        self.assertIsNone(probe.parse_driver_version(""))

    def test_driver_needs_r570(self):
        self.assertTrue(probe.driver_ok((570, 86, 10)))
        self.assertTrue(probe.driver_ok((580, 1)))
        self.assertFalse(probe.driver_ok((550, 54, 15)))
        self.assertFalse(probe.driver_ok(None))

    def test_cuda_index(self):
        self.assertEqual(probe.parse_cuda_index("cuda"), 0)
        self.assertEqual(probe.parse_cuda_index("cuda:1"), 1)
        for bad in ("cpu", "cuda:x", "gpu:0"):
            with self.assertRaises(ValueError, msg=bad):
                probe.parse_cuda_index(bad)

    def test_prepare_environ_pins_the_gpu_and_goes_offline(self):
        environ: dict[str, str] = {"CUDA_VISIBLE_DEVICES": "0,1"}
        self.assertEqual(probe.prepare_environ("cuda:1", True, environ), 1)
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(environ["VLLM_WORKER_MULTIPROC_METHOD"], "spawn")
        self.assertEqual(environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environ["TRANSFORMERS_OFFLINE"], "1")

    def test_prepare_environ_online_keeps_hf_unset_and_respects_values(self):
        environ = {"VLLM_WORKER_MULTIPROC_METHOD": "fork"}
        probe.prepare_environ("cuda", False, environ)
        self.assertNotIn("HF_HUB_OFFLINE", environ)
        self.assertEqual(environ["VLLM_WORKER_MULTIPROC_METHOD"], "fork")


class AudioHelpersTest(unittest.TestCase):
    def test_split_at_the_utterance_cap(self):
        audio = np.arange(25 * SR, dtype=np.float32)
        pieces = probe.split_utterances(audio, 10 * SR)
        self.assertEqual([p.size for p in pieces], [10 * SR, 10 * SR, 5 * SR])
        np.testing.assert_array_equal(np.concatenate(pieces), audio)

    def test_split_edge_cases(self):
        self.assertEqual(probe.split_utterances(np.zeros(0, dtype=np.float32), SR), [])
        self.assertEqual([p.size for p in probe.split_utterances(np.zeros(2 * SR), 2 * SR)], [2 * SR])
        self.assertEqual([p.size for p in probe.split_utterances(np.zeros(3 * SR), 0)], [3 * SR])

    def test_step_pieces(self):
        self.assertEqual([p.size for p in probe.step_pieces(np.zeros(5 * SR), 2 * SR)], [2 * SR, 2 * SR, SR])

    def test_tile_to(self):
        np.testing.assert_array_equal(probe.tile_to(np.array([1.0, 2.0, 3.0]), 7), [1, 2, 3, 1, 2, 3, 1])
        np.testing.assert_array_equal(probe.tile_to(np.array([1.0, 2.0, 3.0]), 2), [1, 2])
        with self.assertRaises(ValueError):
            probe.tile_to(np.zeros(0), 5)


class NumbersTest(unittest.TestCase):
    def test_summary(self):
        s = probe.summarize_ms([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean"], 25.0)
        self.assertAlmostEqual(s["p50"], 25.0)
        self.assertAlmostEqual(s["max"], 40.0)
        self.assertIsNone(probe.summarize_ms([]))

    def test_serial_streams(self):
        # 1.0 s of GPU per 10 s utterance = 0.1 s per speech second -> 7 streams at 70%.
        self.assertAlmostEqual(probe.serial_streams(1.0, 10.0, 0.7), 7.0)
        self.assertEqual(probe.serial_streams(0.0, 10.0, 0.7), math.inf)
        self.assertEqual(probe.serial_streams(1.0, 0.0, 0.7), 0.0)

    def test_call_budget(self):
        self.assertAlmostEqual(probe.call_budget_ms(2.0, 8, 0.7), 175.0)
        self.assertAlmostEqual(probe.call_budget_ms(1.0, 0, 0.5), 500.0)


class StreamUtteranceTest(unittest.TestCase):
    def run_stream(self, seconds: float, chunk: float = 2.0, asr: FakeAsr | None = None):
        asr = asr or FakeAsr()
        run = probe.stream_utterance(
            asr,
            np.zeros(round(seconds * SR), dtype=np.float32),
            chunk_size_sec=chunk,
            language="Vietnamese",
            context="hot words",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            clock=ticking_clock(0.1),
        )
        return asr, run

    def test_one_model_call_per_chunk_then_finish_on_the_rest(self):
        asr, run = self.run_stream(9.5)
        self.assertEqual(asr.generated, [2 * SR, 4 * SR, 6 * SR, 8 * SR, round(9.5 * SR)])
        self.assertEqual(run.call_audio_s, [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(len(run.call_ms), 4)
        for ms in run.call_ms:
            self.assertAlmostEqual(ms, 100.0)
        self.assertAlmostEqual(run.finish_ms, 100.0)
        # 5 feeds (the last one only buffers 1.5 s) + finish, 0.1 s each
        self.assertAlmostEqual(run.compute_s, 0.6)
        self.assertEqual(run.text, "text9")
        self.assertEqual(run.language, "Vietnamese")

    def test_finish_on_an_empty_buffer_is_not_a_model_call(self):
        asr, run = self.run_stream(4.0)
        self.assertEqual(asr.generated, [2 * SR, 4 * SR])
        self.assertIsNone(run.finish_ms)
        self.assertEqual(run.text, "text4")

    def test_state_gets_the_streaming_settings(self):
        asr, _ = self.run_stream(1.0, chunk=1.0)
        self.assertEqual(
            asr.init_kwargs,
            [{"context": "hot words", "language": "Vietnamese", "unfixed_chunk_num": 2, "unfixed_token_num": 5,
              "chunk_size_sec": 1.0}],
        )


class TranscribeRowTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        write_wav(self.dir / "long.wav", 23.0)
        self.row = {"id": "c1", "wav": str(self.dir / "long.wav"), "content_type": "news",
                    "reference": "text10 text10 text3", "lang": "vi"}

    def args(self, **overrides) -> argparse.Namespace:
        argv = ["wer", "--manifest", "m.tsv", "--out", "o.jsonl", "--no-narrowband"]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return probe.build_parser().parse_args(argv)

    def test_stream_mode_cuts_at_the_cap_and_joins_the_texts(self):
        asr = FakeAsr()
        row = probe.transcribe_row(asr, self.row, self.args(), clock=ticking_clock(0.1))
        self.assertEqual(row["error"], "")
        self.assertEqual(row["mode"], "stream")
        self.assertEqual(row["utterances"], 3)  # 10 + 10 + 3 s
        self.assertEqual(row["hyp"], "text10 text10 text3")
        self.assertEqual(row["n_final"], 3)
        self.assertEqual(row["audio_s"], 23.0)
        self.assertEqual(row["language_detected"], "Vietnamese")
        self.assertEqual(row["calls"], 5 + 5 + 1)  # chunks of 2 s; 10 s ends on a chunk, 3 s leaves 1 s to finish
        self.assertEqual(row["finish_ms_max"], 100.0)
        self.assertIsNone(row["first_result_s"])
        self.assertEqual(asr.init_kwargs[0]["language"], "Vietnamese")
        # eval_wer.py score reads the row as it is
        self.assertEqual(probe.wer_pairs([row]), {"c1": (0, 3)})

    def test_offline_mode_sends_the_whole_clip_with_its_rate(self):
        asr = FakeAsr()
        row = probe.transcribe_row(asr, self.row, self.args(mode="offline", language="auto"))
        self.assertEqual(row["mode"], "offline")
        self.assertEqual(row["hyp"], "offline text")
        (call,) = asr.transcribed
        audio, rate = call["audio"]
        self.assertEqual(rate, SR)
        self.assertEqual(audio.size, 23 * SR)
        self.assertIsNone(call["language"])

    def test_a_bad_clip_becomes_an_error_row(self):
        row = probe.transcribe_row(FakeAsr(), dict(self.row, wav=str(self.dir / "missing.wav")), self.args())
        self.assertIn("FileNotFoundError", row["error"])
        self.assertEqual(row["hyp"], "")
        self.assertEqual(row["n_final"], 0)
        self.assertEqual(row["id"], "c1")


class CommandLineTest(unittest.TestCase):
    def test_wer_defaults_follow_the_design(self):
        args = probe.build_parser().parse_args(["wer", "--manifest", "m.tsv", "--out", "o.jsonl"])
        self.assertEqual(args.mode, "stream")
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.chunk_size_sec, 2.0)
        self.assertEqual(args.max_utterance_s, 10.0)
        self.assertEqual(args.language, "vi")
        self.assertTrue(args.narrowband)
        self.assertTrue(args.offline)
        self.assertEqual(args.device, "cuda:0")

    def test_llm_kwargs_only_carry_set_options(self):
        args = probe.build_parser().parse_args(["wer", "--manifest", "m", "--out", "o"])
        self.assertEqual(
            probe.llm_kwargs(args),
            {"gpu_memory_utilization": 0.4, "max_model_len": 2048, "max_num_seqs": 1, "max_new_tokens": 64},
        )
        args = probe.build_parser().parse_args(
            ["wer", "--manifest", "m", "--out", "o", "--attention-backend", "TRITON_ATTN", "--enforce-eager",
             "--quantization", "fp8", "--dtype", "float16"]
        )
        kwargs = probe.llm_kwargs(args)
        self.assertEqual(kwargs["attention_backend"], "TRITON_ATTN")
        self.assertTrue(kwargs["enforce_eager"])
        self.assertEqual(kwargs["quantization"], "fp8")
        self.assertEqual(kwargs["dtype"], "float16")

    def test_latency_needs_a_wav_and_has_json_args(self):
        args = probe.build_parser().parse_args(["latency", "--wav", "a.wav", "--wav", "b.wav", "--json-out", "r.json"])
        self.assertEqual(args.wav, [Path("a.wav"), Path("b.wav")])
        self.assertEqual(args.chunk_sizes, [1.0, 2.0])
        self.assertEqual(args.utterance_s, [9.5])
        plain = probe.args_for_json(args)
        self.assertNotIn("func", plain)
        self.assertEqual(plain["wav"], ["a.wav", "b.wav"])
        json.dumps(plain)  # serialisable
        with self.assertRaises(SystemExit):
            probe.build_parser().parse_args(["latency"])

    def test_model_dir_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(
                probe.check_model_dir(folder),
                ["missing config.json", "missing tokenizer_config.json", "no *.safetensors"],
            )
            for name in ("config.json", "tokenizer_config.json", "model-00001-of-00002.safetensors"):
                (folder / name).write_text("{}")
            self.assertEqual(probe.check_model_dir(folder), [])
            self.assertEqual(probe.check_model_dir(folder / "nope"), [f"{folder / 'nope'} is not a folder"])


if __name__ == "__main__":
    unittest.main()
