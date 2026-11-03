"""ASR input (docs/asr-feature-fix-design.md): the mel of the pieces equals the mel of the whole utterance and the
chunks given to the model are the ones NeMo's reference streaming (CacheAwareStreamingAudioBuffer) gives."""

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from nemotron_testbed.domain import RuntimeCommand, RuntimeCommandType, StreamKey
from nemotron_testbed.runtime import NemotronRuntime, _check_mel_config

HAS_TORCH = importlib.util.find_spec("torch") is not None

PIECE = 8960  # one STEP: 560 ms at lookahead_tokens 6
# what the engine logged on the server for attention_context=[56, 6]
STREAMING_CFG = SimpleNamespace(
    chunk_size=[49, 56], shift_size=[49, 56], pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=2
)
KEY = StreamKey("rpc", "call", "MO")


class FakePreprocessor:
    """Like NeMo FilterbankFeatures (exact_pad false): pre-emphasis, centered STFT with zero padding, log power,
    valid length = samples // hop and the frames past it set to 0."""

    featurizer = SimpleNamespace(hop_length=160, n_fft=512)

    def __call__(self, input_signal, length):
        import torch

        x = torch.cat([input_signal[:, :1], input_signal[:, 1:] - 0.97 * input_signal[:, :-1]], dim=1)
        spec = torch.stft(
            x, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400), center=True,
            pad_mode="constant", return_complex=True,
        )
        features = torch.log(spec.abs().pow(2) + 2**-24)
        valid = length // 160
        padding = torch.arange(features.size(-1)) >= valid[:, None]
        return features.masked_fill(padding[:, None, :], 0.0), valid


class FakeModel:
    def __init__(self):
        self.encoder = SimpleNamespace(get_initial_cache_state=lambda batch_size=1: (0, 0, 0))
        self.decoding = SimpleNamespace(set_strip_lang_tags=lambda *args, **kwargs: None)
        self.calls = []  # (frames given, drop_extra_pre_encoded, keep_all_outputs)

    def set_inference_prompt(self, language):
        pass

    def conformer_stream_step(self, *, processed_signal, drop_extra_pre_encoded, keep_all_outputs, **caches):
        self.calls.append((processed_signal.clone(), drop_extra_pre_encoded, keep_all_outputs))
        hypothesis = SimpleNamespace(text=f"t{len(self.calls)}", token_confidence=[0.9])
        return (
            None, [hypothesis], caches["cache_last_channel"], caches["cache_last_time"],
            caches["cache_last_channel_len"], None,
        )


def make_runtime(legacy=False) -> NemotronRuntime:
    import torch

    runtime = NemotronRuntime(
        model_path="", device="cpu", language="vi-VN", left_context=56, lookahead_tokens=6, queue_size=8,
        warmup_seconds=0.0, result_callback=lambda result: None,
    )
    runtime.model = FakeModel()
    runtime.preprocessor = FakePreprocessor()
    runtime.compute_dtype = torch.float32
    runtime.streaming_cfg = STREAMING_CFG
    runtime.legacy_features = legacy
    return runtime


def utterance(samples: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(samples) / 16_000
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(samples)).astype(np.float32)


def run(runtime, pieces, tail):
    """START, one STEP per piece, FINALIZE with tail; returns (model steps run by each STEP, frames waiting after
    each STEP, mel_tail size after each STEP)."""
    runtime._execute(RuntimeCommand(RuntimeCommandType.START, KEY, 1, "vi-VN"))
    state = runtime._states[(KEY, 1)]
    ran, waiting, tails = [], [], []
    for piece in pieces:
        before = len(runtime.model.calls)
        runtime._execute(RuntimeCommand(RuntimeCommandType.STEP, KEY, 1, "vi-VN", audio=piece))
        ran.append(len(runtime.model.calls) - before)
        waiting.append(int(state.feature_cache.size(-1)) - state.feature_index)
        tails.append(state.mel_tail.size)
    runtime._execute(RuntimeCommand(RuntimeCommandType.FINALIZE, KEY, 1, "vi-VN", audio=tail))
    return ran, waiting, tails


def nemo_reference_chunks(features):
    """NeMo CacheAwareStreamingAudioBuffer.__iter__ over the mel of the whole utterance
    (pad_and_drop_preencoded false): (frames, drop_extra_pre_encoded, keep_all_outputs) per step."""
    import torch

    cfg, chunks, index = STREAMING_CFG, [], 0
    while index < features.size(-1):
        first = index == 0
        chunk_size = cfg.chunk_size[0] if first else cfg.chunk_size[1]
        shift = cfg.shift_size[0] if first else cfg.shift_size[1]
        cache = features[:, :, max(0, index - cfg.pre_encode_cache_size[1]) : index]
        chunk = torch.cat([cache, features[:, :, index : index + chunk_size]], dim=-1)
        chunks.append((chunk, 0 if first else cfg.drop_extra_pre_encoded, index + shift >= features.size(-1)))
        index += shift
    return chunks


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class FeatureFixTest(unittest.TestCase):
    def assert_same_chunks(self, given, expected):
        import torch

        self.assertEqual(len(given), len(expected))
        for step, ((frames, drop, last), (ref_frames, ref_drop, ref_last)) in enumerate(zip(given, expected)):
            with self.subTest(step=step):
                self.assertEqual(frames.shape, ref_frames.shape)
                self.assertTrue(torch.allclose(frames, ref_frames, atol=1e-4))
                self.assertEqual((drop, last), (ref_drop, ref_last))

    def test_chunks_are_nemo_reference_chunks_of_the_whole_utterance_mel(self):
        import torch

        for steps, tail_samples in ((20, 3000), (20, 0), (3, 1), (0, 5000), (0, 300), (0, 100)):
            with self.subTest(steps=steps, tail=tail_samples):
                audio = utterance(steps * PIECE + tail_samples)
                pieces = [audio[i * PIECE : (i + 1) * PIECE] for i in range(steps)]
                runtime = make_runtime()
                run(runtime, pieces, audio[steps * PIECE :])
                whole, length = FakePreprocessor()(torch.from_numpy(audio).unsqueeze(0), torch.tensor([audio.size]))
                self.assert_same_chunks(runtime.model.calls, nemo_reference_chunks(whole[:, :, : int(length[0])]))

    def test_every_step_runs_at_once_and_nothing_piles_up(self):
        runtime = make_runtime()
        audio = utterance(20 * PIECE)
        ran, waiting, tails = run(runtime, [audio[i * PIECE : (i + 1) * PIECE] for i in range(20)], audio[:0])
        self.assertEqual(ran, [1] * 20)
        self.assertEqual(set(waiting), {6})  # was 8, 9, 10 ... before the fix
        self.assertTrue(all(size <= 2 * 160 + 256 for size in tails))  # audio kept for the next mel: < 40 ms
        frames, drop, _ = runtime.model.calls[1]
        self.assertEqual((frames.size(-1), drop), (9 + 56, 2))  # step 1 gets its pre-encode cache

    def test_final_release_happens_once(self):
        runtime = make_runtime()
        runtime._execute(RuntimeCommand(RuntimeCommandType.START, KEY, 1, "vi-VN"))
        state = runtime._states[(KEY, 1)]
        state.raw_audio = utterance(PIECE + 1000)
        runtime._update_feature_cache(state, final=True)
        frames, cached = state.mel_frames, state.feature_cache.clone()
        self.assertEqual(frames, (PIECE + 1000) // 160)
        runtime._update_feature_cache(state, final=True)
        self.assertEqual(state.mel_frames, frames)
        self.assertTrue(cached.equal(state.feature_cache))

    def test_legacy_switch_gives_the_path_before_the_fix(self):
        import torch

        runtime = make_runtime(legacy=True)
        audio = utterance(5 * PIECE)
        pieces = [audio[i * PIECE : (i + 1) * PIECE] for i in range(5)]
        _, waiting, _ = run(runtime, pieces, audio[:0])
        per_piece = torch.cat([FakePreprocessor()(torch.from_numpy(p).unsqueeze(0), torch.tensor([PIECE]))[0]
                               for p in pieces], dim=-1)  # 57 frames per piece, the last one 0
        self.assertEqual(per_piece.size(-1), 5 * 57)
        self.assertEqual(waiting, [8, 9, 10, 11, 12])
        expected = [(per_piece[:, :, 0:49], 0), (per_piece[:, :, 49:105], 2)]  # step 1: no pre-encode cache
        expected += [(per_piece[:, :, 49 + 56 * (k - 1) - 9 : 49 + 56 * k], 2) for k in range(2, 5)]
        for (frames, drop, _), (ref, ref_drop) in zip(runtime.model.calls, expected):
            self.assertEqual((frames.shape, drop), (ref.shape, ref_drop))
            self.assertTrue(frames.equal(ref))


class MelConfigTest(unittest.TestCase):
    def test_unsupported_preprocessor_config_is_refused(self):
        for cfg in ({"exact_pad": True}, {"normalize": "per_feature"}, {"normalize": "all_features"}):
            with self.subTest(cfg=cfg), self.assertRaises(RuntimeError):
                _check_mel_config(cfg)
        for cfg in ({}, {"normalize": "NA"}, {"normalize": "None", "exact_pad": False}):
            with self.subTest(cfg=cfg):
                _check_mel_config(cfg)


if __name__ == "__main__":
    unittest.main()
