import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nemotron_testbed import audx_model as ax

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _NotATensor:
    pass


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class AudxModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.tmp.name) / "audx.pth"
        cls.state = ax.build_network().state_dict()
        torch.save(cls.state, cls.checkpoint)
        cls.model = ax.AudxModel(cls.checkpoint, "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _signal(self, n, seed=1):
        return (0.1 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)

    def test_same_length_float32_and_finite(self):
        for n in (0, 1, 255, 256, 16_000, 16_001):
            with self.subTest(n=n):
                out = self.model(self._signal(n))
                self.assertEqual(out.shape, (n,))
                self.assertEqual(out.dtype, np.float32)
                self.assertTrue(np.isfinite(out).all())

    def test_state_dict_names_and_shapes_are_the_audx_ones(self):
        self.assertEqual({k: tuple(v.shape) for k, v in self.state.items()}, ax.EXPECTED_SHAPES)

    def test_identity_network_reconstructs_the_input(self):
        class Identity(self.torch.nn.Module):
            def forward(self, magnitude):
                return magnitude

        model = ax.AudxModel(self.checkpoint, "cpu")
        model.net = Identity()
        x = self._signal(16_000 + 77)
        np.testing.assert_allclose(model(x), x, atol=1e-5)

    def test_output_only_looks_64_ms_ahead(self):
        a = self._signal(16_000, seed=2)
        b = a.copy()
        change_at = 9_000
        b[change_at:] = self._signal(16_000 - change_at, seed=3)
        out_a, out_b = self.model(a), self.model(b)
        cut = change_at - ax.LOOKAHEAD_SAMPLES
        np.testing.assert_allclose(out_a[:cut], out_b[:cut], atol=1e-6)
        self.assertGreater(float(np.max(np.abs(out_a[change_at:] - out_b[change_at:]))), 1e-4)

    def test_sha256_of_the_checkpoint(self):
        self.assertEqual(self.model.sha256, hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def test_checkpoint_with_arbitrary_objects_is_refused(self):
        path = Path(self.tmp.name) / "evil.pth"
        self.torch.save({"payload": _NotATensor()}, path)
        with self.assertRaises(Exception):
            ax.AudxModel(path, "cpu")

    def test_wrong_state_dicts_are_reported(self):
        good = {name: self.torch.zeros(shape) for name, shape in ax.EXPECTED_SHAPES.items()}
        ax.check_state_dict(good)
        cases = {
            "missing": {k: v for k, v in good.items() if k != "fc.bias"},
            "unexpected": {**good, "extra.weight": self.torch.zeros(1)},
            "wrong_shape": {**good, "fc.bias": self.torch.zeros(256)},
        }
        for label, state in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, label):
                ax.check_state_dict(state)
        with self.assertRaisesRegex(ValueError, "state_dict"):
            ax.check_state_dict([1, 2])

    def test_poc_wrapped_checkpoint_loads_and_other_wrappers_are_refused(self):
        path = Path(self.tmp.name) / "wrapped.pth"
        meta = {"architecture": "audx_cnn_lstm_9m", "sample_rate": 16000, "n_fft": 512, "hop_length": 256}
        self.torch.save({"state_dict": self.state, **meta}, path)
        x = self._signal(4_000)
        np.testing.assert_array_equal(ax.AudxModel(path, "cpu")(x), self.model(x))
        cases = {
            "non-metadata": {"state_dict": self.state, "optimizer": {"lr": self.torch.zeros(1)}},
            "not an AudX state_dict": {"model": self.state},
        }
        for message, payload in cases.items():
            with self.subTest(message=message):
                self.torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, message):
                    ax.AudxModel(path, "cpu")

    # ── step(): one stream piece by piece ─────────────────────────────────────

    def _stream(self, model, x, sizes, state=None):
        state = state or model.new_state()
        parts, pos, i = [], 0, 0
        while pos < x.size:
            size = sizes[i % len(sizes)]
            parts.append(model.step(state, x[pos : pos + size]))
            pos += size
            i += 1
            self.assertLessEqual(state.received - state.emitted, ax.LOOKAHEAD_SAMPLES + max(sizes))
        parts.append(model.step(state, np.zeros(0, dtype=np.float32), final=True))
        return np.concatenate(parts)

    def test_step_equals_the_whole_signal_for_any_piece_sizes(self):
        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0)
        for n in (1, 255, 256, 257, 1023, 1024, 1025, 9_001):
            for sizes in ([320], [1280], [1, 333, 4001, 77], [n]):
                with self.subTest(n=n, sizes=sizes):
                    x = self._signal(n, seed=n)
                    y = self._stream(model, x, sizes)
                    self.assertEqual(y.shape, (n,))
                    self.assertEqual(y.dtype, np.float32)
                    np.testing.assert_allclose(y, model(x), atol=2e-5)

    def test_step_output_lags_at_most_the_lookahead(self):
        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0)
        state = model.new_state()
        x = self._signal(12_800)  # whole 80 ms pieces
        for pos in range(0, x.size, 1280):
            model.step(state, x[pos : pos + 1280])
            self.assertLessEqual(state.received - state.emitted, ax.LOOKAHEAD_SAMPLES)
            self.assertEqual(state.received - state.emitted, 3 * ax.HOP)  # pieces on the STFT grid: 48 ms

    def test_streams_keep_separate_states(self):
        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0)
        a, b = self._signal(6_155, seed=4), self._signal(7_217, seed=5)
        sa, sb = model.new_state(), model.new_state()
        out_a, out_b = [], []
        for pos in range(0, 8_000, 640):
            out_a.append(model.step(sa, a[pos : pos + 640]))
            out_b.append(model.step(sb, b[pos : pos + 640]))
        out_a.append(model.step(sa, np.zeros(0, np.float32), final=True))
        out_b.append(model.step(sb, np.zeros(0, np.float32), final=True))
        np.testing.assert_allclose(np.concatenate(out_a), model(a), atol=2e-5)
        np.testing.assert_allclose(np.concatenate(out_b), model(b), atol=2e-5)

    def test_step_rejects_bad_input_and_use_after_final(self):
        state = self.model.new_state()
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.model.step(state, np.array([0.0, np.nan], dtype=np.float32))
        fresh = self.model.new_state()
        self.assertEqual(self.model.step(fresh, np.zeros(0, np.float32), final=True).size, 0)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            self.model.step(fresh, np.zeros(10, np.float32))

    def test_stateful_preprocessor_with_audx_is_the_whole_signal_delayed(self):
        import asyncio

        from nemotron_testbed.audio_preprocess_stateful import StatefulStreamPreprocessor
        from nemotron_testbed.config import AudioPreprocessSettings
        from nemotron_testbed.domain import AudioPreprocessResult, StreamKey, SubmitStatus

        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0)
        settings = AudioPreprocessSettings(
            enabled=True, mode="audx", left_context_ms=0, hop_ms=80, right_context_ms=64,
            crossfade_ms=0, compute_budget_ms=100,
        )
        pre = None

        def submit(job):
            pre.deliver(AudioPreprocessResult(job.key, job.region_id, model.step(job.state, job.window, job.final)))
            return SubmitStatus.ACCEPTED

        pre = StatefulStreamPreprocessor(StreamKey("r", "s", "t"), settings, submit, new_state=model.new_state)
        x = self._signal(20_000, seed=9)
        out = [pre.process(x[pos : pos + 320]).enhanced for pos in range(0, x.size, 320)]
        out.append(asyncio.run(pre.flush()).enhanced)
        y = np.concatenate(out)
        d = pre.delay_samples
        np.testing.assert_array_equal(y[:d], 0.0)
        np.testing.assert_allclose(y[d:], model(x), atol=2e-5)
        self.assertEqual(pre.stats["enhanced"], -(-x.size // 1280))
        pre.close()

    def test_agc_only_boosts_and_divides_its_gain_back_out(self):
        off = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0)
        loud = self._signal(8_000, seed=7) * 3.0  # ~-10 dBFS, above the -28 dBFS target: gain stays 1
        np.testing.assert_allclose(self._stream(self.model, loud, [1280]), self._stream(off, loud, [1280]), atol=2e-5)
        quiet = self._signal(8_000, seed=8) * 0.03  # ~-50 dBFS
        state = self.model.new_state()
        gains = self.model._agc(state, quiet)
        self.assertEqual(gains[: ax.AGC_BLOCK].max(), 1.0)  # nothing measured before the first block
        expected = min(10 ** ((-28 + 50.5) / 20), 10 ** 1.5)
        self.assertAlmostEqual(float(gains[-1]), expected, delta=0.15 * expected)
        self.assertTrue(np.all(np.diff(gains[: 2 * ax.AGC_BLOCK]) >= 0))  # ramped, not stepped
        silent = np.full(8_000, 1e-5, dtype=np.float32)  # below the -65 dBFS floor: never boosted
        np.testing.assert_array_equal(self.model._agc(self.model.new_state(), silent), 1.0)
        self.assertEqual(off._agc(off.new_state(), quiet), None)

    # ── section 9 of docs/audx-stateful-design.md: AGC ceiling, clamp of the model output ──

    def _gains(self, model, x, sizes):
        """_agc over x in pieces, keeping state.received as step() does."""
        state = model.new_state()
        parts, pos, i = [], 0, 0
        while pos < x.size:
            piece = x[pos : pos + sizes[i % len(sizes)]]
            parts.append(model._agc(state, piece))
            state.received += piece.size
            pos += piece.size
            i += 1
        return np.concatenate(parts)

    def test_agc_ceiling_keeps_the_boosted_block_below_it(self):
        # as on the first real trace: a -61 dBFS click sets the AGC level, then loud speech arrives
        block = ax.AGC_BLOCK
        rng = np.random.default_rng(11)
        click = (rng.standard_normal(block) * 10 ** (-61 / 20)).astype(np.float32)
        start = 11 * block
        capped = ax.AudxModel(self.checkpoint, "cpu", agc_ceiling_dbfs=-16.0)

        def signal(speech_dbfs):
            speech = (rng.standard_normal(8 * block) * 10 ** (speech_dbfs / 20)).astype(np.float32)
            return np.concatenate([click, np.zeros(10 * block, np.float32), speech])

        def block_dbfs(x, gains):
            y = (x[start:] * gains[start:]).astype(np.float64).reshape(-1, block)
            return 10 * np.log10(np.mean(y**2, axis=1))

        x = signal(-22.0)  # between the -28 dBFS target and the ceiling
        for sizes in ([1280], [100, 333]):  # whole 20 ms blocks, and blocks split across step() calls
            with self.subTest(sizes=sizes):
                self.assertGreater(block_dbfs(x, self._gains(self.model, x, sizes))[0], 0.0)  # no ceiling: overload
                levels = block_dbfs(x, self._gains(capped, x, sizes))
                self.assertLessEqual(float(levels.max()), -16.0 + 1e-3)
                # the first block is boosted up to the ceiling; a split block keeps the lowest cap of its parts
                # for the rest of the block (a short first part estimates its power loosely), so it may stay lower
                self.assertGreater(float(levels[0]), -16.5 if sizes == [1280] else -20.0)
        # louder than the ceiling: boost-only, so not boosted at all (and never cut)
        x = signal(-12.0)
        np.testing.assert_array_equal(self._gains(capped, x, [1280])[start:], 1.0)
        # a ceiling that never binds changes nothing
        quiet = self._signal(8_000, seed=8) * 0.03  # ~-50 dBFS, boosted to about -28 dBFS
        np.testing.assert_array_equal(self._gains(capped, quiet, [1280]), self._gains(self.model, quiet, [1280]))

    def _constant_estimate(self, value, **kwargs):
        """An AudxModel whose network predicts `value` for every bin, whatever its input."""
        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0, **kwargs)
        with self.torch.no_grad():
            model.net.fc.weight.zero_()
            model.net.fc.bias.fill_(value)
        return model

    def test_clamp_keeps_every_bin_between_the_floor_and_the_input(self):
        x = self._signal(9_001, seed=12)
        floor = 10 ** (-12 / 20)
        for value, kwargs, expected in (
            (1e3, {"clamp_to_input": True}, x),  # never above the input bin: the input comes back
            (0.0, {"max_attenuation_db": 12.0}, x * floor),  # never cut by more than 12 dB
            (1e3, {"clamp_to_input": True, "max_attenuation_db": 12.0}, x),
            (0.0, {"clamp_to_input": True, "max_attenuation_db": 12.0}, x * floor),
        ):
            model = self._constant_estimate(value, **kwargs)
            with self.subTest(value=value, **kwargs):
                np.testing.assert_allclose(model(x), expected, atol=2e-5)
                np.testing.assert_allclose(self._stream(model, x, [1280]), expected, atol=2e-5)
        # off: what the network says
        np.testing.assert_allclose(self._constant_estimate(0.0)(x), 0.0, atol=1e-6)

    def test_step_equals_the_whole_signal_with_the_clamp(self):
        model = ax.AudxModel(self.checkpoint, "cpu", agc_max_gain_db=0, clamp_to_input=True, max_attenuation_db=12)
        x = self._signal(9_001, seed=13)
        for sizes in ([1280], [1, 333, 4001, 77]):
            with self.subTest(sizes=sizes):
                np.testing.assert_allclose(self._stream(model, x, sizes), model(x), atol=2e-5)

    def test_engine_loader_passes_the_section_9_settings(self):
        from nemotron_testbed.audio_preprocess_runtime import make_model_loader
        from nemotron_testbed.config import AudioPreprocessSettings

        base = {"enabled": True, "model": "audx", "mode": "audx", "audx_checkpoint": str(self.checkpoint)}
        on = make_model_loader(AudioPreprocessSettings(
            **base, audx_agc_ceiling_dbfs=-16.0, audx_clamp_to_input=True, audx_max_attenuation_db=12.0,
        ))("cpu")
        self.assertAlmostEqual(on._agc_ceiling, 10 ** (-16 / 10))
        self.assertTrue(on._clamp_to_input)
        self.assertAlmostEqual(on._min_ratio, 10 ** (-12 / 20))
        off = make_model_loader(AudioPreprocessSettings(**base))("cpu")
        self.assertEqual((off._agc_ceiling, off._clamp_to_input, off._min_ratio), (None, False, None))

    def test_missing_file_and_unavailable_cuda(self):
        with self.assertRaises(FileNotFoundError):
            ax.AudxModel(Path(self.tmp.name) / "none.pth", "cpu")
        if not self.torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "cuda"):
                ax.AudxModel(self.checkpoint, "cuda:0")
        with self.assertRaises(ValueError):
            ax.AudxModel(self.checkpoint, "tpu")


class AudxConstantsTest(unittest.TestCase):
    def test_lookahead_is_64_ms(self):
        self.assertEqual(ax.LOOKAHEAD_SAMPLES * 1000 // ax.SAMPLE_RATE, 64)
        self.assertEqual(ax.FREQ_BINS, 257)


if __name__ == "__main__":
    unittest.main()
