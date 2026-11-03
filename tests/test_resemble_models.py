import tempfile
import unittest
from pathlib import Path

import numpy as np

from nemotron_testbed import resemble_models as rm


def noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


class ParseDeviceTest(unittest.TestCase):
    def test_accepted_names(self):
        self.assertEqual(rm.parse_device("cpu"), ("cpu", 0))
        self.assertEqual(rm.parse_device(" CPU "), ("cpu", 0))
        self.assertEqual(rm.parse_device("cuda"), ("cuda", 0))
        self.assertEqual(rm.parse_device("cuda:3"), ("cuda", 3))

    def test_rejected_names(self):
        for bad in ("", "gpu", "cuda:", "cuda:x", "cuda:-1", "mps"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rm.parse_device(bad)


class OrtSessionConfigTest(unittest.TestCase):
    def test_cuda_pins_threads_and_cudnn_like_piper(self):
        attributes, providers = rm.ort_session_config("cuda", 1)
        self.assertEqual(attributes["intra_op_num_threads"], rm.ORT_CUDA_INTRA_THREADS)
        self.assertEqual(attributes["inter_op_num_threads"], 1)
        self.assertEqual(
            providers[0],
            (
                "CUDAExecutionProvider",
                {"device_id": 1, "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0"},
            ),
        )
        self.assertEqual(providers[1:], ["CPUExecutionProvider"])

    def test_cpu_has_cpu_provider_only_and_default_threads(self):
        attributes, providers = rm.ort_session_config("cpu", 0)
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertNotIn("intra_op_num_threads", attributes)
        self.assertNotIn("inter_op_num_threads", attributes)

    def test_module_settings_are_read_at_call_time(self):
        saved = dict(rm.ORT_CUDA_PROVIDER_OPTIONS), rm.ORT_CUDA_INTRA_THREADS, rm.ORT_LOG_SEVERITY
        try:
            rm.ORT_CUDA_PROVIDER_OPTIONS["cudnn_conv_algo_search"] = "EXHAUSTIVE"
            rm.ORT_CUDA_INTRA_THREADS = 4
            rm.ORT_LOG_SEVERITY = 1
            attributes, providers = rm.ort_session_config("cuda", 0)
        finally:
            rm.ORT_CUDA_PROVIDER_OPTIONS.clear()
            rm.ORT_CUDA_PROVIDER_OPTIONS.update(saved[0])
            rm.ORT_CUDA_INTRA_THREADS, rm.ORT_LOG_SEVERITY = saved[1], saved[2]
        self.assertEqual(attributes["intra_op_num_threads"], 4)
        self.assertEqual(attributes["log_severity_level"], 1)
        self.assertEqual(providers[0][1]["cudnn_conv_algo_search"], "EXHAUSTIVE")


class AlignmentTest(unittest.TestCase):
    def test_shift_earlier_moves_both_directions(self):
        y = np.arange(10, dtype=np.float64)
        np.testing.assert_array_equal(rm.shift_earlier(y, 0), y)
        np.testing.assert_array_equal(rm.shift_earlier(y, 3), [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])
        np.testing.assert_array_equal(rm.shift_earlier(y, -2), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(rm.shift_earlier(y, 3)), len(y))

    def test_estimate_lag_finds_delay_and_advance(self):
        ref = noise(8000)
        for lag in (-300, -1, 0, 5, 220):
            with self.subTest(lag=lag):
                y = np.zeros_like(ref)
                if lag > 0:
                    y[lag:] = ref[:-lag]
                elif lag < 0:
                    y[:lag] = ref[-lag:]
                else:
                    y = ref.copy()
                self.assertEqual(rm.estimate_lag(ref, y, 640), lag)

    def test_aligning_with_estimated_lag_restores_signal(self):
        ref = noise(8000, seed=2)
        y = np.zeros_like(ref)
        y[100:] = ref[:-100]
        lag = rm.estimate_lag(ref, y, 640)
        fixed = rm.shift_earlier(y, lag)
        np.testing.assert_allclose(fixed[:-100], ref[:-100])

    def test_estimate_lag_outside_search_range_is_not_followed(self):
        ref = noise(8000, seed=3)
        y = np.zeros_like(ref)
        y[2000:] = ref[:-2000]
        self.assertEqual(rm.estimate_lag(ref, y, 100), 0)

    def test_estimate_lag_unrelated_or_silent_returns_zero(self):
        self.assertEqual(rm.estimate_lag(noise(4000, 1), noise(4000, 2), 320), 0)
        self.assertEqual(rm.estimate_lag(np.zeros(4000), noise(4000), 320), 0)
        self.assertEqual(rm.estimate_lag(noise(4000), np.zeros(4000), 320), 0)
        self.assertEqual(rm.estimate_lag(np.zeros(0), np.zeros(0), 320), 0)
        self.assertEqual(rm.estimate_lag(noise(100), noise(100), 0), 0)

    def test_fit_length(self):
        self.assertEqual(len(rm._fit_length(np.ones(5), 8)), 8)
        np.testing.assert_array_equal(rm._fit_length(np.ones(5), 8)[5:], 0)
        np.testing.assert_array_equal(rm._fit_length(np.arange(5.0), 3), [0, 1, 2])


class LoaderErrorsTest(unittest.TestCase):
    def kwargs(self):
        return dict(
            enhancer_nfe=8,
            enhancer_lambd=0.9,
            enhancer_tau=0.5,
            enhancer_align=True,
            enhancer_align_max_ms=40,
        )

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            rm.load_window_model("magic", "/nowhere", "cpu", **self.kwargs())

    def test_missing_denoiser_onnx_fails_before_importing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                rm.load_window_model("denoiser", tmp, "cpu", **self.kwargs())

    def test_missing_enhancer_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                rm.load_window_model("enhancer", tmp, "cpu", **self.kwargs())
            self.assertIn("resemble-enhance", str(ctx.exception))

    def test_missing_checkpoint_file_never_reaches_the_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "resemble-enhance" / "resemble_enhance").mkdir(parents=True)
            checkpoint = root / "enhancer_stage2"
            for rel in rm.ENHANCER_CHECKPOINT_FILES[:-1]:
                path = checkpoint / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            with self.assertRaises(FileNotFoundError) as ctx:
                rm.load_window_model("enhancer", tmp, "cpu", **self.kwargs())
            self.assertIn("mp_rank_00_model_states.pt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
