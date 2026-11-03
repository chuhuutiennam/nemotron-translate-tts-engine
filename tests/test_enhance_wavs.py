import contextlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "enhance_wavs.py"
_spec = importlib.util.spec_from_file_location("enhance_wavs", _SCRIPT)
ew = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ew)

try:
    import scipy.signal  # noqa: F401

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def write_wav(path: Path, pcm: np.ndarray, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.astype("<i2").tobytes())


def read_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2"), wav.getframerate()


class ZeroRunsTest(unittest.TestCase):
    def test_only_runs_of_min_length_are_returned_half_open(self):
        pcm = np.ones(300, dtype=np.int16)
        pcm[10:20] = 0
        pcm[100:170] = 0
        self.assertEqual(ew.zero_runs(pcm, 64), [(100, 170)])
        self.assertEqual(ew.zero_runs(pcm, 10), [(10, 20), (100, 170)])

    def test_runs_touching_both_ends(self):
        pcm = np.ones(200, dtype=np.int16)
        pcm[:80] = 0
        pcm[-70:] = 0
        self.assertEqual(ew.zero_runs(pcm), [(0, 80), (130, 200)])

    def test_no_zeros_and_all_zeros(self):
        self.assertEqual(ew.zero_runs(np.ones(100, dtype=np.int16)), [])
        self.assertEqual(ew.zero_runs(np.zeros(100, dtype=np.int16)), [(0, 100)])
        self.assertEqual(ew.zero_runs(np.zeros(0, dtype=np.int16)), [])


class DitherTest(unittest.TestCase):
    def test_only_the_runs_change_and_it_is_deterministic(self):
        pcm = np.full(400, 7, dtype=np.int16)
        pcm[100:300] = 0
        runs = ew.zero_runs(pcm)
        out = ew.add_dither(pcm, runs, 3.0)
        np.testing.assert_array_equal(out[:100], pcm[:100])
        np.testing.assert_array_equal(out[300:], pcm[300:])
        self.assertGreater(int(np.count_nonzero(out[100:300])), 100)
        np.testing.assert_array_equal(out, ew.add_dither(pcm, runs, 3.0))
        self.assertFalse(np.array_equal(out, ew.add_dither(pcm, runs, 3.0, seed=1)))

    def test_amplitude_follows_sigma(self):
        pcm = np.zeros(20_000, dtype=np.int16)
        out = ew.add_dither(pcm, ew.zero_runs(pcm), 10.0)
        self.assertAlmostEqual(float(out.std()), 10.0, delta=0.5)
        self.assertEqual(out.dtype, np.int16)


class SlidingCrossfadeTest(unittest.TestCase):
    def test_chunk_starts_drop_a_tail_not_longer_than_the_overlap(self):
        self.assertEqual(ew.chunk_starts(100, 40, 10), [0, 30, 60])
        self.assertEqual(ew.chunk_starts(101, 40, 10), [0, 30, 60, 90])
        self.assertEqual(ew.chunk_starts(20, 40, 10), [0])
        self.assertEqual(ew.chunk_starts(0, 40, 10), [])
        self.assertEqual(len(ew.sliding_crossfade(np.zeros(0), 40, 10, lambda part: part)), 0)

    def test_identity_process_reproduces_the_input(self):
        rng = np.random.default_rng(0)
        for n in (5, 39, 40, 100, 101, 1234):
            x = rng.normal(size=n)
            y = ew.sliding_crossfade(x, 40, 10, lambda part: part.copy(), silent_rms=0.0)
            np.testing.assert_allclose(y, x, atol=1e-12, err_msg=f"n={n}")

    def test_silent_chunks_are_not_processed_and_give_zeros(self):
        x = np.concatenate([np.ones(60), np.zeros(60), np.ones(60)])
        seen = []

        def process(part):
            seen.append(len(part))
            return part * 2.0

        y = ew.sliding_crossfade(x, 40, 10, process, silent_rms=1e-6)
        self.assertLess(len(seen), len(ew.chunk_starts(len(x), 40, 10)))
        np.testing.assert_array_equal(y[65:115], 0.0)
        self.assertGreater(y[:50].min(), 0.0)

    def test_wrong_shape_or_non_finite_output_raises(self):
        x = np.ones(100)
        with self.assertRaises(ValueError):
            ew.sliding_crossfade(x, 40, 10, lambda part: part[:-1])
        with self.assertRaises(ValueError):
            ew.sliding_crossfade(x, 40, 10, lambda part: part * np.nan)


@unittest.skipUnless(HAVE_SCIPY, "scipy is not installed")
class ResampleTest(unittest.TestCase):
    def test_length_is_exactly_n_and_values_are_int16(self):
        for n in (1, 160, 1600, 16_001):
            y44 = np.zeros(int(n * 441 / 160) + 5)
            out = ew.resampled_pcm16(y44, n)
            self.assertEqual(out.shape, (n,))
            self.assertEqual(out.dtype, np.int16)

    def test_short_input_is_padded_and_loud_input_is_clipped(self):
        out = ew.resampled_pcm16(np.full(441, 5.0), 400)
        self.assertEqual(len(out), 400)
        self.assertEqual(int(out.max()), 32767)

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            ew.resampled_pcm16(np.full(441, np.nan), 160)

    def test_round_trip_keeps_a_tone(self):
        t = np.arange(16_000) / 16_000
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        from scipy.signal import resample_poly

        up = resample_poly(tone, 441, 160, window=ew.RESAMPLE_WINDOW)
        back = ew.resampled_pcm16(up, len(tone)).astype(np.float64) / 32768.0
        self.assertLess(float(np.abs(back[200:-200] - tone[200:-200]).max()), 5e-3)


class PreprocessClipTest(unittest.TestCase):
    def setUp(self):
        self.pcm = np.full(1000, 100, dtype=np.int16)
        self.pcm[200:400] = 0

    def test_mask_marks_the_input_zero_runs(self):
        out, mask = ew.preprocess_clip(self.pcm, lambda p: p.copy(), 0.0, False)
        np.testing.assert_array_equal(out, self.pcm)
        self.assertEqual(int(mask.sum()), 200)
        self.assertTrue(mask[200:400].all())

    def test_rezero_forces_the_masked_samples_to_zero(self):
        out, _ = ew.preprocess_clip(self.pcm, lambda p: p + 5, 0.0, True)
        self.assertTrue((out[200:400] == 0).all())
        self.assertTrue((out[:200] == 105).all())
        out, _ = ew.preprocess_clip(self.pcm, lambda p: p + 5, 0.0, False)
        self.assertTrue((out[200:400] == 5).all())

    def test_dither_reaches_the_processor_but_not_the_mask(self):
        received = []

        def processor(p):
            received.append(p.copy())
            return p

        _, mask = ew.preprocess_clip(self.pcm, processor, 3.0, False)
        self.assertGreater(int(np.count_nonzero(received[0][200:400])), 50)
        self.assertEqual(int(mask.sum()), 200)

    def test_length_change_raises(self):
        with self.assertRaises(ValueError):
            ew.preprocess_clip(self.pcm, lambda p: p[:-1], 0.0, False)


class AssetsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_source(self, root: Path, body: str = "x = 1\n") -> Path:
        (root / "resemble_enhance" / "sub").mkdir(parents=True)
        (root / "resemble_enhance" / "a.py").write_text(body, encoding="utf-8")
        (root / "resemble_enhance" / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
        (root / "resemble_enhance" / "notes.txt").write_text("ignored", encoding="utf-8")
        return root

    def make_tarball(self, source: Path, name: str = "pkg-abc") -> Path:
        archive = self.tmp / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=name)
        return archive

    def test_tree_hash_is_stable_and_follows_python_files_only(self):
        a = self.make_source(self.tmp / "a")
        b = self.make_source(self.tmp / "b")
        self.assertEqual(ew.tree_sha256(a), ew.tree_sha256(b))
        (b / "resemble_enhance" / "notes.txt").write_text("changed", encoding="utf-8")
        self.assertEqual(ew.tree_sha256(a), ew.tree_sha256(b))
        (b / "resemble_enhance" / "a.py").write_text("x = 2\n", encoding="utf-8")
        self.assertNotEqual(ew.tree_sha256(a), ew.tree_sha256(b))

    def test_download_returns_the_sha_and_leaves_no_part_file(self):
        src = self.tmp / "payload.bin"
        src.write_bytes(b"abc" * 1000)
        dest = self.tmp / "out" / "payload.bin"
        self.assertEqual(ew.download(src.as_uri(), dest), ew.sha256_file(src))
        self.assertEqual(dest.read_bytes(), src.read_bytes())
        self.assertFalse(dest.with_name("payload.bin.part").exists())

    def test_download_failure_raises_after_the_retries_and_writes_nothing(self):
        dest = self.tmp / "missing.bin"
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(RuntimeError):
            ew.download((self.tmp / "nope.bin").as_uri(), dest, retries=2)
        self.assertEqual(err.getvalue().count("failed"), 2)
        self.assertFalse(dest.exists())
        self.assertFalse(dest.with_name("missing.bin.part").exists())

    def test_fetch_file_skips_a_valid_copy_and_replaces_a_wrong_one(self):
        src = self.tmp / "payload.bin"
        src.write_bytes(b"payload")
        sha = ew.sha256_file(src)
        dest = self.tmp / "dest.bin"
        self.assertTrue(ew.fetch_file(src.as_uri(), dest, sha))
        self.assertFalse(ew.fetch_file((self.tmp / "gone.bin").as_uri(), dest, sha))
        dest.write_bytes(b"corrupt")
        self.assertTrue(ew.fetch_file(src.as_uri(), dest, sha))
        self.assertEqual(dest.read_bytes(), b"payload")

    def test_fetch_file_rejects_a_wrong_sha_and_removes_the_file(self):
        src = self.tmp / "payload.bin"
        src.write_bytes(b"payload")
        dest = self.tmp / "dest.bin"
        with self.assertRaisesRegex(RuntimeError, "sha256"):
            ew.fetch_file(src.as_uri(), dest, "0" * 64)
        self.assertFalse(dest.exists())

    def test_fetch_enhancer_source_extracts_and_verifies_the_tree(self):
        source = self.make_source(self.tmp / "src")
        expected = ew.tree_sha256(source)
        archive = self.make_tarball(source)
        assets = self.tmp / "assets"
        self.assertTrue(ew.fetch_enhancer_source(assets, archive.as_uri(), expected))
        self.assertEqual(ew.tree_sha256(assets / "resemble-enhance"), expected)
        self.assertFalse(ew.fetch_enhancer_source(assets, (self.tmp / "gone.tar.gz").as_uri(), expected))
        self.assertEqual([p.name for p in assets.iterdir()], ["resemble-enhance"])

    def test_fetch_enhancer_source_rejects_a_different_tree(self):
        archive = self.make_tarball(self.make_source(self.tmp / "src"))
        assets = self.tmp / "assets"
        with self.assertRaisesRegex(RuntimeError, "tree sha256"):
            ew.fetch_enhancer_source(assets, archive.as_uri(), "0" * 64)
        self.assertFalse((assets / "resemble-enhance").exists())

    def test_fetch_enhancer_source_rejects_unsafe_members(self):
        archive = self.tmp / "evil.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("pkg/../../escape.py")
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            ew.fetch_enhancer_source(self.tmp / "assets", archive.as_uri(), "0" * 64)
        self.assertFalse((self.tmp / "escape.py").exists())

    def test_pinned_constants_look_valid(self):
        self.assertEqual(len(ew.ENHANCER_COMMIT), 40)
        self.assertEqual(len(ew.ONNX_COMMIT), 40)
        for sha in (ew.ENHANCER_TREE_SHA256, ew.ONNX_SHA256, *ew.CHECKPOINT_FILES.values()):
            self.assertRegex(sha, r"^[0-9a-f]{64}$")


class RunBranchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        rng = np.random.default_rng(1)
        self.clips = {}
        self.rows = []
        for index in range(3):
            pcm = rng.integers(-2000, 2000, 8000 + 160 * index).astype(np.int16)
            pcm[1000:1400] = 0
            path = self.tmp / f"clip{index}.wav"
            write_wav(path, pcm)
            self.clips[f"vi-{index:03d}"] = pcm
            self.rows.append(
                {
                    "id": f"vi-{index:03d}",
                    "wav": str(path),
                    "duration_s": f"{len(pcm) / 16000:.3f}",
                    "content_type": "audio/wav",
                    "reference": f"câu {index}",
                    "lang": "vi",
                }
            )

    def test_narrowband_false_reads_the_exact_samples(self):
        for row in self.rows:
            np.testing.assert_array_equal(ew.narrowband_pcm16(Path(row["wav"]), False), self.clips[row["id"]])

    def test_narrowband_true_keeps_the_length_and_changes_the_signal(self):
        pcm = self.clips["vi-000"]
        out = ew.narrowband_pcm16(Path(self.rows[0]["wav"]), True)
        self.assertEqual(out.shape, pcm.shape)
        self.assertFalse(np.array_equal(out, pcm))

    def test_run_branch_writes_wavs_rows_and_stats(self):
        out_dir = self.tmp / "out"
        rows, stats = ew.run_branch(self.rows, out_dir, lambda p: ew.narrowband_pcm16(p, False), lambda pcm: (pcm // 2).astype(np.int16))
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in self.rows])
        for src, new in zip(self.rows, rows):
            self.assertEqual(new["reference"], src["reference"])
            self.assertEqual(new["duration_s"], src["duration_s"])
            self.assertEqual(Path(new["wav"]), (out_dir / "wav" / f"{src['id']}.wav").resolve())
            pcm, rate = read_wav_pcm(Path(new["wav"]))
            self.assertEqual(rate, 16000)
            np.testing.assert_array_equal(pcm, (self.clips[src["id"]] // 2).astype(np.int16))
        self.assertEqual(len(stats), 3)
        self.assertAlmostEqual(stats[0]["energy_db"], -6.0, delta=0.1)
        self.assertGreater(stats[0]["zero_run_pct"], 4.0)
        self.assertIn("3 clips", ew.summarize("x", stats))

    def test_cmd_run_writes_manifest_stats_and_config(self):
        manifest = self.tmp / "vi.tsv"
        ew.eval_wer.save_manifest(manifest, self.rows)
        out_dir = self.tmp / "enh" / "denoiser"
        original = ew.build_processor
        ew.build_processor = lambda args: (lambda pcm: pcm.copy(), {"backend": "fake", "branch": args.branch})
        self.addCleanup(setattr, ew, "build_processor", original)
        args = ew.build_parser().parse_args(
            ["run", "--assets-dir", str(self.tmp), "--manifest", str(manifest), "--out", str(out_dir), "--branch", "denoiser", "--no-narrowband", "--limit", "2"]
        )
        with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
            args.func(args)
        self.assertIn("wrote 2 clips", stdout.getvalue())
        new_rows = ew.eval_wer.load_manifest(out_dir / "vi.tsv")
        self.assertEqual([r["id"] for r in new_rows], ["vi-000", "vi-001"])
        self.assertTrue(all(Path(r["wav"]).is_file() for r in new_rows))
        self.assertEqual(len((out_dir / "stats.jsonl").read_text(encoding="utf-8").splitlines()), 2)
        config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["info"]["backend"], "fake")
        self.assertEqual(config["args"]["branch"], "denoiser")
        self.assertIs(config["args"]["narrowband"], False)

    def test_negative_dither_is_rejected(self):
        manifest = self.tmp / "vi.tsv"
        ew.eval_wer.save_manifest(manifest, self.rows)
        args = ew.build_parser().parse_args(
            ["run", "--assets-dir", str(self.tmp), "--manifest", str(manifest), "--out", str(self.tmp / "o"), "--branch", "enhancer", "--dither-sigma", "-1"]
        )
        with self.assertRaises(SystemExit):
            args.func(args)


class ParserTest(unittest.TestCase):
    def parse(self, *argv):
        with contextlib.redirect_stderr(io.StringIO()):
            return ew.build_parser().parse_args(argv)

    def test_run_defaults_follow_the_agreed_enhancer_setting(self):
        args = self.parse("run", "--assets-dir", "a", "--manifest", "m.tsv", "--out", "o", "--branch", "enhancer")
        self.assertEqual((args.nfe, args.lambd, args.tau), (32, 1.0, 0.5))
        self.assertEqual((args.chunk_seconds, args.overlap_seconds), (10.0, 1.0))
        self.assertTrue(args.narrowband)
        self.assertEqual(args.denoiser_backend, "onnx")
        self.assertEqual((args.dither_sigma, args.rezero, args.seed, args.device), (0.0, False, 0, "auto"))

    def test_bench_defaults_time_nfe_32_and_64_on_2_s_and_10_s_chunks(self):
        args = self.parse("bench", "--assets-dir", "a", "--manifest", "m.tsv")
        self.assertEqual(args.nfe, [32, 64])
        self.assertEqual(args.chunk_seconds, [2.0, 10.0])

    def test_fetch_defaults_to_both_asset_groups(self):
        self.assertEqual(self.parse("fetch", "--assets-dir", "a").what, ["enhancer", "onnx"])

    def test_missing_required_arguments_exit(self):
        with self.assertRaises(SystemExit):
            self.parse("run", "--assets-dir", "a")
        with self.assertRaises(SystemExit):
            self.parse("run", "--assets-dir", "a", "--manifest", "m", "--out", "o", "--branch", "other")


if __name__ == "__main__":
    unittest.main()
