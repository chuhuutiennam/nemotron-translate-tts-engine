import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_preproc_gpu.py"
_spec = importlib.util.spec_from_file_location("probe_preproc_gpu", _SCRIPT)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class NvidiaLibPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.site = self._tmp.name
        for sub in ("cublas", "cudnn", "cufft"):
            (Path(self.site) / "nvidia" / sub / "lib").mkdir(parents=True)

    def lib(self, sub):
        return str(Path(self.site) / "nvidia" / sub / "lib")

    def test_dirs_are_only_existing_ones_in_run_gpu_server_order(self):
        self.assertEqual(
            probe.nvidia_lib_dirs([self.site]),
            [self.lib("cudnn"), self.lib("cublas"), self.lib("cufft")],
        )

    def test_no_nvidia_dirs_means_nothing_to_add(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(probe.nvidia_lib_dirs([empty]), [])
            self.assertIsNone(probe.ld_library_path_with_nvidia("/usr/lib", [empty]))

    def test_missing_dirs_go_in_front_of_the_current_value(self):
        new = probe.ld_library_path_with_nvidia(os.pathsep.join(["/usr/lib", "/opt/x"]), [self.site])
        self.assertEqual(
            new.split(os.pathsep),
            [self.lib("cudnn"), self.lib("cublas"), self.lib("cufft"), "/usr/lib", "/opt/x"],
        )

    def test_empty_current_value_has_no_stray_separator(self):
        new = probe.ld_library_path_with_nvidia("", [self.site])
        self.assertEqual(new.split(os.pathsep), [self.lib("cudnn"), self.lib("cublas"), self.lib("cufft")])

    def test_only_the_absent_dirs_are_added(self):
        current = os.pathsep.join(["/usr/lib", self.lib("cublas")])
        new = probe.ld_library_path_with_nvidia(current, [self.site])
        self.assertEqual(
            new.split(os.pathsep),
            [self.lib("cudnn"), self.lib("cufft"), "/usr/lib", self.lib("cublas")],
        )

    def test_nothing_to_add_when_all_dirs_are_present(self):
        current = os.pathsep.join(probe.nvidia_lib_dirs([self.site]))
        self.assertIsNone(probe.ld_library_path_with_nvidia(current, [self.site]))


class ReexecTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        (Path(self._tmp.name) / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
        patch = mock.patch.object(probe, "site_dirs", return_value=[self._tmp.name])
        patch.start()
        self.addCleanup(patch.stop)
        self.execv = mock.Mock()

    def run_reexec(self, environ):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            done = probe.maybe_reexec_with_nvidia_libs(environ, self.execv)
        return done, out.getvalue()

    def test_reexecs_once_with_the_new_path_and_a_guard(self):
        environ = {"LD_LIBRARY_PATH": "/usr/lib"}
        with mock.patch.object(probe.sys, "argv", ["probe_preproc_gpu.py", "--device", "cuda:1"]):
            done, out = self.run_reexec(environ)
        self.assertTrue(done)
        self.assertIn("re-running", out)
        self.assertEqual(
            environ["LD_LIBRARY_PATH"],
            os.pathsep.join([str(Path(self._tmp.name) / "nvidia" / "cudnn" / "lib"), "/usr/lib"]),
        )
        self.assertEqual(environ[probe.REEXEC_GUARD], "1")
        self.execv.assert_called_once_with(
            probe.sys.executable, [probe.sys.executable, "probe_preproc_gpu.py", "--device", "cuda:1"]
        )

    def test_guard_prevents_a_second_exec(self):
        environ = {probe.REEXEC_GUARD: "1"}
        done, _ = self.run_reexec(environ)
        self.assertFalse(done)
        self.execv.assert_not_called()
        self.assertNotIn("LD_LIBRARY_PATH", environ)

    def test_no_exec_when_the_path_is_already_complete(self):
        lib = str(Path(self._tmp.name) / "nvidia" / "cudnn" / "lib")
        environ = {"LD_LIBRARY_PATH": lib}
        done, out = self.run_reexec(environ)
        self.assertFalse(done)
        self.assertEqual(out, "")
        self.execv.assert_not_called()
        self.assertNotIn(probe.REEXEC_GUARD, environ)


if __name__ == "__main__":
    unittest.main()
