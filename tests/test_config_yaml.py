"""Tests for YAML-based config loading in Settings."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nemotron_testbed.config import (
    Settings,
    _Cfg,
    _discover_config_file,
    _load_yaml_file,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_yaml(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True)


def _clean_env(*drop_keys: str) -> dict[str, str]:
    """Return copy of os.environ minus the specified keys."""
    return {k: v for k, v in os.environ.items() if k not in drop_keys}


_DROP = (
    "ENGINE_CONFIG_FILE", "APP_DIR",
    "MODEL_PATH", "ASR_DEVICE", "ASR_DEVICES",
    "GRPC_SERVER_ADDRESS", "TTS_BACKEND",
    "TTS_EN_MODEL_PATH", "TTS_EN_CONFIG_PATH",
    "TTS_VI_MODEL_PATH", "TTS_VI_CONFIG_PATH",
    "TTS_ENABLED", "TTS_MAGPIE_MODEL_PATH",
    "TTS_MAGPIE_CODEC_MODEL_PATH", "TTS_SPEAKER_INDEX",
    "TRANSLATION_DEVICES", "TRANSLATION_DEVICE",
    "TTS_DEVICES", "TTS_DEVICE",
)


# ── _Cfg unit tests ───────────────────────────────────────────────────────────

class TestCfgStr(unittest.TestCase):
    def _cfg(self, data):
        return _Cfg(data, "test")

    def test_yaml_value_returned(self):
        cfg = self._cfg({"a": {"b": "hello"}})
        with patch.dict(os.environ, {"MY_VAR": "world"}):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "default"), "hello")

    def test_env_fallback_when_key_absent(self):
        cfg = self._cfg({})
        with patch.dict(os.environ, {"MY_VAR": "from_env"}, clear=False):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "default"), "from_env")

    def test_default_when_key_absent_and_no_env(self):
        cfg = self._cfg({})
        env = _clean_env("MY_VAR")
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "fallback"), "fallback")

    def test_yaml_null_returns_default_not_env(self):
        # Explicit null in YAML → code default, env var NOT consulted.
        cfg = self._cfg({"a": {"b": None}})
        with patch.dict(os.environ, {"MY_VAR": "env_val"}, clear=False):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "default"), "default")

    def test_yaml_overrides_env(self):
        cfg = self._cfg({"a": {"b": "from_yaml"}})
        with patch.dict(os.environ, {"MY_VAR": "from_env"}):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "default"), "from_yaml")

    def test_none_data_falls_through(self):
        cfg = _Cfg(None, "env")
        env = _clean_env("MY_VAR")
        env["MY_VAR"] = "from_env"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(cfg.str("a.b", "MY_VAR", "default"), "from_env")


class TestCfgInt(unittest.TestCase):
    def _cfg(self, data):
        return _Cfg(data, "test")

    def test_yaml_int(self):
        cfg = self._cfg({"workers": 4})
        self.assertEqual(cfg.int("workers", "WORKERS", 2), 4)

    def test_yaml_string_int(self):
        cfg = self._cfg({"workers": "8"})
        self.assertEqual(cfg.int("workers", "WORKERS", 2), 8)

    def test_yaml_bool_raises(self):
        cfg = self._cfg({"flag": True})
        with self.assertRaises(ValueError):
            cfg.int("flag", "FLAG", 0)

    def test_yaml_invalid_raises(self):
        cfg = self._cfg({"workers": "not_a_number"})
        with self.assertRaises(ValueError):
            cfg.int("workers", "WORKERS", 2)

    def test_env_fallback(self):
        cfg = self._cfg({})
        with patch.dict(os.environ, {"WORKERS": "16"}, clear=False):
            self.assertEqual(cfg.int("workers", "WORKERS", 2), 16)


class TestCfgFloat(unittest.TestCase):
    def _cfg(self, data):
        return _Cfg(data, "test")

    def test_yaml_float(self):
        cfg = self._cfg({"scale": 1.5})
        self.assertAlmostEqual(cfg.float("scale", "SCALE", 1.0), 1.5)

    def test_yaml_int_coerced_to_float(self):
        cfg = self._cfg({"scale": 2})
        self.assertAlmostEqual(cfg.float("scale", "SCALE", 1.0), 2.0)

    def test_yaml_null_returns_default(self):
        cfg = self._cfg({"scale": None})
        self.assertAlmostEqual(cfg.float("scale", "SCALE", 3.0), 3.0)


class TestCfgBool(unittest.TestCase):
    def _cfg(self, data):
        return _Cfg(data, "test")

    def test_yaml_true(self):
        self.assertTrue(_Cfg({"f": True}, "t").bool("f", "F", False))

    def test_yaml_false(self):
        self.assertFalse(_Cfg({"f": False}, "t").bool("f", "F", True))

    def test_yaml_string_true(self):
        self.assertTrue(_Cfg({"f": "yes"}, "t").bool("f", "F", False))

    def test_yaml_string_false(self):
        self.assertFalse(_Cfg({"f": "off"}, "t").bool("f", "F", True))

    def test_env_fallback(self):
        cfg = _Cfg({}, "test")
        with patch.dict(os.environ, {"ENABLED": "true"}, clear=False):
            self.assertTrue(cfg.bool("enabled", "ENABLED", False))


class TestCfgDevices(unittest.TestCase):
    def _cfg(self, data):
        return _Cfg(data, "test")

    def test_yaml_list(self):
        cfg = self._cfg({"devices": ["cuda:0", "cuda:1"]})
        result = cfg.devices("devices", "DEVICES", "device", "DEVICE")
        self.assertEqual(result, ["cuda:0", "cuda:1"])

    def test_yaml_comma_string(self):
        cfg = self._cfg({"devices": "cuda:0,cuda:1"})
        result = cfg.devices("devices", "DEVICES", "device", "DEVICE")
        self.assertEqual(result, ["cuda:0", "cuda:1"])

    def test_yaml_null_uses_fallback(self):
        cfg = self._cfg({"devices": None, "device": "cuda:3"})
        result = cfg.devices("devices", "DEVICES", "device", "DEVICE")
        self.assertEqual(result, ["cuda:3"])

    def test_env_fallback(self):
        cfg = self._cfg({})
        env = _clean_env("DEVICES", "DEVICE")
        env["DEVICES"] = "cuda:2,cuda:3"
        with patch.dict(os.environ, env, clear=True):
            result = cfg.devices("devices", "DEVICES", "device", "DEVICE")
        self.assertEqual(result, ["cuda:2", "cuda:3"])

    def test_single_device_in_list(self):
        cfg = self._cfg({"devices": ["cpu"]})
        result = cfg.devices("devices", "DEVICES", "device", "DEVICE")
        self.assertEqual(result, ["cpu"])


# ── _discover_config_file unit tests ─────────────────────────────────────────

class TestDiscoverConfigFile(unittest.TestCase):

    def test_explicit_path_returned_as_is(self):
        result = _discover_config_file("/explicit/path.yaml")
        self.assertEqual(result, "/explicit/path.yaml")

    def test_engine_config_file_env_var_used(self):
        env = _clean_env("ENGINE_CONFIG_FILE", "APP_DIR")
        env["ENGINE_CONFIG_FILE"] = "/env/path/config.yaml"
        with patch.dict(os.environ, env, clear=True):
            result = _discover_config_file(None)
        self.assertEqual(result, "/env/path/config.yaml")

    def test_app_dir_etc_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            etc = Path(tmpdir) / "etc"
            etc.mkdir()
            cfg = etc / "nemotron-engine.yaml"
            cfg.write_text("", encoding="utf-8")

            env = _clean_env("ENGINE_CONFIG_FILE", "APP_DIR")
            env["APP_DIR"] = tmpdir
            with patch.dict(os.environ, env, clear=True):
                result = _discover_config_file(None)
        self.assertEqual(result, str(cfg))

    def test_cwd_etc_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            etc = Path(tmpdir) / "etc"
            etc.mkdir()
            cfg = etc / "nemotron-engine.yaml"
            cfg.write_text("", encoding="utf-8")

            env = _clean_env("ENGINE_CONFIG_FILE", "APP_DIR")
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, env, clear=True):
                    result = _discover_config_file(None)
            finally:
                os.chdir(old_cwd)
        self.assertEqual(result, str(cfg))

    def test_returns_none_when_nothing_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = _clean_env("ENGINE_CONFIG_FILE", "APP_DIR")
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, env, clear=True):
                    result = _discover_config_file(None)
            finally:
                os.chdir(old_cwd)
        self.assertIsNone(result)

    def test_explicit_takes_priority_over_env_var(self):
        env = _clean_env("ENGINE_CONFIG_FILE", "APP_DIR")
        env["ENGINE_CONFIG_FILE"] = "/from_env.yaml"
        with patch.dict(os.environ, env, clear=True):
            result = _discover_config_file("/from_explicit.yaml")
        self.assertEqual(result, "/from_explicit.yaml")


# ── _load_yaml_file unit tests ────────────────────────────────────────────────

class TestLoadYamlFile(unittest.TestCase):

    def _tmp_yaml(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_loads_nested_mapping(self):
        path = self._tmp_yaml("asr:\n  workers: 4\n  language: vi-VN\n")
        data = _load_yaml_file(path)
        self.assertEqual(data["asr"]["workers"], 4)
        self.assertEqual(data["asr"]["language"], "vi-VN")

    def test_empty_file_returns_empty_dict(self):
        path = self._tmp_yaml("")
        self.assertEqual(_load_yaml_file(path), {})

    def test_top_level_list_raises(self):
        path = self._tmp_yaml("- item1\n- item2\n")
        with self.assertRaises(ValueError, msg="Expected a YAML mapping"):
            _load_yaml_file(path)

    def test_null_document_returns_empty_dict(self):
        path = self._tmp_yaml("null\n")
        self.assertEqual(_load_yaml_file(path), {})

    def test_preserves_types(self):
        path = self._tmp_yaml(
            "asr:\n"
            "  workers: 4\n"
            "  enabled: true\n"
            "  fraction: 0.9\n"
            "  devices:\n"
            "    - cuda:0\n"
            "    - cuda:1\n"
        )
        data = _load_yaml_file(path)
        self.assertIsInstance(data["asr"]["workers"], int)
        self.assertIsInstance(data["asr"]["enabled"], bool)
        self.assertIsInstance(data["asr"]["fraction"], float)
        self.assertIsInstance(data["asr"]["devices"], list)


# ── Settings integration tests ────────────────────────────────────────────────

class TestSettingsFromFileOrEnv(unittest.TestCase):
    """Integration tests covering full Settings construction from YAML."""

    def _write(self, path: str | Path, data: dict) -> None:
        _write_yaml(path, data)

    def _env(self, extra: dict | None = None, **override) -> dict:
        """Clean env with mandatory MODEL_PATH plus any extras."""
        env = _clean_env(*_DROP)
        env["MODEL_PATH"] = "models/asr.nemo"
        if extra:
            env.update(extra)
        env.update(override)
        return env

    # ── config_source ─────────────────────────────────────────────────────────

    def test_config_source_env_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, self._env(), clear=True):
                    s = Settings.from_file_or_env()
            finally:
                os.chdir(old)
        self.assertEqual(s.config_source, "env")

    def test_config_source_file_path_when_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {"asr": {"model_path": "models/asr.nemo"}})
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.config_source, f"file:{cfg_path}")

    # ── priority: YAML > env ──────────────────────────────────────────────────

    def test_yaml_model_path_wins_over_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {"asr": {"model_path": "models/yaml.nemo"}})
            env = self._env(
                ENGINE_CONFIG_FILE=str(cfg_path),
                MODEL_PATH="models/env.nemo",
            )
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.model_path, "models/yaml.nemo")

    def test_env_used_for_key_absent_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            # model_path absent from YAML → must fall back to env
            self._write(cfg_path, {"grpc": {"server_address": "0.0.0.0:9999"}})
            env = self._env(
                ENGINE_CONFIG_FILE=str(cfg_path),
                MODEL_PATH="models/from_env.nemo",
            )
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.model_path, "models/from_env.nemo")
        self.assertEqual(s.grpc_address, "0.0.0.0:9999")

    def test_yaml_and_env_mixed_for_different_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/yaml.nemo", "workers": 8},
            })
            env = self._env(
                ENGINE_CONFIG_FILE=str(cfg_path),
                ASR_WORKERS="2",        # env says 2, YAML says 8 → YAML wins
                GRPC_SERVER_ADDRESS="0.0.0.0:7777",
            )
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.asr_workers, 8)
        self.assertEqual(s.grpc_address, "0.0.0.0:7777")

    # ── devices ───────────────────────────────────────────────────────────────

    def test_yaml_devices_as_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {
                    "model_path": "models/asr.nemo",
                    "device": "cuda:0",
                    "devices": ["cuda:0", "cuda:1"],
                }
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.asr_devices, ["cuda:0", "cuda:1"])

    def test_yaml_devices_as_comma_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {
                    "model_path": "models/asr.nemo",
                    "device": "cuda:0",
                    "devices": "cuda:0,cuda:1",
                }
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.asr_devices, ["cuda:0", "cuda:1"])

    # ── TTS backends ──────────────────────────────────────────────────────────

    def test_magpie_backend_fields_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/asr.nemo"},
                "tts": {
                    "backend": "magpie",
                    "magpie": {
                        "model_path": "models/magpie.nemo",
                        "codec_model_path": "models/codec-offline.nemo",
                        "speaker_index": 2,
                        "cfg_scale": 3.0,
                    }
                }
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.tts_backend, "magpie")
        self.assertEqual(s.tts_magpie_model_path, "models/magpie.nemo")
        self.assertEqual(s.tts_magpie_codec_model_path, "models/codec-offline.nemo")
        self.assertEqual(s.tts_speaker_index, 2)
        self.assertAlmostEqual(s.tts_cfg_scale, 3.0)

    def test_tts_disabled_skips_path_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/asr.nemo"},
                "tts": {"enabled": False},
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertFalse(s.tts_enabled)

    def test_translation_disabled_skips_path_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/asr.nemo"},
                "translation": {"enabled": False},
                "tts": {"enabled": False},
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertFalse(s.translation_enabled)

    # ── error cases ───────────────────────────────────────────────────────────

    def test_explicit_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            Settings.from_file_or_env("/nonexistent/path/config.yaml")

    def test_invalid_backend_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/asr.nemo"},
                "tts": {"backend": "unknown_backend"},
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_file_or_env()

    def test_missing_model_path_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            # No model_path in YAML, no MODEL_PATH in env
            self._write(cfg_path, {"grpc": {"server_address": "0.0.0.0:50052"}})
            env = _clean_env(*_DROP)  # no MODEL_PATH
            env["ENGINE_CONFIG_FILE"] = str(cfg_path)
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_file_or_env()

    def test_invalid_yaml_int_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {
                "asr": {
                    "model_path": "models/asr.nemo",
                    "workers": True,  # bool where int expected
                }
            })
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path))
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError):
                    Settings.from_file_or_env()

    # ── backward compatibility ────────────────────────────────────────────────

    def test_from_env_still_works(self):
        env = self._env(GRPC_SERVER_ADDRESS="0.0.0.0:9876")
        with patch.dict(os.environ, env, clear=True):
            s = Settings.from_env()
        self.assertEqual(s.config_source, "env")
        self.assertEqual(s.model_path, "models/asr.nemo")
        self.assertEqual(s.grpc_address, "0.0.0.0:9876")

    def test_from_env_and_from_file_or_env_produce_same_result_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._env()
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, env, clear=True):
                    s_env = Settings.from_env()
                    s_file = Settings.from_file_or_env()
            finally:
                os.chdir(old)
        # All fields except config_source should be identical
        for field in Settings.__dataclass_fields__:
            if field == "config_source":
                continue
            self.assertEqual(
                getattr(s_env, field),
                getattr(s_file, field),
                msg=f"Field {field!r} differs between from_env and from_file_or_env",
            )

    # ── APP_DIR discovery ─────────────────────────────────────────────────────

    def test_app_dir_etc_auto_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            etc = Path(tmpdir) / "etc"
            etc.mkdir()
            cfg_path = etc / "nemotron-engine.yaml"
            self._write(cfg_path, {
                "asr": {"model_path": "models/asr.nemo"},
                "metrics": {"port": 9999},
            })
            env = _clean_env(*_DROP)
            env["APP_DIR"] = tmpdir
            with patch.dict(os.environ, env, clear=True):
                s = Settings.from_file_or_env()
        self.assertEqual(s.metrics_port, 9999)
        self.assertIn("etc", s.config_source)


if __name__ == "__main__":
    unittest.main()
