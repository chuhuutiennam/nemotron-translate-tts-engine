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
    "MODEL_PATH", "ASR_DEVICE", "ASR_DEVICES", "ASR_USE_LANGUAGE_SOURCE",
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

    # ── tts.max_audio_chunk_bytes ─────────────────────────────────────────────

    def _chunk_bytes(self, **env_extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, self._env(env_extra), clear=True):
                    return Settings.from_file_or_env().tts_max_audio_chunk_bytes
            finally:
                os.chdir(old)

    def test_max_audio_chunk_bytes_default_and_env(self):
        self.assertEqual(self._chunk_bytes(), 0)
        self.assertEqual(self._chunk_bytes(TTS_MAX_AUDIO_CHUNK_BYTES="16000"), 16_000)
        self.assertEqual(self._chunk_bytes(TTS_MAX_AUDIO_CHUNK_BYTES="32000"), 32_000)

    def test_max_audio_chunk_bytes_rejects_unaligned_or_negative(self):
        for bad in ("100", "-160"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "max_audio_chunk_bytes"):
                    self._chunk_bytes(TTS_MAX_AUDIO_CHUNK_BYTES=bad)

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

    def _trace_level(self, logging_cfg: dict | None, **env_extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            data = {"asr": {"model_path": "models/asr.nemo"}}
            if logging_cfg is not None:
                data["logging"] = logging_cfg
            self._write(cfg_path, data)
            env = self._env(ENGINE_CONFIG_FILE=str(cfg_path), **env_extra)
            with patch.dict(os.environ, env, clear=True):
                return Settings.from_file_or_env().pipeline_trace_level

    def test_pipeline_trace_level_defaults_to_debug(self):
        self.assertEqual(self._trace_level(None), "DEBUG")

    def test_pipeline_trace_level_from_yaml_is_uppercased(self):
        self.assertEqual(self._trace_level({"pipeline_trace_level": "info"}), "INFO")

    def test_pipeline_trace_level_env_fallback_and_yaml_precedence(self):
        self.assertEqual(self._trace_level(None, LOG_PIPELINE_TRACE_LEVEL="warning"), "WARNING")
        self.assertEqual(
            self._trace_level({"pipeline_trace_level": "INFO"}, LOG_PIPELINE_TRACE_LEVEL="WARNING"),
            "INFO",
        )

    def test_pipeline_trace_level_invalid_raises(self):
        with self.assertRaises(ValueError):
            self._trace_level({"pipeline_trace_level": "LOUD"})

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

    # ── segmentation ──────────────────────────────────────────────────────────

    def _segmentation(self, yaml_section=None, **env_extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._env(env_extra)
            if yaml_section is not None:
                cfg_path = Path(tmpdir) / "engine.yaml"
                self._write(cfg_path, {
                    "asr": {"model_path": "models/asr.nemo"},
                    "segmentation": yaml_section,
                })
                env["ENGINE_CONFIG_FILE"] = str(cfg_path)
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, env, clear=True):
                    return Settings.from_file_or_env()
            finally:
                os.chdir(old)

    def test_segmentation_defaults_are_off(self):
        s = self._segmentation()
        self.assertEqual(s.segmentation_mode, "off")
        self.assertEqual(s.segmentation_stable_n, 2)
        self.assertEqual(s.segmentation_stable_ms, 0.0)
        self.assertEqual(s.segmentation_punct_stable_n, 2)
        self.assertEqual(s.segmentation_punct_stable_ms, 0.0)
        self.assertEqual(s.segmentation_min_tokens, 4)
        self.assertEqual(s.segmentation_max_segment_ms, 8000.0)
        self.assertEqual(s.segmentation_comma_after_ms, 0.0)

    def test_segmentation_punct_stable_inherits_and_overrides(self):
        s = self._segmentation({"stable_n": 3, "stable_ms": 800})
        self.assertEqual(s.segmentation_punct_stable_n, 3)
        self.assertEqual(s.segmentation_punct_stable_ms, 800.0)
        s = self._segmentation({"stable_n": 3, "punct_stable_n": 1, "punct_stable_ms": 300})
        self.assertEqual(s.segmentation_stable_n, 3)
        self.assertEqual(s.segmentation_punct_stable_n, 1)
        self.assertEqual(s.segmentation_punct_stable_ms, 300.0)
        s = self._segmentation(SEGMENTATION_PUNCT_STABLE_N="1", SEGMENTATION_STABLE_N="3")
        self.assertEqual(s.segmentation_punct_stable_n, 1)

    def test_segmentation_punct_stable_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self._segmentation({"punct_stable_n": 0})
        with self.assertRaises(ValueError):
            self._segmentation({"punct_stable_ms": -1})

    def test_segmentation_stt_defaults_inherit_and_override(self):
        s = self._segmentation({"punct_stable_n": 1, "punct_stable_ms": 300})
        self.assertFalse(s.segmentation_speech_to_text)
        self.assertEqual(s.segmentation_stt_punct_stable_n, 1)
        self.assertEqual(s.segmentation_stt_punct_stable_ms, 300.0)
        s = self._segmentation(
            {"speech_to_text": True, "punct_stable_n": 1, "stt_punct_stable_n": 3, "stt_punct_stable_ms": 500}
        )
        self.assertTrue(s.segmentation_speech_to_text)
        self.assertEqual(s.segmentation_punct_stable_n, 1)
        self.assertEqual(s.segmentation_stt_punct_stable_n, 3)
        self.assertEqual(s.segmentation_stt_punct_stable_ms, 500.0)
        s = self._segmentation(SEGMENTATION_SPEECH_TO_TEXT="true", SEGMENTATION_STT_PUNCT_STABLE_N="4")
        self.assertTrue(s.segmentation_speech_to_text)
        self.assertEqual(s.segmentation_stt_punct_stable_n, 4)

    def test_segmentation_stt_punct_stable_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self._segmentation({"stt_punct_stable_n": 0})
        with self.assertRaises(ValueError):
            self._segmentation({"stt_punct_stable_ms": -1})

    def test_segmentation_from_yaml_and_env(self):
        s = self._segmentation({"mode": "shadow", "stable_n": 3, "stable_ms": 800})
        self.assertEqual(s.segmentation_mode, "shadow")
        self.assertEqual(s.segmentation_stable_n, 3)
        self.assertEqual(s.segmentation_stable_ms, 800.0)
        s = self._segmentation(SEGMENTATION_MODE="SHADOW", SEGMENTATION_MIN_TOKENS="6")
        self.assertEqual(s.segmentation_mode, "shadow")
        self.assertEqual(s.segmentation_min_tokens, 6)

    def _segmentation_raw_yaml(self, mode_line: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            cfg_path.write_text(
                "asr:\n  model_path: models/asr.nemo\n"
                f"segmentation:\n  {mode_line}\n",
                encoding="utf-8",
            )
            env = self._env({"ENGINE_CONFIG_FILE": str(cfg_path)})
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, env, clear=True):
                    return Settings.from_file_or_env()
            finally:
                os.chdir(old)

    def test_segmentation_mode_accepts_unquoted_yaml_on_off(self):
        for line, expected in (
            ("mode: on", "on"),
            ("mode: off", "off"),
            ("mode: shadow", "shadow"),
            ('mode: "on"', "on"),
            ('mode: "off"', "off"),
            ("mode: ON", "on"),
        ):
            with self.subTest(line=line):
                self.assertEqual(self._segmentation_raw_yaml(line).segmentation_mode, expected)

    def test_segmentation_mode_rejects_unknown_yaml_value(self):
        with self.assertRaises(ValueError):
            self._segmentation_raw_yaml("mode: bogus")

    def test_shipped_engine_yaml_loads_with_segmentation_on(self):
        shipped = Path(__file__).resolve().parents[1] / "etc" / "nemotron-engine.yaml"
        with patch.dict(os.environ, self._env({"ENGINE_CONFIG_FILE": str(shipped)}), clear=True):
            settings = Settings.from_file_or_env()
        self.assertEqual(settings.segmentation_mode, "on")
        self.assertEqual(settings.max_postprocess_pending_per_rpc, 16)

    def test_segmentation_on_raises_postprocess_pending_default(self):
        self.assertEqual(self._segmentation().max_postprocess_pending_per_rpc, 8)
        self.assertEqual(
            self._segmentation(SEGMENTATION_MODE="shadow").max_postprocess_pending_per_rpc, 8
        )
        s = self._segmentation(SEGMENTATION_MODE="on")
        self.assertEqual(s.segmentation_mode, "on")
        self.assertEqual(s.max_postprocess_pending_per_rpc, 16)
        s = self._segmentation(SEGMENTATION_MODE="on", MAX_POSTPROCESS_PENDING_PER_RPC="5")
        self.assertEqual(s.max_postprocess_pending_per_rpc, 5)

    def test_segmentation_rejects_invalid_values(self):
        for env in (
            {"SEGMENTATION_MODE": "bogus"},
            {"SEGMENTATION_STABLE_N": "0"},
            {"SEGMENTATION_STABLE_MS": "-1"},
            {"SEGMENTATION_MIN_TOKENS": "0"},
            {"SEGMENTATION_MAX_SEGMENT_MS": "-5"},
            {"SEGMENTATION_COMMA_AFTER_MS": "-5"},
        ):
            with self.subTest(env=env):
                with self.assertRaisesRegex(ValueError, "segmentation"):
                    self._segmentation(**env)

    # ── vad.neg_threshold ─────────────────────────────────────────────────────

    def _vad_thresholds(self, yaml_vad: dict | None = None, **env_extra) -> tuple[float, float]:
        env = self._env(env_extra)
        for key in ("VAD_THRESHOLD", "VAD_NEG_THRESHOLD"):
            if key not in env_extra:
                env.pop(key, None)
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                if yaml_vad is not None:
                    cfg_path = Path(tmpdir) / "engine.yaml"
                    self._write(cfg_path, {"vad": yaml_vad})
                    env["ENGINE_CONFIG_FILE"] = str(cfg_path)
                with patch.dict(os.environ, env, clear=True):
                    s = Settings.from_file_or_env()
                    return s.vad_threshold, s.vad_neg_threshold
            finally:
                os.chdir(old)

    def test_neg_threshold_defaults_to_silero_upstream_formula(self):
        threshold, neg = self._vad_thresholds()
        self.assertEqual(threshold, 0.5)
        self.assertAlmostEqual(neg, 0.35)
        self.assertAlmostEqual(self._vad_thresholds(VAD_THRESHOLD="0.6")[1], 0.45)
        self.assertAlmostEqual(self._vad_thresholds(VAD_THRESHOLD="0.1")[1], 0.01)

    def test_neg_threshold_from_env_and_yaml(self):
        self.assertEqual(self._vad_thresholds(VAD_NEG_THRESHOLD="0.4")[1], 0.4)
        self.assertEqual(self._vad_thresholds({"neg_threshold": 0.3}, VAD_NEG_THRESHOLD="0.4")[1], 0.3)

    def test_neg_threshold_yaml_null_uses_default(self):
        _, neg = self._vad_thresholds({"threshold": 0.6, "neg_threshold": None}, VAD_NEG_THRESHOLD="0.4")
        self.assertAlmostEqual(neg, 0.45)

    def test_neg_threshold_equal_to_threshold_disables_dead_band(self):
        self.assertEqual(self._vad_thresholds(VAD_NEG_THRESHOLD="0.5")[1], 0.5)

    def test_neg_threshold_rejects_out_of_range(self):
        for bad in ("0.6", "0", "-0.1"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "vad.neg_threshold"):
                    self._vad_thresholds(VAD_NEG_THRESHOLD=bad)

    # ── asr.use_language_source ───────────────────────────────────────────────

    def _use_language_source(self, yaml_asr: dict | None = None, **env_extra) -> bool:
        env = self._env(env_extra)
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                if yaml_asr is not None:
                    cfg_path = Path(tmpdir) / "engine.yaml"
                    self._write(cfg_path, {"asr": {"model_path": "models/asr.nemo", **yaml_asr}})
                    env["ENGINE_CONFIG_FILE"] = str(cfg_path)
                with patch.dict(os.environ, env, clear=True):
                    return Settings.from_file_or_env().use_language_source
            finally:
                os.chdir(old)

    def test_use_language_source_defaults_to_off(self):
        self.assertFalse(self._use_language_source())

    def test_use_language_source_from_env_and_yaml(self):
        self.assertTrue(self._use_language_source(ASR_USE_LANGUAGE_SOURCE="true"))
        self.assertFalse(self._use_language_source(ASR_USE_LANGUAGE_SOURCE="false"))
        self.assertTrue(self._use_language_source({"use_language_source": True}))
        self.assertFalse(
            self._use_language_source({"use_language_source": False}, ASR_USE_LANGUAGE_SOURCE="true")
        )

    def test_use_language_source_yaml_null_uses_default(self):
        self.assertFalse(
            self._use_language_source({"use_language_source": None}, ASR_USE_LANGUAGE_SOURCE="true")
        )

    def test_shipped_engine_yaml_enables_language_source(self):
        shipped = Path(__file__).resolve().parents[1] / "etc" / "nemotron-engine.yaml"
        data = yaml.safe_load(shipped.read_text(encoding="utf-8"))
        self.assertIs(data["asr"]["use_language_source"], True)

    # ── audio_preprocessing ───────────────────────────────────────────────────

    def _preproc(self, yaml_section=None, **env_extra):
        env = self._env(env_extra)
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                if yaml_section is not None:
                    cfg_path = Path(tmpdir) / "engine.yaml"
                    self._write(cfg_path, {
                        "asr": {"model_path": "models/asr.nemo"},
                        "audio_preprocessing": yaml_section,
                    })
                    env["ENGINE_CONFIG_FILE"] = str(cfg_path)
                with patch.dict(os.environ, env, clear=True):
                    return Settings.from_file_or_env().audio_preprocessing
            finally:
                os.chdir(old)

    def test_audio_preprocessing_defaults_off_with_denoiser_window(self):
        p = self._preproc()
        self.assertFalse(p.enabled)
        self.assertEqual((p.model, p.mode), ("resemble", "denoiser"))
        self.assertEqual(
            (p.left_context_ms, p.hop_ms, p.right_context_ms, p.compute_budget_ms), (1000, 500, 500, 500)
        )
        self.assertEqual(p.delay_ms, 1500)
        self.assertEqual(p.vad_input, "enhanced")
        self.assertEqual(p.sample_percent, 100)

    def test_audio_preprocessing_enhancer_mode_changes_window_defaults(self):
        p = self._preproc({"enabled": True, "resemble": {"mode": "enhancer"}})
        self.assertTrue(p.enabled)
        self.assertEqual((p.model, p.mode), ("resemble", "enhancer"))
        self.assertEqual(
            (p.left_context_ms, p.hop_ms, p.right_context_ms, p.compute_budget_ms), (1000, 2000, 1000, 1500)
        )
        self.assertEqual(p.delay_ms, 4500)

    def test_audio_preprocessing_audx_model_is_stateful_with_its_own_window(self):
        p = self._preproc({"enabled": True, "model": "audx", "resemble": {"assets_dir": ""}})
        self.assertEqual((p.model, p.mode), ("audx", "audx"))
        self.assertEqual(
            (p.left_context_ms, p.hop_ms, p.right_context_ms, p.compute_budget_ms, p.crossfade_ms),
            (0, 80, 64, 100, 10),
        )
        self.assertEqual(p.delay_ms, 244)
        self.assertEqual(p.audx_checkpoint, "models/audx/audx_weights_safe.pth")
        self.assertEqual((p.audx_agc_target_dbfs, p.audx_agc_max_gain_db), (-28.0, 30.0))
        p = self._preproc({
            "enabled": True,
            "model": "audx",
            "audx": {
                "checkpoint": "m/a.pth", "agc_target_dbfs": -30, "agc_max_gain_db": 0,
                "window": {"hop_ms": 160, "compute_budget_ms": 60, "crossfade_ms": 20},
            },
        })
        self.assertEqual((p.audx_checkpoint, p.audx_agc_target_dbfs, p.audx_agc_max_gain_db), ("m/a.pth", -30.0, 0.0))
        self.assertEqual((p.hop_ms, p.crossfade_ms, p.delay_ms), (160, 20, 160 + 64 + 60))
        p = self._preproc(
            AUDIO_PREPROCESSING_MODEL="AudX",
            AUDIO_PREPROCESSING_AUDX_CHECKPOINT="/opt/audx.pth",
            AUDIO_PREPROCESSING_AUDX_AGC_TARGET_DBFS="-26",
            AUDIO_PREPROCESSING_AUDX_AGC_MAX_GAIN_DB="20",
            AUDIO_PREPROCESSING_AUDX_HOP_MS="96",
            AUDIO_PREPROCESSING_AUDX_CROSSFADE_MS="8",
            AUDIO_PREPROCESSING_AUDX_COMPUTE_BUDGET_MS="50",
            # the resemble window envs do not touch audx
            AUDIO_PREPROCESSING_HOP_MS="500",
            AUDIO_PREPROCESSING_COMPUTE_BUDGET_MS="200",
        )
        self.assertEqual(
            (p.mode, p.audx_checkpoint, p.audx_agc_target_dbfs, p.audx_agc_max_gain_db),
            ("audx", "/opt/audx.pth", -26.0, 20.0),
        )
        self.assertEqual((p.hop_ms, p.crossfade_ms, p.compute_budget_ms, p.delay_ms), (96, 8, 50, 96 + 64 + 50))

    def test_audio_preprocessing_switching_model_needs_no_other_change(self):
        section = {
            "enabled": True,
            "resemble": {"mode": "denoiser", "window": {"crossfade_ms": 40, "compute_budget_ms": 200}},
            "audx": {"window": {"hop_ms": 80, "crossfade_ms": 10, "compute_budget_ms": 100}},
        }
        resemble = self._preproc({**section, "model": "resemble"})
        audx = self._preproc({**section, "model": "audx"})
        self.assertEqual((resemble.mode, resemble.crossfade_ms, resemble.delay_ms), ("denoiser", 40, 1200))
        self.assertEqual((audx.mode, audx.crossfade_ms, audx.delay_ms), ("audx", 10, 244))

    def test_audio_preprocessing_audx_rejects_settings_it_cannot_use(self):
        for env, message in (
            ({"AUDIO_PREPROCESSING_AUDX_HOP_MS": "100"}, "multiple of 16"),
            ({"AUDIO_PREPROCESSING_AUDX_HOP_MS": "8"}, "audx.window.hop_ms must be >= 16"),
            ({"AUDIO_PREPROCESSING_AUDX_CROSSFADE_MS": "60", "AUDIO_PREPROCESSING_AUDX_COMPUTE_BUDGET_MS": "50"},
             "< compute_budget_ms"),
            ({"AUDIO_PREPROCESSING_AUDX_AGC_TARGET_DBFS": "-3"}, "agc_target_dbfs"),
            ({"AUDIO_PREPROCESSING_AUDX_AGC_MAX_GAIN_DB": "-1"}, "agc_max_gain_db"),
            ({"AUDIO_PREPROCESSING_ENABLED": "true", "AUDIO_PREPROCESSING_AUDX_CHECKPOINT": ""}, "audx.checkpoint"),
        ):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, message):
                self._preproc(AUDIO_PREPROCESSING_MODEL="audx", **env)
        with self.assertRaisesRegex(ValueError, "model must be 'resemble' or 'audx'"):
            self._preproc(AUDIO_PREPROCESSING_MODEL="denoiser")
        # AudX is a model, not a resemble mode
        with self.assertRaisesRegex(ValueError, "resemble.mode must be 'denoiser' or 'enhancer'.*model: audx"):
            self._preproc(AUDIO_PREPROCESSING_MODE="audx")
        s = self._preproc_settings({"model": "audx", "audx": {"window": {"left_context_ms": 100}}})
        self.assertEqual(s.audio_preprocessing.left_context_ms, 0)
        self.assertEqual(len(s.config_warnings), 1)
        self.assertIn("audx.window.left_context_ms is not used", s.config_warnings[0])

    def test_audio_preprocessing_old_mode_and_window_are_not_read_any_more(self):
        s = self._preproc_settings({"mode": "audx", "window": {"hop_ms": 80, "compute_budget_ms": 100}})
        p = s.audio_preprocessing
        # neither key is read: model stays the default resemble with the denoiser window
        self.assertEqual((p.model, p.mode, p.hop_ms, p.compute_budget_ms), ("resemble", "denoiser", 500, 500))
        self.assertEqual(
            list(s.config_warnings),
            [
                "audio_preprocessing.mode is no longer read; use audio_preprocessing.model (resemble | audx) "
                "and resemble.mode",
                "audio_preprocessing.window is no longer read; use audio_preprocessing.resemble.window / audx.window",
            ],
        )

    def _preproc_settings(self, yaml_section):
        """Whole Settings (to read config_warnings) for a YAML audio_preprocessing section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "engine.yaml"
            self._write(cfg_path, {"asr": {"model_path": "models/asr.nemo"}, "audio_preprocessing": yaml_section})
            with patch.dict(os.environ, self._env({"ENGINE_CONFIG_FILE": str(cfg_path)}), clear=True):
                return Settings.from_file_or_env()

    def test_audio_preprocessing_resemble_keys_live_under_resemble(self):
        s = self._preproc_settings({
            "zero_run_min_samples": 32,
            "resemble": {
                "mode": "enhancer",
                "assets_dir": "/opt/re",
                "enhancer": {"nfe": 8, "lambd": 0.5, "tau": 0.25, "align_regions": False, "align_max_ms": 20},
                "dither": {"sigma_lsb": 1.5, "seed": 7},
            },
        })
        p = s.audio_preprocessing
        self.assertEqual(
            (p.assets_dir, p.enhancer_nfe, p.enhancer_lambd, p.enhancer_tau, p.enhancer_align_regions,
             p.enhancer_align_max_ms, p.dither_sigma_lsb, p.dither_seed, p.dither_min_zero_run),
            ("/opt/re", 8, 0.5, 0.25, False, 20, 1.5, 7, 32),
        )
        self.assertEqual(s.config_warnings, ())

    def test_audio_preprocessing_old_paths_still_work_with_a_warning(self):
        old = {
            "resemble": {"mode": "enhancer"},
            "assets_dir": "/opt/old",
            "enhancer": {"nfe": 8, "lambd": 0.5, "tau": 0.25, "align_regions": False, "align_max_ms": 20},
            "dither": {"sigma_lsb": 1.5, "min_zero_run_samples": 32, "seed": 7},
        }
        s = self._preproc_settings(old)
        p = s.audio_preprocessing
        self.assertEqual(
            (p.assets_dir, p.enhancer_nfe, p.enhancer_lambd, p.enhancer_tau, p.enhancer_align_regions,
             p.enhancer_align_max_ms, p.dither_sigma_lsb, p.dither_seed, p.dither_min_zero_run),
            ("/opt/old", 8, 0.5, 0.25, False, 20, 1.5, 7, 32),
        )
        self.assertEqual(len(s.config_warnings), 9)
        self.assertTrue(all("deprecated" in w for w in s.config_warnings))
        self.assertIn(
            "audio_preprocessing.dither.min_zero_run_samples is deprecated; "
            "move it to audio_preprocessing.zero_run_min_samples",
            s.config_warnings,
        )
        # both present: the new path wins, the old one is reported as ignored
        s = self._preproc_settings({"assets_dir": "/opt/old", "resemble": {"assets_dir": "/opt/new"}})
        self.assertEqual(s.audio_preprocessing.assets_dir, "/opt/new")
        self.assertEqual(len(s.config_warnings), 1)
        self.assertIn("is ignored", s.config_warnings[0])
        # env names did not change
        p = self._preproc(AUDIO_PREPROCESSING_ASSETS_DIR="/env/re", AUDIO_PREPROCESSING_DITHER_MIN_ZERO_RUN="16")
        self.assertEqual((p.assets_dir, p.dither_min_zero_run), ("/env/re", 16))

    def test_audio_preprocessing_reads_only_devices(self):
        s = self._preproc_settings({"devices": ["cuda:1", "cuda:2"]})
        self.assertEqual(s.audio_preprocessing.devices, ("cuda:1", "cuda:2"))
        self.assertEqual(s.config_warnings, ())
        # the removed single `device` key is not read (default cpu) and is reported
        s = self._preproc_settings({"device": "cuda:3"})
        self.assertEqual(s.audio_preprocessing.devices, ("cpu",))
        self.assertEqual(len(s.config_warnings), 1)
        self.assertIn("audio_preprocessing.device (AUDIO_PREPROCESSING_DEVICE) is no longer read", s.config_warnings[0])
        s = self._preproc_settings({"device": "cuda:3", "devices": ["cuda:1"]})
        self.assertEqual(s.audio_preprocessing.devices, ("cuda:1",))
        self.assertEqual(len(s.config_warnings), 1)
        self.assertEqual(self._preproc(AUDIO_PREPROCESSING_DEVICE="cuda:2").devices, ("cpu",))
        self.assertEqual(
            self._preproc(AUDIO_PREPROCESSING_DEVICES="cuda:1, cuda:2", AUDIO_PREPROCESSING_DEVICE="cuda:3").devices,
            ("cuda:1", "cuda:2"),
        )

    def test_audio_preprocessing_zero_means_mode_default_nonzero_overrides(self):
        p = self._preproc({"resemble": {"window": {"hop_ms": 0, "right_context_ms": 250, "compute_budget_ms": 300}}})
        self.assertEqual((p.hop_ms, p.right_context_ms, p.compute_budget_ms), (500, 250, 300))
        self.assertEqual(p.delay_ms, 1050)

    def test_audio_preprocessing_from_env_and_yaml_precedence(self):
        p = self._preproc(
            AUDIO_PREPROCESSING_ENABLED="true",
            AUDIO_PREPROCESSING_MODE="Enhancer",
            AUDIO_PREPROCESSING_DEVICES="cuda:1,cuda:2",
            AUDIO_PREPROCESSING_DITHER_SIGMA_LSB="10",
        )
        self.assertTrue(p.enabled)
        self.assertEqual(p.mode, "enhancer")
        self.assertEqual(p.devices, ("cuda:1", "cuda:2"))
        self.assertEqual(p.dither_sigma_lsb, 10.0)
        p = self._preproc({"enabled": False, "resemble": {"dither": {"sigma_lsb": 3}}}, AUDIO_PREPROCESSING_ENABLED="true",
                          AUDIO_PREPROCESSING_DITHER_SIGMA_LSB="10")
        self.assertFalse(p.enabled)
        self.assertEqual(p.dither_sigma_lsb, 3.0)

    def test_audio_preprocessing_rejects_invalid_values(self):
        for env in (
            {"AUDIO_PREPROCESSING_MODE": "bogus"},
            {"AUDIO_PREPROCESSING_SAMPLE_PERCENT": "101"},
            {"AUDIO_PREPROCESSING_WORKERS": "0"},
            {"AUDIO_PREPROCESSING_QUEUE_MAX": "0"},
            {"AUDIO_PREPROCESSING_MAX_STREAMS": "-1"},
            {"AUDIO_PREPROCESSING_HOP_MS": "10"},
            {"AUDIO_PREPROCESSING_CROSSFADE_MS": "600"},
            {"AUDIO_PREPROCESSING_ENHANCER_NFE": "1"},
            {"AUDIO_PREPROCESSING_ENHANCER_LAMBD": "1.5"},
            {"AUDIO_PREPROCESSING_DITHER_SIGMA_LSB": "-1"},
            {"AUDIO_PREPROCESSING_DITHER_MIN_ZERO_RUN": "0"},
            {"AUDIO_PREPROCESSING_VAD_INPUT": "both"},
            {"AUDIO_PREPROCESSING_FALLBACK_BAD_REGIONS": "0"},
            {"AUDIO_PREPROCESSING_FALLBACK_BYPASS_S": "-1"},
            {"AUDIO_PREPROCESSING_ENABLED": "true", "AUDIO_PREPROCESSING_ASSETS_DIR": ""},
        ):
            with self.subTest(env=env):
                with self.assertRaisesRegex(ValueError, "audio_preprocessing"):
                    self._preproc(**env)

    def test_audio_preprocessing_right_context_must_cover_crossfade_and_mask(self):
        with self.assertRaisesRegex(ValueError, "zero_run_min_samples"):
            self._preproc(
                AUDIO_PREPROCESSING_RIGHT_CONTEXT_MS="20", AUDIO_PREPROCESSING_CROSSFADE_MS="17"
            )
        p = self._preproc(
            AUDIO_PREPROCESSING_RIGHT_CONTEXT_MS="20", AUDIO_PREPROCESSING_CROSSFADE_MS="16"
        )
        self.assertEqual((p.right_context_ms, p.crossfade_ms), (20, 16))

    def test_audio_preprocessing_enhancer_right_context_must_cover_alignment_search(self):
        narrow = {
            "AUDIO_PREPROCESSING_MODE": "enhancer",
            "AUDIO_PREPROCESSING_RIGHT_CONTEXT_MS": "60",
            "AUDIO_PREPROCESSING_CROSSFADE_MS": "40",
        }
        with self.assertRaisesRegex(ValueError, "align_max_ms"):
            self._preproc(**narrow)
        self.assertEqual(
            self._preproc(**narrow, AUDIO_PREPROCESSING_ENHANCER_ALIGN_REGIONS="false").right_context_ms, 60
        )
        self.assertEqual(
            self._preproc(**{**narrow, "AUDIO_PREPROCESSING_MODE": "denoiser"}).right_context_ms, 60
        )

    def test_shipped_engine_yaml_runs_audx_on_every_stream(self):
        # 2026-10-30: the shipped yaml turns on mode audx for 100% of streams (trial; Vietnamese WER not measured)
        shipped = Path(__file__).resolve().parents[1] / "etc" / "nemotron-engine.yaml"
        with patch.dict(os.environ, self._env({"ENGINE_CONFIG_FILE": str(shipped)}), clear=True):
            settings = Settings.from_file_or_env()
        p = settings.audio_preprocessing
        self.assertEqual(settings.config_warnings, ())  # new layout: resemble.*, zero_run_min_samples
        self.assertEqual((p.assets_dir, p.dither_min_zero_run), ("models/resemble-enhance", 64))
        self.assertTrue(p.enabled)
        self.assertEqual((p.model, p.mode), ("audx", "audx"))
        self.assertEqual(p.sample_percent, 100)
        self.assertEqual(p.devices, ("cuda:1",))
        self.assertEqual(
            (p.left_context_ms, p.hop_ms, p.right_context_ms, p.crossfade_ms, p.compute_budget_ms),
            (0, 80, 64, 10, 100),
        )
        self.assertEqual(p.delay_ms, 244)
        self.assertEqual(p.audx_checkpoint, "models/audx/audx_weights_safe.pth")
        self.assertEqual((p.audx_agc_target_dbfs, p.audx_agc_max_gain_db), (-28.0, 30.0))

    def test_kaggle_overlay_keeps_audio_preprocessing_off(self):
        # run_tests_kaggle.sh deep-merges scripts/kaggle/kaggle-engine.yaml onto the shipped yaml (make_config);
        # the shipped yaml turns audx on, the WER baselines must stay raw.
        root = Path(__file__).resolve().parents[1]

        def merge(dst, src):
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    merge(dst[key], value)
                else:
                    dst[key] = value

        cfg = yaml.safe_load((root / "etc" / "nemotron-engine.yaml").read_text(encoding="utf-8"))
        merge(cfg, yaml.safe_load((root / "scripts" / "kaggle" / "kaggle-engine.yaml").read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kaggle.yaml"
            self._write(path, cfg)
            with patch.dict(os.environ, self._env({"ENGINE_CONFIG_FILE": str(path)}), clear=True):
                p = Settings.from_file_or_env().audio_preprocessing
        self.assertFalse(p.enabled)

    def test_k8s_configmaps_define_every_audio_preprocessing_env_and_stay_off(self):
        root = Path(__file__).resolve().parents[1]
        shipped_yaml = (root / "etc" / "nemotron-engine.yaml").read_text(encoding="utf-8")
        env_names = {
            line.split("#", 1)[1].split()[0]
            for line in shipped_yaml.splitlines()
            if "AUDIO_PREPROCESSING_" in line
        }
        self.assertGreaterEqual(len(env_names), 20)
        for name in ("k8s-configmap.yaml", "k8s-configmap-cpu.yaml", "k8s-configmap-gpu.yaml"):
            with self.subTest(configmap=name):
                data = yaml.safe_load((root / name).read_text(encoding="utf-8"))["data"]
                missing = sorted(env_names - set(data))
                self.assertEqual(missing, [], msg=f"{name} lacks {missing}")
                env = self._env({k: str(v) for k, v in data.items() if k.startswith("AUDIO_PREPROCESSING_")})

                def load(env):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        old = os.getcwd()
                        try:
                            os.chdir(tmpdir)
                            with patch.dict(os.environ, env, clear=True):
                                return Settings.from_env().audio_preprocessing
                        finally:
                            os.chdir(old)

                p = load(env)
                self.assertFalse(p.enabled)
                self.assertEqual((p.model, p.mode), ("resemble", "denoiser"))
                # GPU budget 200 ms; CPU keeps the mode default 500 ms (CPU cannot meet 200 ms).
                expected_delay = 1500 if name == "k8s-configmap-cpu.yaml" else 1200
                self.assertEqual(p.delay_ms, expected_delay)
                # switching only the model gives AudX its own window (no D = 344 ms from the denoiser values)
                audx = load({**env, "AUDIO_PREPROCESSING_MODEL": "audx"})
                self.assertEqual((audx.mode, audx.crossfade_ms, audx.delay_ms), ("audx", 10, 244))

    # ── grpc keepalive / ping policy ──────────────────────────────────────────

    _PING_POLICY = {
        "grpc_keepalive_time_ms": 60_000,
        "grpc_keepalive_timeout_ms": 20_000,
        "grpc_keepalive_permit_without_calls": True,
        "grpc_min_ping_interval_without_data_ms": 60_000,
        "grpc_max_ping_strikes": 2,
    }
    _PING_ENV = {
        "grpc_keepalive_time_ms": "GRPC_KEEPALIVE_TIME_MS",
        "grpc_keepalive_timeout_ms": "GRPC_KEEPALIVE_TIMEOUT_MS",
        "grpc_keepalive_permit_without_calls": "GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS",
        "grpc_min_ping_interval_without_data_ms": "GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS",
        "grpc_max_ping_strikes": "GRPC_MAX_PING_STRIKES",
    }

    def _ping_policy(self, yaml_grpc: dict | None = None, **env_extra) -> dict:
        env = self._env()
        for name in self._PING_ENV.values():
            env.pop(name, None)
        env.update(env_extra)
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                if yaml_grpc is not None:
                    cfg_path = Path(tmpdir) / "engine.yaml"
                    self._write(
                        cfg_path,
                        {"asr": {"model_path": "models/asr.nemo"}, "grpc": yaml_grpc},
                    )
                    env["ENGINE_CONFIG_FILE"] = str(cfg_path)
                with patch.dict(os.environ, env, clear=True):
                    s = Settings.from_file_or_env()
            finally:
                os.chdir(old)
        return {field: getattr(s, field) for field in self._PING_ENV}

    def test_ping_policy_defaults(self):
        self.assertEqual(self._ping_policy(), self._PING_POLICY)

    def test_ping_policy_from_env(self):
        got = self._ping_policy(
            GRPC_KEEPALIVE_TIME_MS="90000",
            GRPC_KEEPALIVE_TIMEOUT_MS="15000",
            GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS="false",
            GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS="10000",
            GRPC_MAX_PING_STRIKES="5",
        )
        self.assertEqual(
            got,
            {
                "grpc_keepalive_time_ms": 90_000,
                "grpc_keepalive_timeout_ms": 15_000,
                "grpc_keepalive_permit_without_calls": False,
                "grpc_min_ping_interval_without_data_ms": 10_000,
                "grpc_max_ping_strikes": 5,
            },
        )

    def test_ping_policy_yaml_wins_over_env_and_null_uses_default(self):
        got = self._ping_policy(
            {
                "keepalive_permit_without_calls": False,
                "min_ping_interval_without_data_ms": 30000,
                "max_ping_strikes": None,
            },
            GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS="true",
            GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS="1000",
            GRPC_MAX_PING_STRIKES="9",
        )
        self.assertFalse(got["grpc_keepalive_permit_without_calls"])
        self.assertEqual(got["grpc_min_ping_interval_without_data_ms"], 30_000)
        self.assertEqual(got["grpc_max_ping_strikes"], 2)

    def test_ping_policy_zero_disables_the_two_ping_limits(self):
        got = self._ping_policy(
            GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS="0", GRPC_MAX_PING_STRIKES="0"
        )
        self.assertEqual(got["grpc_min_ping_interval_without_data_ms"], 0)
        self.assertEqual(got["grpc_max_ping_strikes"], 0)

    def test_ping_policy_rejects_out_of_range(self):
        for env, message in (
            ({"GRPC_KEEPALIVE_TIME_MS": "0"}, "keepalive_time_ms"),
            ({"GRPC_KEEPALIVE_TIMEOUT_MS": "-1"}, "keepalive_timeout_ms"),
            ({"GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS": "-1"}, "min_ping_interval_without_data_ms"),
            ({"GRPC_MAX_PING_STRIKES": "-1"}, "max_ping_strikes"),
        ):
            with self.subTest(env=env):
                with self.assertRaisesRegex(ValueError, message):
                    self._ping_policy(**env)

    def test_shipped_yaml_and_configmaps_carry_the_same_ping_policy(self):
        root = Path(__file__).resolve().parents[1]
        shipped = root / "etc" / "nemotron-engine.yaml"
        self.assertEqual(
            self._ping_policy(
                yaml.safe_load(shipped.read_text(encoding="utf-8"))["grpc"]
            ),
            self._PING_POLICY,
        )
        for name in ("k8s-configmap.yaml", "k8s-configmap-cpu.yaml", "k8s-configmap-gpu.yaml"):
            with self.subTest(configmap=name):
                data = yaml.safe_load((root / name).read_text(encoding="utf-8"))["data"]
                self.assertEqual(
                    self._ping_policy(
                        **{env: str(data[env]) for env in self._PING_ENV.values()}
                    ),
                    self._PING_POLICY,
                )

    def test_dockerfile_and_run_script_carry_the_same_ping_policy(self):
        import re

        root = Path(__file__).resolve().parents[1]
        expected = {
            "GRPC_KEEPALIVE_TIME_MS": "60000",
            "GRPC_KEEPALIVE_TIMEOUT_MS": "20000",
            "GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS": "true",
            "GRPC_MIN_PING_INTERVAL_WITHOUT_DATA_MS": "60000",
            "GRPC_MAX_PING_STRIKES": "2",
        }
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
        script = (root / "scripts" / "run-gpu-server.sh").read_text(encoding="utf-8")
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertRegex(dockerfile, rf"(?m)^\s+{name}={value}\b")
                self.assertRegex(
                    script, rf'export {name}="\${{{name}:-{re.escape(value)}}}"'
                )

    # ── trace_pcap (docs/trace-pcap-design.md) ────────────────────────────────

    def _trace_pcap(self, env: dict | None = None, data: dict | None = None):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.getcwd()
            try:
                os.chdir(tmpdir)
                extra = dict(env or {})
                if data is not None:
                    path = Path(tmpdir) / "engine.yaml"
                    self._write(path, data)
                    extra["ENGINE_CONFIG_FILE"] = str(path)
                with patch.dict(os.environ, self._env(extra), clear=True):
                    return Settings.from_file_or_env().trace_pcap
            finally:
                os.chdir(old)

    def test_trace_pcap_defaults_are_off(self):
        t = self._trace_pcap()
        self.assertEqual((t.enabled, t.dir, t.max_streams, t.max_seconds, t.min_free_mb), (False, "logs/pcap", 0, 0, 5120))

    def test_trace_pcap_from_yaml_and_env(self):
        t = self._trace_pcap(data={"asr": {"model_path": "m.nemo"}, "trace_pcap": {
            "enabled": True, "dir": "/tmp/tr", "max_streams": 3, "max_seconds": 600, "min_free_mb": 0}})
        self.assertEqual((t.enabled, t.dir, t.max_streams, t.max_seconds, t.min_free_mb), (True, "/tmp/tr", 3, 600, 0))
        t = self._trace_pcap(env={"TRACE_PCAP_ENABLED": "true", "TRACE_PCAP_DIR": "x", "TRACE_PCAP_MAX_STREAMS": "2",
                                  "TRACE_PCAP_MAX_SECONDS": "5", "TRACE_PCAP_MIN_FREE_MB": "7"})
        self.assertEqual((t.enabled, t.dir, t.max_streams, t.max_seconds, t.min_free_mb), (True, "x", 2, 5, 7))

    def test_trace_pcap_rejects_negative_limits(self):
        for key in ("max_streams", "max_seconds", "min_free_mb"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"trace_pcap.{key}"):
                self._trace_pcap(data={"asr": {"model_path": "m.nemo"}, "trace_pcap": {key: -1}})

    def test_shipped_yaml_and_configmaps_keep_trace_pcap_off(self):
        root = Path(__file__).resolve().parents[1]
        shipped = root / "etc" / "nemotron-engine.yaml"
        with patch.dict(os.environ, self._env({"ENGINE_CONFIG_FILE": str(shipped)}), clear=True):
            t = Settings.from_file_or_env().trace_pcap
        self.assertEqual((t.enabled, t.dir, t.max_streams, t.max_seconds, t.min_free_mb), (False, "logs/pcap", 0, 0, 5120))
        names = {"TRACE_PCAP_ENABLED", "TRACE_PCAP_DIR", "TRACE_PCAP_MAX_STREAMS", "TRACE_PCAP_MAX_SECONDS",
                 "TRACE_PCAP_MIN_FREE_MB"}
        for name in ("k8s-configmap.yaml", "k8s-configmap-cpu.yaml", "k8s-configmap-gpu.yaml"):
            with self.subTest(configmap=name):
                data = yaml.safe_load((root / name).read_text(encoding="utf-8"))["data"]
                self.assertEqual(sorted(names - set(data)), [])
                self.assertEqual(data["TRACE_PCAP_ENABLED"], "false")

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
