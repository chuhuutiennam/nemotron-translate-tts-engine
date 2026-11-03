import io
import json
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from nemotron_testbed.logging_config import (
    PIPELINE_TRACE_LOGGERS,
    JsonFormatter,
    configure_logging,
    suppress_external_stdio,
)


class LoggingConfigTest(unittest.TestCase):
    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            handler.close()
        root.handlers.clear()
        logging.captureWarnings(False)
        for name in PIPELINE_TRACE_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)

    def test_console_can_be_quieter_than_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "engine.log"
            output = io.StringIO()
            with redirect_stdout(output):
                configure_logging(
                    "INFO",
                    str(log_path),
                    console_level="ERROR",
                    third_party_level="ERROR",
                )
                logger = logging.getLogger("nemotron_testbed.test")
                logger.info("info-to-file-only")
                logger.error("error-to-console-and-file")
                for handler in logging.getLogger().handlers:
                    handler.flush()

            console_text = output.getvalue()
            file_text = log_path.read_text(encoding="utf-8")
            # On Windows, RotatingFileHandler holds an open file lock until
            # explicitly closed.  Close before TemporaryDirectory.__exit__
            # tries to delete the directory, or we get PermissionError.
            root = logging.getLogger()
            for handler in list(root.handlers):
                handler.close()
            root.handlers.clear()

        self.assertNotIn("info-to-file-only", console_text)
        self.assertIn("error-to-console-and-file", console_text)
        self.assertIn("info-to-file-only", file_text)
        self.assertIn("error-to-console-and-file", file_text)

    def test_file_handler_rotates_daily_and_keeps_backup_count_days(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "engine.log"
            configure_logging("INFO", str(log_path), backup_count=7)
            file_handlers = [
                h for h in logging.getLogger().handlers
                if isinstance(h, TimedRotatingFileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            handler = file_handlers[0]
            self.assertEqual(handler.when, "MIDNIGHT")
            self.assertEqual(handler.backupCount, 7)
            self.assertEqual(handler.suffix, "%Y-%m-%d")

            logging.getLogger("nemotron_testbed.test").info("before-rotate")
            handler.doRollover()
            logging.getLogger("nemotron_testbed.test").info("after-rotate")
            handler.flush()

            rotated = [p for p in Path(tmpdir).iterdir() if p.name.startswith("engine.log.")]
            self.assertEqual(len(rotated), 1)
            self.assertIn("before-rotate", rotated[0].read_text(encoding="utf-8"))
            current = log_path.read_text(encoding="utf-8")
            self.assertIn("after-rotate", current)
            self.assertNotIn("before-rotate", current)

            root = logging.getLogger()
            for h in list(root.handlers):
                h.close()
            root.handlers.clear()

    def _run_with_file(self, level, trace_level):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "engine.log"
            configure_logging(level, str(log_path), pipeline_trace_level=trace_level)
            logging.getLogger("nemotron_testbed.coordinator").debug("plain-debug")
            logging.getLogger("nemotron_testbed.coordinator").info("plain-info")
            logging.getLogger(PIPELINE_TRACE_LOGGERS[0]).log(logging.DEBUG, "trace-debug")
            logging.getLogger(PIPELINE_TRACE_LOGGERS[1]).log(logging.DEBUG, "trace-debug-svc")
            root = logging.getLogger()
            for handler in list(root.handlers):
                handler.flush()
            text = log_path.read_text(encoding="utf-8")
            for handler in list(root.handlers):
                handler.close()
            root.handlers.clear()
        return text

    def test_trace_lines_reach_file_when_level_is_info(self):
        text = self._run_with_file("INFO", "DEBUG")
        self.assertIn("plain-info", text)
        self.assertNotIn("plain-debug", text)
        self.assertIn("trace-debug", text)
        self.assertIn("trace-debug-svc", text)

    def test_trace_lines_follow_level_when_trace_disabled(self):
        text = self._run_with_file("INFO", None)
        self.assertIn("plain-info", text)
        self.assertNotIn("plain-debug", text)
        self.assertNotIn("trace-debug", text)

    def test_trace_loggers_are_the_ones_used_by_modules(self):
        from nemotron_testbed import coordinator, service

        self.assertEqual(coordinator.trace_logger.name, PIPELINE_TRACE_LOGGERS[0])
        self.assertEqual(service.trace_logger.name, PIPELINE_TRACE_LOGGERS[1])

    def test_suppress_external_stdio_keeps_python_stream_objects_stable(self):
        stdout_before = os.sys.stdout
        stderr_before = os.sys.stderr
        with suppress_external_stdio(True):
            # Direct writes to fd 1/2 are discarded, while the Python objects
            # themselves are not replaced with a temporary file.
            os.write(1, b"hidden-fd-stdout\n")
            os.write(2, b"hidden-fd-stderr\n")
            self.assertIs(os.sys.stdout, stdout_before)
            self.assertIs(os.sys.stderr, stderr_before)
        self.assertIs(os.sys.stdout, stdout_before)
        self.assertIs(os.sys.stderr, stderr_before)
        self.assertFalse(getattr(stdout_before, "closed", False))
        self.assertFalse(getattr(stderr_before, "closed", False))

    def test_handler_created_inside_suppression_is_not_closed_after_exit(self):
        logger = logging.getLogger("test.fd.suppression")
        logger.propagate = False
        logger.handlers.clear()
        try:
            with suppress_external_stdio(True):
                handler = logging.StreamHandler(os.sys.stderr)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.info("suppressed")
            self.assertFalse(handler.stream.closed)
            # This used to raise ValueError when redirect_stderr(devnull) was
            # used and the handler captured the temporary devnull stream.
            logger.info("still-writable")
        finally:
            for handler in list(logger.handlers):
                handler.close()
            logger.handlers.clear()
            logger.propagate = True


class JsonFormatterFieldsTest(unittest.TestCase):
    def test_segment_fields_are_serialised(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "PIPELINE SEGMENT_COMMIT", None, None)
        fields = {
            "stage": "SEGMENT_COMMIT", "segment_idx": 0, "kind": "punct", "gain_ms": 12.5,
            "premature": False, "revised": True, "forced_cut": False, "language_known": True,
            "phase": "final",
        }
        for key, value in fields.items():
            setattr(record, key, value)
        payload = json.loads(JsonFormatter().format(record))
        for key, value in fields.items():
            self.assertEqual(payload[key], value)

    def test_stream_open_language_source_fields_are_serialised(self):
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "PIPELINE STREAM_OPEN", None, None)
        fields = {"stage": "STREAM_OPEN", "language_source": "vi", "source_mode": "explicit"}
        for key, value in fields.items():
            setattr(record, key, value)
        payload = json.loads(JsonFormatter().format(record))
        for key, value in fields.items():
            self.assertEqual(payload[key], value)


if __name__ == "__main__":
    unittest.main()
