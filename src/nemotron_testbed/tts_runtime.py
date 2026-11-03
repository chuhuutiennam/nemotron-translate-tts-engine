from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import resample
from .domain import QueueFullError, SubmitStatus, TtsJob, TtsResult
from .g711 import encode_g711
from .languages import short_language
from .logging_config import suppress_external_stdio
from .metrics import (
    TTS_AUDIO_SECONDS,
    TTS_LATENCY,
    TTS_NO_AUDIO_TOTAL,
    TTS_QUEUE,
    TTS_QUEUE_WAIT_MS,
    TTS_SYNTHESIS_RTF,
    TTS_WORKER_ERRORS,
)


logger = logging.getLogger(__name__)


class TtsRuntime:
    """Single-owner Piper runtime for English and Vietnamese voices.

    Each runtime owns independent Piper/ONNX Runtime sessions.

    ONNX Runtime thread counts are configured explicitly so ORT does not
    automatically create/pin a large number of CPU threads based on the
    Kubernetes host CPU count.

    The runtime synthesizes translated text, converts the result to mono
    target sample rate, and encodes it as raw G.711 payload.
    RTP packetization remains the Gateway's responsibility.
    """

    def __init__(
        self,
        en_model_path: str,
        en_config_path: str | None,
        vi_model_path: str,
        vi_config_path: str | None,
        target_sample_rate: int,
        queue_size: int,
        max_text_chars: int,
        length_scale: float,
        noise_scale: float,
        noise_w_scale: float,
        volume: float,
        result_callback: Callable[[TtsResult], None],
        worker_id: int = 0,
        suppress_model_init_output: bool = True,
        intra_threads: int = 2,
        device: str = "cpu",
    ) -> None:
        self.en_model_path = en_model_path
        self.en_config_path = en_config_path

        self.vi_model_path = vi_model_path
        self.vi_config_path = vi_config_path

        self.target_sample_rate = target_sample_rate
        self.max_text_chars = max_text_chars

        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.volume = volume

        self._result_callback = result_callback

        self.worker_id = worker_id
        self.suppress_model_init_output = suppress_model_init_output

        if intra_threads < 1:
            raise ValueError("TTS_INTRA_THREADS must be >= 1")

        self.intra_threads = intra_threads
        self.device = device

        self._jobs: queue.Queue[TtsJob | None] = queue.Queue(
            maxsize=queue_size
        )

        self._thread = threading.Thread(
            target=self._run,
            name=f"piper-tts-runtime-{worker_id}",
            daemon=True,
        )

        self._ready = threading.Event()
        self._stopped = threading.Event()

        self._startup_error: BaseException | None = None

        # Each TTS worker owns its own Piper/ORT sessions.
        self._voices: dict[str, object] = {}

        self._synthesis_config_cls = None

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def start(
        self,
        timeout: float = 180.0,
    ) -> None:
        self._thread.start()

        if not self._ready.wait(timeout):
            raise TimeoutError(
                "Timed out while loading Piper voices"
            )

        if self._startup_error is not None:
            raise RuntimeError(
                "TTS runtime failed to start"
            ) from self._startup_error

    def submit_nowait(
        self,
        job: TtsJob,
    ) -> None:
        if self._stopped.is_set():
            raise QueueFullError("TTS runtime is stopped")

        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise QueueFullError("TTS runtime queue is full") from exc

        TTS_QUEUE.labels(
            worker=str(self.worker_id)
        ).set(
            self._jobs.qsize()
        )

    def try_submit_nowait(self, job: TtsJob) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._jobs.put_nowait(job)
            TTS_QUEUE.labels(worker=str(self.worker_id)).set(self._jobs.qsize())
            return SubmitStatus.ACCEPTED
        except queue.Full:
            return SubmitStatus.OVERLOADED

    def shutdown(
        self,
        timeout: float = 10.0,
    ) -> None:
        if self._stopped.is_set():
            return

        try:
            self._jobs.put_nowait(None)

        except queue.Full:
            try:
                self._jobs.put(
                    None,
                    timeout=1.0,
                )
            except queue.Full:
                logger.warning(
                    "Could not enqueue TTS shutdown sentinel"
                )

        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "TtsRuntime worker=%s did not stop in %.1fs",
                self.worker_id,
                timeout,
            )
        self._stopped.set()

    def _emit(
        self,
        result: TtsResult,
    ) -> None:
        try:
            self._result_callback(result)

        except Exception:
            logger.exception(
                "TTS result callback failed"
            )

    @staticmethod
    def _parse_cuda_device_id(device: str) -> int:
        parts = device.split(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0

    def _build_providers(self) -> list:
        if self.device.startswith("cuda"):
            device_id = self._parse_cuda_device_id(self.device)
            return [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                        # HEURISTIC: cuDNN selects algorithm via built-in
                        # heuristics without exhaustive search. Avoids the
                        # large workspace of EXHAUSTIVE (CUDNN_STATUS_
                        # EXECUTION_FAILED on shared GPU) while finding a
                        # valid algorithm for Piper VITS Conv shapes (DEFAULT
                        # has no algo for these shapes → CPU Fallback mode).
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "cudnn_conv_use_max_workspace": "0",
                    },
                ),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def _load_voice(
        self,
        model_path: str,
        config_path: str | None,
    ):
        """Load Piper with explicit ONNX Runtime SessionOptions.

        PiperVoice.load() normally creates an ONNX Runtime session using
        default thread settings.

        On a large Kubernetes host, ORT may see many host CPUs and create
        a large thread pool. It may also try to set CPU affinity for those
        threads.

        If the container cpuset does not contain those CPUs, Linux returns
        EINVAL from pthread_setaffinity_np.

        Explicit intra/inter-op thread counts prevent that auto-sizing
        behavior and keep each Piper worker bounded.
        """

        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig

        model = Path(model_path)

        if not model.is_file():
            raise FileNotFoundError(
                f"Piper model not found: {model}"
            )

        if config_path:
            config = Path(config_path)
        else:
            config = Path(
                f"{model_path}.json"
            )

        if not config.is_file():
            raise FileNotFoundError(
                f"Piper config not found: {config}"
            )

        with config.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            config_dict = json.load(
                config_file
            )

        # ------------------------------------------------------
        # ONNX Runtime threading
        # ------------------------------------------------------

        session_options = ort.SessionOptions()

        # Critical for Kubernetes / cgroup CPU isolation.
        #
        # Do NOT leave intra_op_num_threads=0 here.
        # 0 lets ORT auto-size the thread pool based on visible host CPUs.
        session_options.intra_op_num_threads = (
            self.intra_threads
        )

        # We run each Piper runtime as its own worker already,
        # therefore ORT does not need additional inter-op parallelism.
        session_options.inter_op_num_threads = 1

        # Sequential is appropriate for one inference graph per worker
        # and avoids additional inter-op scheduling.
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # 0 = verbose / 3 = error / 4 = fatal
        session_options.log_severity_level = 3

        providers = self._build_providers()
        session = ort.InferenceSession(
            str(model),
            sess_options=session_options,
            providers=providers,
        )

        active_providers = session.get_providers()
        if self.device.startswith("cuda") and "CUDAExecutionProvider" not in active_providers:
            logger.warning(
                "TTS: CUDAExecutionProvider not available, "
                "running on CPU (active=%s worker=%s)",
                active_providers,
                self.worker_id,
            )
        else:
            logger.info(
                "TTS session loaded: worker=%s device=%s active_providers=%s "
                "cudnn_algo=%s",
                self.worker_id,
                self.device,
                active_providers,
                providers[0][1].get("cudnn_conv_algo_search", "n/a")
                if self.device.startswith("cuda") else "n/a",
            )

        piper_config = PiperConfig.from_dict(
            config_dict
        )

        return PiperVoice(
            session=session,
            config=piper_config,
        )

    def _load(self) -> None:
        try:
            from piper import SynthesisConfig

            self._synthesis_config_cls = (
                SynthesisConfig
            )

        except ImportError:
            # Compatibility with older Piper API.
            self._synthesis_config_cls = None

        logger.info(
            "Loading Piper English voice "
            "worker=%s device=%s intra_threads=%s from %s",
            self.worker_id,
            self.device,
            self.intra_threads,
            self.en_model_path,
        )

        self._voices["en"] = self._load_voice(
            self.en_model_path,
            self.en_config_path,
        )

        logger.info(
            "Loading Piper Vietnamese voice "
            "worker=%s device=%s intra_threads=%s from %s",
            self.worker_id,
            self.device,
            self.intra_threads,
            self.vi_model_path,
        )

        self._voices["vi"] = self._load_voice(
            self.vi_model_path,
            self.vi_config_path,
        )

        logger.info(
            "Piper TTS runtime ready: "
            "worker=%s device=%s intra_threads=%s",
            self.worker_id,
            self.device,
            self.intra_threads,
        )

    def _run(self) -> None:
        try:
            # Suppress stdout only — stderr must stay open so cuDNN JIT
            # kernel compilation succeeds when CUDA contexts from PyTorch
            # and CTranslate2 are already active in the same process.
            with suppress_external_stdio(
                self.suppress_model_init_output, stderr=False
            ):
                self._load()

        except BaseException as exc:
            self._startup_error = exc

            logger.exception(
                "TTS startup failed"
            )

            self._ready.set()
            self._stopped.set()

            return

        self._ready.set()

        while True:
            job = self._jobs.get()

            TTS_QUEUE.labels(
                worker=str(self.worker_id)
            ).set(
                self._jobs.qsize()
            )

            if job is None:
                self._jobs.task_done()
                break

            TTS_QUEUE_WAIT_MS.observe(
                (time.perf_counter() - job.enqueue_time) * 1000.0
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "POOL_TRACE TTS worker=%s stream=%s utterance=%s language=%s",
                    self.worker_id,
                    job.key.stream_id,
                    job.utterance_id,
                    job.target_language,
                )

            started = time.perf_counter()

            try:
                (
                    payload,
                    source_rate,
                    duration_s,
                ) = self._synthesize(job)

                elapsed = (
                    time.perf_counter()
                    - started
                )

                TTS_LATENCY.observe(
                    elapsed
                )

                TTS_AUDIO_SECONDS.inc(
                    duration_s
                )

                self._emit(
                    TtsResult(
                        key=job.key,
                        utterance_id=(
                            job.utterance_id
                        ),
                        text=job.text,
                        target_language=(
                            job.target_language
                        ),
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        audio_payload=payload,
                        audio_pt=job.audio_pt,
                        latency_ms=(
                            elapsed * 1000.0
                        ),
                        source_sample_rate=(
                            source_rate
                        ),
                        phrase_id=job.phrase_id,
                        is_last_phrase=job.is_last_phrase,
                        translated_text=job.translated_text,
                        segment_idx=job.segment_idx,
                        queue_ms=(started - job.enqueue_time) * 1000.0,
                    )
                )

            except Exception as exc:
                TTS_WORKER_ERRORS.labels(
                    worker=str(self.worker_id),
                    device=self.device,
                ).inc()
                logger.exception(
                    "TTS job failed",
                    extra={
                        "rpc_id": (
                            job.key.rpc_id
                        ),
                        "session_id": (
                            job.key.session_id
                        ),
                        "stream_id": (
                            job.key.stream_id
                        ),
                        "utterance_id": (
                            job.utterance_id
                        ),
                        "worker_id": (
                            self.worker_id
                        ),
                    },
                )

                self._emit(
                    TtsResult(
                        key=job.key,
                        utterance_id=(
                            job.utterance_id
                        ),
                        text=job.text,
                        target_language=(
                            job.target_language
                        ),
                        ts_start=job.ts_start,
                        ts_end=job.ts_end,
                        audio_pt=job.audio_pt,
                        latency_ms=(
                            (
                                time.perf_counter()
                                - started
                            )
                            * 1000.0
                        ),
                        error=(
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        phrase_id=job.phrase_id,
                        is_last_phrase=job.is_last_phrase,
                        translated_text=job.translated_text,
                        segment_idx=job.segment_idx,
                    )
                )

            finally:
                self._jobs.task_done()

        self._stopped.set()

        logger.info(
            "TTS runtime stopped: worker=%s",
            self.worker_id,
        )

    def _synthesis_config(self):
        if self._synthesis_config_cls is None:
            return None

        return self._synthesis_config_cls(
            volume=self.volume,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )

    def _voice_to_pcm(
        self,
        voice: object,
        text: str,
    ) -> tuple[np.ndarray, int]:
        """Synthesize Piper output to WAV in memory.

        The WAV API is used for compatibility with both the current
        and older Piper Python APIs.
        """

        buffer = io.BytesIO()

        if hasattr(voice, "synthesize_wav"):
            # synthesize_wav(text, wav_file, ...) writes directly into a
            # wave.Wave_write object (it calls wav_file.setframerate, etc.).
            # Manage wav_file manually to avoid wave.Wave_write.__exit__
            # raising wave.Error: # channels not specified which would mask
            # the real exception when Piper fails before writing any audio.
            syn_config = self._synthesis_config()
            wav_file = wave.open(buffer, "wb")
            _t_synth = time.perf_counter()
            try:
                if syn_config is not None:
                    voice.synthesize_wav(text, wav_file, syn_config=syn_config)
                else:
                    voice.synthesize_wav(text, wav_file)
            except Exception:
                # Discard wav_file without closing — wave.Wave_write.__del__
                # would call close() and raise wave.Error (# channels not
                # specified) as a noisy GC warning. Setting _file=None makes
                # wave.Wave_write.close() return early without error.
                wav_file._file = None
                raise
            else:
                _synth_ms = (time.perf_counter() - _t_synth) * 1000.0
                logger.debug(
                    "TTS synthesize_wav: worker=%s device=%s chars=%d synth_ms=%.1f",
                    self.worker_id,
                    self.device,
                    len(text),
                    _synth_ms,
                )
                try:
                    wav_file.close()
                except wave.Error:
                    # synthesize_wav returned normally but wrote nothing —
                    # the text produced no phonemes (punctuation-only, etc.).
                    logger.warning(
                        "Piper synthesize_wav produced no audio: "
                        "worker=%s text=%r",
                        self.worker_id,
                        text[:120],
                    )
                    TTS_NO_AUDIO_TOTAL.labels(
                        worker=str(self.worker_id),
                    ).inc()
                    return np.zeros(0, dtype=np.float32), 0
        else:
            # Compatibility with archived Piper API.
            with wave.open(buffer, "wb") as wav_file:
                voice.synthesize(text, wav_file)

        buffer.seek(0)

        with wave.open(
            buffer,
            "rb",
        ) as reader:

            channels = (
                reader.getnchannels()
            )

            width = (
                reader.getsampwidth()
            )

            rate = (
                reader.getframerate()
            )

            frames = reader.readframes(
                reader.getnframes()
            )

        if width != 2:
            raise RuntimeError(
                "Piper returned unsupported "
                f"sample width: {width}"
            )

        pcm = np.frombuffer(
            frames,
            dtype="<i2",
        )

        if channels > 1:
            pcm = (
                pcm
                .reshape(
                    -1,
                    channels,
                )
                .astype(
                    np.float32
                )
                .mean(
                    axis=1
                )
                .astype(
                    np.int16
                )
            )

        audio = (
            pcm.astype(
                np.float32
            )
            / 32768.0
        )

        return (
            audio,
            int(rate),
        )

    def _synthesize(
        self,
        job: TtsJob,
    ) -> tuple[
        bytes,
        int,
        float,
    ]:
        text = job.text.strip()

        if not text:
            return (
                b"",
                0,
                0.0,
            )

        if len(text) > self.max_text_chars:
            raise ValueError(
                f"TTS text length {len(text)} "
                "exceeds "
                f"TTS_MAX_TEXT_CHARS="
                f"{self.max_text_chars}"
            )

        lang = short_language(
            job.target_language
        )

        if lang not in self._voices:
            raise ValueError(
                "No Piper voice configured for "
                "target language "
                f"{job.target_language}"
            )

        _t0 = time.perf_counter()

        audio, source_rate = (
            self._voice_to_pcm(
                self._voices[lang],
                text,
            )
        )

        if audio.size == 0:
            TTS_NO_AUDIO_TOTAL.labels(worker=str(self.worker_id)).inc()
            logger.warning(
                "TTS produced no audio: worker=%s text=%r",
                self.worker_id,
                text[:120],
            )
            return (b"", 0, 0.0)

        audio_target = resample(
            audio,
            source_rate,
            self.target_sample_rate,
        )

        payload = encode_g711(
            audio_target,
            job.audio_pt,
        )

        duration_s = (
            float(audio_target.size)
            / float(
                self.target_sample_rate
            )
        )

        _total_ms = (time.perf_counter() - _t0) * 1000.0
        # RTF < 0.1 → GPU; RTF 0.3–2.0 → CPU fallback
        _rtf = _total_ms / (duration_s * 1000.0) if duration_s > 0 else 0.0
        logger.info(
            "TTS synthesize: worker=%s device=%s lang=%s chars=%d "
            "audio_s=%.2f payload_bytes=%d total_ms=%.1f RTF=%.3f",
            self.worker_id,
            self.device,
            lang,
            len(text),
            duration_s,
            len(payload),
            _total_ms,
            _rtf,
        )
        TTS_SYNTHESIS_RTF.labels(
            worker=str(self.worker_id),
            device=self.device,
        ).observe(_rtf)

        return (
            payload,
            source_rate,
            duration_s,
        )