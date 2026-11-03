from __future__ import annotations

import itertools
import logging
import math
import os
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .domain import (
    QueueFullError,
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeResult,
    RuntimeResultType,
    StreamKey,
    SubmitStatus,
)
from .metrics import ASR_FINALIZE_STEPS, ASR_QUEUE_WAIT_MS, MODEL_STEP_SECONDS, RUNTIME_QUEUE
from .logging_config import quiet_nemo_logging, suppress_external_stdio

logger = logging.getLogger(__name__)

_LANG_TAG_ANY_RE = re.compile(r"<([a-zA-Z]{2}(?:-[a-zA-Z]{2})?)>", re.IGNORECASE)
_LANG_TAG_PATTERN = r"<([a-zA-Z]{2}(?:-[a-zA-Z]{2})??)>"


def _check_mel_config(cfg) -> None:
    """The overlapped mel (docs/asr-feature-fix-design.md) needs centered frames and no normalization over the whole
    input; with anything else it would give wrong features without any sign, so refuse to start."""
    if cfg.get("exact_pad", False):
        raise RuntimeError("ASR preprocessor exact_pad=true is not supported (docs/asr-feature-fix-design.md)")
    if cfg.get("normalize") in ("per_feature", "all_features"):
        raise RuntimeError(
            f"ASR preprocessor normalize={cfg.get('normalize')!r} is not supported (docs/asr-feature-fix-design.md)"
        )


@dataclass(slots=True)
class _ModelStreamState:
    key: StreamKey
    utterance_id: int
    language: str
    cache_last_channel: object
    cache_last_time: object
    cache_last_channel_len: object
    previous_hypotheses: object = None
    pred_out_stream: object = None
    step_num: int = 0
    feature_index: int = 0
    raw_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    # docs/asr-feature-fix-design.md: the mel of new audio is computed together with the audio before it (mel_tail),
    # so feature_cache holds the mel of the whole utterance; mel_frames = frames already in feature_cache.
    mel_tail: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    mel_frames: int = 0
    mel_final: bool = False
    feature_cache: object = None  # torch.Tensor [1, D, T] or None
    last_text: str = ""
    last_confidence: float = 0.0
    confidence_source: str = "unavailable"
    detected_language: str = ""
    # Beam search returns only per-chunk text; accumulate across chunks here.
    accumulated_text: str = ""


@dataclass(order=True, slots=True)
class _PrioritizedCommand:
    priority: int
    sequence: int
    command: RuntimeCommand = field(compare=False)
    enqueue_time: float = field(compare=False, default_factory=time.perf_counter)


def _command_priority(command_type: RuntimeCommandType) -> int:
    # Free resources and close utterances before normal streaming work.
    return {
        RuntimeCommandType.SHUTDOWN: -10,
        RuntimeCommandType.CANCEL: 0,
        RuntimeCommandType.FINALIZE: 1,
        RuntimeCommandType.START: 2,
        RuntimeCommandType.STEP: 3,
    }[command_type]


class NemotronRuntime:
    """Single-owner runtime for Nemotron weights and all per-stream caches.

    No gRPC/asyncio task may call NeMo directly. FINALIZE/CANCEL commands receive
    higher priority than normal STEP work so a stream does not remain stuck in
    FINALIZING while new audio accumulates in the deferred buffer.
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        language: str,
        left_context: int,
        lookahead_tokens: int,
        queue_size: int,
        warmup_seconds: float,
        result_callback: Callable[[RuntimeResult], None],
        suppress_model_init_output: bool = True,
        beam_size: int = 1,
        worker_id: int = 0,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.default_language = language
        self.left_context = left_context
        self.lookahead_tokens = lookahead_tokens
        self.warmup_seconds = warmup_seconds
        self.suppress_model_init_output = suppress_model_init_output
        self.beam_size = max(1, int(beam_size))
        self._result_callback = result_callback
        self.worker_id = worker_id

        self._commands: queue.PriorityQueue[_PrioritizedCommand] = queue.PriorityQueue(
            maxsize=queue_size
        )
        self._sequence = itertools.count()
        self._thread = threading.Thread(
            target=self._run, name=f"nemotron-runtime-{worker_id}", daemon=True
        )
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: BaseException | None = None
        self._queue_wait_ms = 0.0

        self.model = None
        self.preprocessor = None
        self.streaming_cfg = None
        self.compute_dtype = None
        self.sample_rate = 16_000
        self.mel_hop = 160  # set from the model's preprocessor at load
        self.mel_half_window = 256  # n_fft // 2
        # Temporary, Kaggle A/B only (docs/asr-feature-fix-design.md Q1): the mel path before the fix. Remove after.
        self.legacy_features = os.environ.get("ASR_FEATURE_LEGACY", "") == "1"
        # Key includes utterance_id so two consecutive utterances on the same
        # logical stream can coexist during finalize-overlap.
        self._states: dict[tuple[StreamKey, int], _ModelStreamState] = {}
        self._current_prompt_language: str | None = None
        self._strip_lang_tags_unsupported: bool = False

    @property
    def queue_depth(self) -> int:
        return self._commands.qsize()

    def start(self, timeout: float = 300.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("Timed out while loading Nemotron model")
        if self._startup_error is not None:
            raise RuntimeError("Nemotron runtime failed to start") from self._startup_error

    def _wrap(self, command: RuntimeCommand) -> _PrioritizedCommand:
        return _PrioritizedCommand(
            priority=_command_priority(command.type),
            sequence=next(self._sequence),
            command=command,
        )

    def submit(self, command: RuntimeCommand, timeout: float = 2.0) -> None:
        if self._stopped.is_set():
            raise QueueFullError("Nemotron runtime is stopped")
        try:
            self._commands.put(self._wrap(command), timeout=timeout)
        except queue.Full as exc:
            raise QueueFullError("Nemotron runtime queue is full") from exc
        RUNTIME_QUEUE.labels(worker=str(self.worker_id)).set(self._commands.qsize())

    def submit_nowait(self, command: RuntimeCommand) -> None:
        if self._stopped.is_set():
            raise QueueFullError("Nemotron runtime is stopped")
        try:
            self._commands.put_nowait(self._wrap(command))
        except queue.Full as exc:
            raise QueueFullError("Nemotron runtime queue is full") from exc
        RUNTIME_QUEUE.labels(worker=str(self.worker_id)).set(self._commands.qsize())

    def try_submit_nowait(self, command: RuntimeCommand) -> SubmitStatus:
        if self._stopped.is_set():
            return SubmitStatus.UNAVAILABLE
        try:
            self._commands.put_nowait(self._wrap(command))
            RUNTIME_QUEUE.labels(worker=str(self.worker_id)).set(self._commands.qsize())
            return SubmitStatus.ACCEPTED
        except queue.Full:
            return SubmitStatus.OVERLOADED

    def shutdown(self, timeout: float = 10.0) -> None:
        if self._stopped.is_set():
            return
        try:
            self.submit(RuntimeCommand(type=RuntimeCommandType.SHUTDOWN), timeout=1.0)
        except Exception:
            logger.exception("Failed to enqueue runtime shutdown")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.error(
                "NemotronRuntime worker=%s did not stop in %.1fs",
                self.worker_id,
                timeout,
            )
        self._stopped.set()

    def _emit(self, result: RuntimeResult) -> bool:
        try:
            self._result_callback(result)
            return True
        except Exception:
            logger.exception("Runtime result callback failed")
            return False

    def _run(self) -> None:
        try:
            self._load_model()
            self._warmup()
        except BaseException as exc:
            self._startup_error = exc
            logger.exception("Nemotron startup failed")
            self._ready.set()
            self._stopped.set()
            return

        self._ready.set()
        logger.info("Nemotron runtime ready: worker=%s", self.worker_id)

        while True:
            queued = self._commands.get()
            command = queued.command

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "POOL_TRACE ASR execute worker=%s command=%s stream=%s utterance=%s",
                    self.worker_id,
                    command.type.name,
                    command.key.stream_id if command.key else "",
                    command.utterance_id,
                )

            RUNTIME_QUEUE.labels(worker=str(self.worker_id)).set(self._commands.qsize())
            self._queue_wait_ms = (time.perf_counter() - queued.enqueue_time) * 1000.0
            ASR_QUEUE_WAIT_MS.observe(self._queue_wait_ms)
            started = time.perf_counter()
            try:
                if command.type is RuntimeCommandType.SHUTDOWN:
                    self._states.clear()
                    self._emit(RuntimeResult(type=RuntimeResultType.SHUTDOWN))
                    break
                self._execute(command)
            except Exception:
                logger.exception(
                    "Runtime command failed",
                    extra={
                        "command": command.type.name,
                        "rpc_id": command.key.rpc_id if command.key else None,
                        "session_id": command.key.session_id if command.key else None,
                        "stream_id": command.key.stream_id if command.key else None,
                        "utterance_id": command.utterance_id,
                        "worker_id": self.worker_id,
                    },
                )
                self._emit(
                    RuntimeResult(
                        type=RuntimeResultType.ERROR,
                        key=command.key,
                        utterance_id=command.utterance_id,
                        error=traceback.format_exc(),
                    )
                )
            finally:
                if command.type is not RuntimeCommandType.SHUTDOWN:
                    MODEL_STEP_SECONDS.labels(command=command.type.name).observe(
                        time.perf_counter() - started
                    )
                self._commands.task_done()

        self._stopped.set()
        logger.info("Nemotron runtime stopped: worker=%s", self.worker_id)

    def _load_model(self) -> None:
        import torch

        torch.set_float32_matmul_precision("high")

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {model_file}")

        logger.info(
            "Loading Nemotron model worker=%s from %s on %s",
            self.worker_id,
            model_file,
            self.device,
        )

        # NeMo can emit a large amount of informational output while restoring
        # the model (for example supported-language listings). Keep that third-
        # party startup chatter out of the console by default. If restoration
        # fails, the exception leaves this context and is logged by _run().
        with suppress_external_stdio(self.suppress_model_init_output, stderr=False):
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
                EncDecRNNTBPEModelWithPrompt,
            )
            from omegaconf import OmegaConf, open_dict

            quiet_nemo_logging("ERROR")
            self.model = EncDecRNNTBPEModelWithPrompt.restore_from(
                restore_path=str(model_file),
                map_location=self.device,
            )
            self.model.eval()

            if hasattr(self.model.encoder, "set_default_att_context_size"):
                self.model.encoder.set_default_att_context_size(
                    att_context_size=[self.left_context, self.lookahead_tokens]
                )
            if hasattr(self.model.encoder, "setup_streaming_params"):
                self.model.encoder.setup_streaming_params()
            self.streaming_cfg = self.model.encoder.streaming_cfg

            if hasattr(self.model, "change_decoding_strategy"):
                with open_dict(self.model.cfg.decoding):
                    self.model.cfg.decoding.fused_batch_size = -1
                    if self.beam_size > 1:
                        self.model.cfg.decoding.strategy = "beam"
                        if not hasattr(self.model.cfg.decoding, "beam"):
                            self.model.cfg.decoding.beam = OmegaConf.create({})
                        self.model.cfg.decoding.beam.beam_size = self.beam_size
                        self.model.cfg.decoding.beam.score_norm = True
                        self.model.cfg.decoding.beam.return_best_hypothesis = True
                    else:
                        # Force strategy="greedy" (GreedyRNNTInfer) instead of
                        # "greedy_batch" (GreedyBatchedRNNTInfer). In this NeMo version
                        # "greedy_batch" always uses CUDA graphs via *LabelLoopingComputer
                        # classes that have no pytorch fallback. GreedyRNNTInfer produces
                        # identical output for batch_size=1 streaming and never uses CUDA
                        # graphs.
                        self.model.cfg.decoding.strategy = "greedy"
                    # Per-token max-softmax confidence. GreedyRNNTInfer populates
                    # hypothesis.token_confidence when preserve_token_confidence=True,
                    # giving calibrated per-token probabilities in [0,1]. Falls back to
                    # rnnt_score_geomean if this NeMo build does not support it.
                    try:
                        if not hasattr(self.model.cfg.decoding, "confidence_cfg"):
                            self.model.cfg.decoding.confidence_cfg = OmegaConf.create({})
                        self.model.cfg.decoding.confidence_cfg.preserve_token_confidence = True
                        self.model.cfg.decoding.confidence_cfg.exclude_blank = True
                        self.model.cfg.decoding.confidence_cfg.aggregation = "mean"
                        if not hasattr(self.model.cfg.decoding.confidence_cfg, "method_cfg"):
                            self.model.cfg.decoding.confidence_cfg.method_cfg = OmegaConf.create({})
                        self.model.cfg.decoding.confidence_cfg.method_cfg.name = "max_prob"
                    except Exception:
                        logger.debug(
                            "Token confidence config not supported on this NeMo build; "
                            "using rnnt_score_geomean fallback"
                        )
                self.model.change_decoding_strategy(self.model.cfg.decoding)

            if not hasattr(self.model, "set_inference_prompt"):
                raise RuntimeError(
                    "This NeMo Speech build/model does not expose set_inference_prompt()."
                )
            self._ensure_prompt(self.default_language)

            cfg = OmegaConf.to_container(self.model.cfg.preprocessor, resolve=True)
            cfg = OmegaConf.create(cfg)
            _check_mel_config(cfg)
            with open_dict(cfg):
                cfg.dither = 0.0
                cfg.pad_to = 0
                if cfg.get("normalize", "NA") not in ("per_feature", "all_features"):
                    cfg.normalize = "None"
            self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(cfg).to(self.device)
            self.preprocessor.eval()
            # read now so a featurizer without them fails at startup, not in a call
            self.mel_hop = int(self.preprocessor.featurizer.hop_length)
            self.mel_half_window = int(self.preprocessor.featurizer.n_fft) // 2

        self.compute_dtype = next(self.model.parameters()).dtype
        self.sample_rate = int(
            getattr(self.model.cfg, "sample_rate", self.model.cfg.preprocessor.sample_rate)
        )
        torch.set_grad_enabled(False)

        _conf_mode = "unavailable"
        try:
            _conf_mode = getattr(
                self.model.cfg.decoding.confidence_cfg, "method_cfg", None
            )
            _conf_mode = getattr(_conf_mode, "name", "unavailable") if _conf_mode else "unavailable"
        except Exception:
            pass
        logger.info(
            "Nemotron loaded: sample_rate=%s dtype=%s attention_context=[%s,%s] "
            "streaming_cfg=%s decoding=%s beam_size=%s confidence=%s",
            self.sample_rate,
            self.compute_dtype,
            self.left_context,
            self.lookahead_tokens,
            self.streaming_cfg,
            "beam" if self.beam_size > 1 else "greedy",
            self.beam_size if self.beam_size > 1 else "n/a",
            _conf_mode,
        )
        if self.legacy_features:
            logger.warning(
                "ASR_FEATURE_LEGACY=1: mel computed per piece and no pre-encode cache at step 2 "
                "(the path before docs/asr-feature-fix-design.md; Kaggle A/B only)"
            )

    def _ensure_prompt(self, language: str) -> None:
        if self._current_prompt_language == language:
            return
        with suppress_external_stdio(self.suppress_model_init_output, stderr=False):
            self.model.set_inference_prompt(language)
            try:
                self.model.decoding.set_strip_lang_tags(
                    language != "auto",
                    lang_tag_pattern=_LANG_TAG_PATTERN,
                )
            except TypeError:
                self.model.decoding.set_strip_lang_tags(language != "auto")
            except Exception:
                if not self._strip_lang_tags_unsupported:
                    self._strip_lang_tags_unsupported = True
                    logger.warning(
                        "set_strip_lang_tags unsupported on this NeMo build; "
                        "language auto-detection will not work (language=%s)",
                        language,
                    )
        self._current_prompt_language = language

    def _warmup(self) -> None:
        if self.warmup_seconds <= 0:
            return
        key = StreamKey("__warmup__", "__warmup__", "__warmup__")
        started = time.perf_counter()
        with suppress_external_stdio(self.suppress_model_init_output, stderr=False):
            self._create_state(key, 1, self.default_language)
            state = self._states[(key, 1)]
            state.raw_audio = np.zeros(
                max(1, int(self.sample_rate * self.warmup_seconds)), dtype=np.float32
            )
            for _ in range(128):
                text = self._run_one_step(state, flush=True)
                if text is None:
                    break
                if state.feature_index >= (int(state.feature_cache.size(-1)) if state.feature_cache is not None else 0):
                    break
            self._states.pop((key, 1), None)
        logger.info("Nemotron warm-up completed in %.2fs", time.perf_counter() - started)

    def _execute(self, command: RuntimeCommand) -> None:
        if command.key is None:
            raise ValueError(f"{command.type.name} command requires a StreamKey")

        if command.type is RuntimeCommandType.START:
            self._create_state(command.key, command.utterance_id, command.language)
            emitted = self._emit(
                RuntimeResult(
                    type=RuntimeResultType.STARTED,
                    key=command.key,
                    utterance_id=command.utterance_id,
                )
            )
            if not emitted:
                self._states.pop((command.key, command.utterance_id), None)
            return

        if command.type is RuntimeCommandType.CANCEL:
            state_key = (command.key, command.utterance_id)
            self._states.pop(state_key, None)
            self._emit(
                RuntimeResult(
                    type=RuntimeResultType.CANCELLED,
                    key=command.key,
                    utterance_id=command.utterance_id,
                )
            )
            return

        state_key = (command.key, command.utterance_id)
        state = self._states.get(state_key)

        if state is None:
            # CANCEL has higher queue priority than STEP/FINALIZE.
            # When RPC is cancelled while STEP/FINALIZE already queued,
            # CANCEL removes the state first — these are stale, not failures.
            if command.type in (RuntimeCommandType.STEP, RuntimeCommandType.FINALIZE):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Ignoring stale ASR command after runtime state removal: "
                        "worker=%s command=%s rpc=%s session=%s stream=%s utterance=%s",
                        getattr(self, "worker_id", 0),
                        command.type.name,
                        command.key.rpc_id,
                        command.key.session_id,
                        command.key.stream_id,
                        command.utterance_id,
                    )
                return
            raise RuntimeError(
                f"No runtime state for {command.key} utterance={command.utterance_id}"
            )

        if command.audio.size:
            state.raw_audio = np.concatenate(
                [state.raw_audio, np.asarray(command.audio, dtype=np.float32).reshape(-1)]
            )

        started = time.perf_counter()
        if command.type is RuntimeCommandType.STEP:
            text = self._run_one_step(state, flush=False)
            self._emit(
                RuntimeResult(
                    type=RuntimeResultType.PARTIAL,
                    key=command.key,
                    utterance_id=command.utterance_id,
                    text=text if text is not None else state.last_text,
                    confidence=state.last_confidence,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    detected_language=state.detected_language,
                    metadata={
                        "model_step_executed": text is not None,
                        "confidence_source": state.confidence_source,
                        "queue_ms": self._queue_wait_ms,
                    },
                )
            )
            return

        if command.type is RuntimeCommandType.FINALIZE:
            guard = 0
            while guard < 1024:
                guard += 1
                text = self._run_one_step(state, flush=True)
                if text is None:
                    break
                if state.feature_index >= (int(state.feature_cache.size(-1)) if state.feature_cache is not None else 0):
                    break
            ASR_FINALIZE_STEPS.observe(guard)
            final_text = state.last_text
            final_confidence = state.last_confidence
            final_detected_lang = state.detected_language
            final_confidence_source = state.confidence_source
            self._states.pop((command.key, command.utterance_id), None)
            self._emit(
                RuntimeResult(
                    type=RuntimeResultType.FINAL,
                    key=command.key,
                    utterance_id=command.utterance_id,
                    text=final_text,
                    confidence=final_confidence,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    detected_language=final_detected_lang,
                    metadata={"confidence_source": final_confidence_source},
                )
            )
            return

        raise ValueError(f"Unsupported runtime command: {command.type}")

    def _create_state(self, key: StreamKey, utterance_id: int, language: str) -> None:
        if self.model is None:
            raise RuntimeError("model not loaded")
        caches = self.model.encoder.get_initial_cache_state(batch_size=1)
        if len(caches) != 3:
            raise RuntimeError(f"Expected 3 encoder caches, got {len(caches)}")
        self._states[(key, utterance_id)] = _ModelStreamState(
            key=key,
            utterance_id=utterance_id,
            language=language,
            cache_last_channel=caches[0],
            cache_last_time=caches[1],
            cache_last_channel_len=caches[2],
        )

    def _update_feature_cache(self, state: _ModelStreamState, final: bool = False) -> None:
        """Append the mel of the new audio (state.raw_audio) to feature_cache.

        docs/asr-feature-fix-design.md: the mel is computed over mel_tail + the new audio and only frames whose whole
        window lies in that audio are kept, so feature_cache is the mel of the whole utterance. The last frames wait
        for the next audio; `final` (FINALIZE: the end of the utterance) releases them.
        """
        import torch

        if self.legacy_features:
            self._update_feature_cache_legacy(state)
            return
        if state.mel_final or (state.raw_audio.size == 0 and not final):
            return
        audio = np.concatenate([state.mel_tail, state.raw_audio]) if state.mel_tail.size else state.raw_audio
        state.raw_audio = np.empty(0, dtype=np.float32)
        state.mel_final = final
        if audio.size == 0:
            return
        hop, half = self.mel_hop, self.mel_half_window
        margin = math.ceil((half + 1) / hop)  # the first frames of `audio` see padding instead of the audio before
        start = max(0, state.mel_frames - margin) * hop  # position of audio[0] in the utterance
        features, length = self._features(audio)
        first = state.mel_frames - start // hop
        if final:
            last = int(length[0])  # the real end: NeMo pads it as it pads the end of the whole utterance
        else:
            last = (audio.size - half) // hop + 1 if audio.size >= half else 0
        if last > first:
            new = features[:, :, first:last]
            state.feature_cache = new if state.feature_cache is None else torch.cat([state.feature_cache, new], dim=-1)
            state.mel_frames += last - first
        state.mel_tail = audio[max(0, state.mel_frames - margin) * hop - start :].copy()  # not a view of `audio`

    def _update_feature_cache_legacy(self, state: _ModelStreamState) -> None:
        """ASR_FEATURE_LEGACY=1 (temporary, Kaggle A/B): mel of each piece on its own, every frame kept."""
        import torch

        if state.raw_audio.size == 0:
            return
        new_features, _ = self._features(state.raw_audio)
        if state.feature_cache is None:
            state.feature_cache = new_features
        else:
            state.feature_cache = torch.cat([state.feature_cache, new_features], dim=-1)
        state.raw_audio = np.empty(0, dtype=np.float32)

    def _features(self, raw_audio: np.ndarray):
        """(mel [1, D, T], valid frame count [1]) of raw_audio; frames past the count are padding."""
        import torch

        signal = torch.from_numpy(raw_audio.astype(np.float32, copy=False)).unsqueeze(0)
        signal = signal.to(self.device)
        length = torch.tensor([raw_audio.size], device=self.device)
        with torch.inference_mode():
            features, length = self.preprocessor(input_signal=signal, length=length)
        return features, length

    @staticmethod
    def _indexed(value, index: int) -> int:
        try:
            return int(value[index])
        except TypeError:
            return int(value)

    @staticmethod
    def _parse_transcript(text: str, language: str) -> tuple[str, str]:
        """Return (clean_text, detected_language).

        For language="auto": find the last <lang-TAG> anywhere in the text as
        detected_language, then strip all tags.
        For explicit languages: strip all lang tags (mid-sentence or trailing),
        return language unchanged.
        Whitespace left by removed mid-sentence tags is collapsed.
        """
        cleaned = (text or "").strip()
        if language == "auto":
            all_tags = _LANG_TAG_ANY_RE.findall(cleaned)
            detected = all_tags[-1] if all_tags else ""
            cleaned = re.sub(r"\s+", " ", _LANG_TAG_ANY_RE.sub("", cleaned)).strip()
            return cleaned, detected
        cleaned = re.sub(r"\s+", " ", _LANG_TAG_ANY_RE.sub("", cleaned)).strip()
        return cleaned, language

    @staticmethod
    def _flatten_numeric(value: object) -> list[float]:
        """Best-effort conversion of NeMo confidence containers to floats.

        NeMo may expose confidence as Python lists, nested RNNT TxU lists,
        NumPy arrays, or torch tensors depending on the decoder/version.
        Keep this helper dependency-light so unit tests do not need NeMo/torch.
        """
        if value is None:
            return []
        if isinstance(value, (int, float, np.number)):
            number = float(value)
            return [number] if math.isfinite(number) else []
        if isinstance(value, np.ndarray):
            return [float(v) for v in value.reshape(-1) if math.isfinite(float(v))]
        if isinstance(value, (list, tuple)):
            flattened: list[float] = []
            for item in value:
                flattened.extend(NemotronRuntime._flatten_numeric(item))
            return flattened

        # torch.Tensor and tensor-like values. Avoid importing torch here.
        try:
            detached = value.detach()  # type: ignore[attr-defined]
            cpu_value = detached.cpu()
            as_list = cpu_value.reshape(-1).tolist()
            return NemotronRuntime._flatten_numeric(as_list)
        except Exception:
            pass

        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
        return [number] if math.isfinite(number) else []

    @staticmethod
    def _mean_probability(values: object) -> float | None:
        numbers = [
            min(1.0, max(0.0, value))
            for value in NemotronRuntime._flatten_numeric(values)
            if 0.0 <= value <= 1.0
        ]
        if not numbers:
            return None
        return float(sum(numbers) / len(numbers))

    @staticmethod
    def _hypothesis_token_count(hypothesis: object) -> int:
        for attr in ("token_confidence", "tokens", "y_sequence"):
            value = getattr(hypothesis, attr, None)
            if value is None:
                continue
            try:
                size = int(value.numel())  # torch.Tensor
            except Exception:
                try:
                    size = len(value)
                except Exception:
                    size = 0
            if size > 0:
                return size
        text = str(getattr(hypothesis, "text", "") or "").strip()
        return max(1, len(text.split())) if text else 0

    @staticmethod
    def _confidence_from_hypothesis(hypothesis: object) -> tuple[float, str]:
        """Return utterance confidence in [0, 1] plus its derivation source.

        Priority is NeMo's native confidence outputs. NeMo Hypothesis can expose
        token_confidence / word_confidence / non_blank_frame_confidence when the
        decoder preserves them. If those are unavailable in the installed NeMo
        build, fall back to a length-normalized RNNT hypothesis score:

            confidence = exp(sequence_log_score / emitted_token_count)

        The score fallback is an approximation, not a calibrated probability.
        """
        if hypothesis is None:
            return 0.0, "unavailable"

        for attr, source in (
            ("token_confidence", "nemo_token_confidence"),
            ("word_confidence", "nemo_word_confidence"),
            ("non_blank_frame_confidence", "nemo_non_blank_frame_confidence"),
            ("frame_confidence", "nemo_frame_confidence"),
        ):
            try:
                values = getattr(hypothesis, attr, None)
            except Exception:
                values = None
            confidence = NemotronRuntime._mean_probability(values)
            if confidence is not None:
                return confidence, source

        score_values = NemotronRuntime._flatten_numeric(
            getattr(hypothesis, "score", None)
        )
        token_count = NemotronRuntime._hypothesis_token_count(hypothesis)
        if score_values and token_count > 0:
            score = score_values[0]
            if score < 0.0:
                # RNNT Hypothesis.score is normally a cumulative log score.
                # Length normalization turns it into a geometric-mean path
                # probability and avoids confidence collapsing with sentence length.
                normalized = math.exp(max(-80.0, score / token_count))
                return min(1.0, max(0.0, normalized)), "rnnt_score_geomean"

        return 0.0, "unavailable"

    def _run_one_step(self, state: _ModelStreamState, flush: bool) -> str | None:
        import torch

        self._update_feature_cache(state, final=flush)
        if state.feature_cache is None:
            return None
        features = state.feature_cache
        total_frames = int(features.size(-1))
        first = state.step_num == 0
        index = 0 if first else 1
        chunk_size = self._indexed(self.streaming_cfg.chunk_size, index)
        shift_size = self._indexed(self.streaming_cfg.shift_size, index)
        pre_cache = self._indexed(self.streaming_cfg.pre_encode_cache_size, index)

        end = state.feature_index + chunk_size
        if end > total_frames:
            if not flush or state.feature_index >= total_frames:
                return None
            end = total_frames

        chunk = features[:, :, state.feature_index : end]
        if pre_cache > 0:
            start = max(0, state.feature_index - pre_cache)
            chunk = torch.cat([features[:, :, start : state.feature_index], chunk], dim=-1)
        chunk = chunk.to(self.compute_dtype)
        chunk_length = torch.tensor([chunk.size(-1)], device=self.device)

        drop = 0 if first else int(self.streaming_cfg.drop_extra_pre_encoded)
        is_last = flush and end >= total_frames
        self._ensure_prompt(state.language)

        with torch.inference_mode():
            (
                state.pred_out_stream,
                transcribed,
                state.cache_last_channel,
                state.cache_last_time,
                state.cache_last_channel_len,
                state.previous_hypotheses,
            ) = self.model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_length,
                cache_last_channel=state.cache_last_channel,
                cache_last_time=state.cache_last_time,
                cache_last_channel_len=state.cache_last_channel_len,
                keep_all_outputs=is_last,
                previous_hypotheses=state.previous_hypotheses,
                previous_pred_out=state.pred_out_stream,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )

        hypothesis = transcribed[0]
        raw_text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
        clean_text, detected_lang = self._parse_transcript(raw_text, state.language)
        if self.beam_size > 1:
            # NeMo beam search seeds each chunk with only the last token of the
            # previous hypothesis, so hypothesis.text covers only the current
            # chunk's new tokens. Accumulate manually across chunks.
            if clean_text:
                sep = " " if state.accumulated_text else ""
                state.accumulated_text = state.accumulated_text + sep + clean_text
            state.last_text = state.accumulated_text
        else:
            state.last_text = clean_text
        state.detected_language = detected_lang
        if state.last_text:
            state.last_confidence, state.confidence_source = self._confidence_from_hypothesis(
                hypothesis
            )
        else:
            state.last_confidence = 0.0
            state.confidence_source = "unavailable"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ASR step result",
                extra={
                    "rpc_id": state.key.rpc_id,
                    "session_id": state.key.session_id,
                    "stream_id": state.key.stream_id,
                    "utterance_id": state.utterance_id,
                    "step_num": state.step_num,
                    "flush": flush,
                    "raw_asr": raw_text,
                    "asr_text": clean_text,
                    "detected_language": detected_lang,
                    "input_language": state.language,
                    "confidence": state.last_confidence,
                    "confidence_source": state.confidence_source,
                },
            )
        state.feature_index += max(1, shift_size)
        state.step_num += 1

        # Drop frames that pre_cache lookback no longer needs, keeping GPU memory
        # O(pre_cache + chunk_size) per stream instead of O(utterance_length).
        # Keep the pre-encode cache of the NEXT step: after step 0 (cache 0) step 1 still needs
        # pre_encode_cache_size[1] frames (docs/asr-feature-fix-design.md, bug 1).
        keep = pre_cache if self.legacy_features else self._indexed(self.streaming_cfg.pre_encode_cache_size, 1)
        trim = max(0, state.feature_index - keep)
        if trim > 0:
            state.feature_cache = state.feature_cache[:, :, trim:]
            state.feature_index -= trim

        return state.last_text
