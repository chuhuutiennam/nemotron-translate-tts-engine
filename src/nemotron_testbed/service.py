from __future__ import annotations

import asyncio
import logging
import time

import grpc
import grpc.aio

from .coordinator import (
    CapacityError,
    DataLossError,
    EngineCoordinator,
    FeatureUnavailableError,
    RecognitionEnvelope,
    RpcFailure,
    _RPC_END,
)
from .domain import QueueFullError
from .metrics import (
    GRPC_BUFFER_WAIT_MS,
    GRPC_SEND_MS,
    SPEECH_END_TO_AUDIO_SENT_MS,
    SPEECH_END_TO_AUDIO_SENT_PREPROC_MS,
)
from .pb import speech_pb2, speech_pb2_grpc

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger(f"{__name__}.pipeline_trace")


_STATUS_BY_NAME = {
    "INVALID_ARGUMENT": grpc.StatusCode.INVALID_ARGUMENT,
    "FAILED_PRECONDITION": grpc.StatusCode.FAILED_PRECONDITION,
    "RESOURCE_EXHAUSTED": grpc.StatusCode.RESOURCE_EXHAUSTED,
    "DATA_LOSS": grpc.StatusCode.DATA_LOSS,
    "UNAVAILABLE": grpc.StatusCode.UNAVAILABLE,
    "INTERNAL": grpc.StatusCode.INTERNAL,
    "CANCELLED": grpc.StatusCode.CANCELLED,
}


def _call_info(context) -> tuple[str, str, float | None]:
    """Peer address, status code the server set and client deadline left in seconds; never raises. Peer is "-" once the deadline has expired; remaining is None without a deadline (or when unreadable), 0 once it expired."""
    try:
        peer = context.peer() or "-"
    except Exception:
        peer = "-"
    try:
        code = context.code()
        code_name = "None" if code is None else code.name
    except Exception:
        code_name = "unknown"
    try:
        remaining = context.time_remaining()
        remaining = None if remaining is None else round(max(0.0, float(remaining)), 3)
    except Exception:
        remaining = None
    return peer, code_name, remaining


class SpeechStreamService(speech_pb2_grpc.SpeechStreamServicer):
    def __init__(self, coordinator: EngineCoordinator) -> None:
        self.coordinator = coordinator

    async def Recognize(self, request_iterator, context):
        try:
            rpc = self.coordinator.open_rpc()
        except CapacityError as exc:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            return

        async def receive_requests() -> None:
            try:
                async for request in request_iterator:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "GRPC_IN "
                            "rpc=%s session=%s stream=%s "
                            "pcm_bytes=%s sample_rate=%s channels=%s "
                            "rtp_timestamp=%s rtp_clock_rate=%s "
                            "chunk_seq=%s duration_ms=%s "
                            "end_of_stream=%s language=%s language_source=%s task=%s",
                            rpc.rpc_id,
                            request.session_id,
                            request.stream_id,
                            len(request.pcm),
                            request.sample_rate,
                            request.channels,
                            request.rtp_timestamp,
                            request.rtp_clock_rate,
                            request.chunk_seq,
                            request.duration_ms,
                            request.end_of_stream,
                            request.language,
                            request.language_source,
                            request.task,
                        )

                    await self.coordinator.accept_chunk(rpc.rpc_id, request)
                await self.coordinator.close_input(rpc.rpc_id)
            except asyncio.CancelledError:
                raise
            except FeatureUnavailableError as exc:
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "FAILED_PRECONDITION", str(exc)
                )
            except DataLossError as exc:
                await self.coordinator.fail_rpc(rpc.rpc_id, "DATA_LOSS", str(exc))
            except ValueError as exc:
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "INVALID_ARGUMENT", str(exc)
                )
            except QueueFullError as exc:
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "RESOURCE_EXHAUSTED", str(exc)
                )
            except RuntimeError as exc:
                logger.error(
                    "Unexpected RuntimeError in receive task: %s",
                    exc,
                    extra={"rpc_id": rpc.rpc_id},
                )
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "INTERNAL", str(exc)
                )
            except grpc.aio.AioRpcError as exc:
                if context.cancelled() or exc.code() is grpc.StatusCode.CANCELLED:
                    return
                logger.warning(
                    "gRPC stream error in receive task: %s",
                    exc,
                    extra={"rpc_id": rpc.rpc_id},
                )
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "INTERNAL", "gRPC stream error"
                )
            except Exception as exc:
                # grpc._cython.cygrpc.AbortError("Locally aborted.") is raised
                # when context.abort() fires (e.g., idle timeout) while the
                # receive coroutine is blocked on the request iterator.
                # It is not an AioRpcError so the handler above misses it.
                # Treat it as a clean termination — the RPC is already closed.
                if type(exc).__name__ == "AbortError":
                    return
                logger.exception("Receive task failed", extra={"rpc_id": rpc.rpc_id})
                await self.coordinator.fail_rpc(
                    rpc.rpc_id, "INTERNAL", "server stream processing error"
                )

        receive_task = asyncio.create_task(
            receive_requests(), name=f"receive-{rpc.rpc_id}"
        )

        settings = self.coordinator.settings
        max_idle_s = settings.rpc_max_idle_s
        sent_log_enabled = settings.pipeline_trace
        sent_log_level = logging.getLevelName(settings.pipeline_trace_level)
        speech_end_utterance: dict[str, int] = {}
        first_audio_utterance: dict[str, int] = {}

        try:
            while True:
                if max_idle_s > 0:
                    try:
                        item = await asyncio.wait_for(
                            rpc.response_buffer.get(), timeout=max_idle_s
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "RPC idle for >%ss — cancelling stuck pipeline",
                            max_idle_s,
                            extra={"rpc_id": rpc.rpc_id},
                        )
                        await self.coordinator.cancel_rpc(
                            rpc.rpc_id,
                            f"pipeline idle for >{max_idle_s}s",
                        )
                        await context.abort(
                            grpc.StatusCode.CANCELLED,
                            f"pipeline idle for >{max_idle_s}s",
                        )
                        break
                else:
                    item = await rpc.response_buffer.get()
                if item is _RPC_END:
                    break
                if isinstance(item, RpcFailure):
                    await context.abort(
                        _STATUS_BY_NAME.get(item.code_name, grpc.StatusCode.INTERNAL),
                        item.details,
                    )
                    break
                if not isinstance(item, RecognitionEnvelope):
                    continue

                buffer_wait_ms = (time.monotonic() - item.created) * 1000.0
                GRPC_BUFFER_WAIT_MS.observe(buffer_wait_ms)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "GRPC_OUT "
                        "rpc=%s session=%s stream=%s "
                        "seq=%s final=%s "
                        "ts_start=%s ts_end=%s "
                        "confidence=%.4f language=%s "
                        "text=%s "
                        "audio_bytes=%s audio_pt=%s "
                        "buffer_wait_ms=%.1f",
                        rpc.rpc_id,
                        item.session_id,
                        item.stream_id,
                        item.seq,
                        item.is_final,
                        item.ts_start,
                        item.ts_end,
                        item.confidence,
                        item.language,
                        item.text,
                        len(item.audio_payload),
                        item.audio_pt,
                        buffer_wait_ms,
                    )

                yield_t = time.monotonic()
                yield speech_pb2.RecognitionResult(
                    session_id=item.session_id,
                    stream_id=item.stream_id,
                    text=item.text,
                    is_final=item.is_final,
                    ts_start=item.ts_start,
                    ts_end=item.ts_end,
                    confidence=item.confidence,
                    language=item.language,
                    seq=item.seq,
                    audio_payload=item.audio_payload,
                    audio_pt=item.audio_pt,
                )
                # The generator resumes only after grpc finished writing the message, so
                # this approximates the time the transport took to accept it (a slow
                # reader / HTTP/2 flow control shows up here, not in buffer_wait_ms).
                sent_t = time.monotonic()
                send_ms = (sent_t - yield_t) * 1000.0
                GRPC_SEND_MS.observe(send_ms)
                first_audio = (
                    bool(item.audio_payload)
                    and first_audio_utterance.get(item.stream_id) != item.utterance_id
                )
                if first_audio:
                    first_audio_utterance[item.stream_id] = item.utterance_id
                speech_end_ms: float | None = None
                speech_end_preproc_ms: float | None = None
                if (
                    item.speech_end_wall > 0
                    and item.audio_payload
                    and speech_end_utterance.get(item.stream_id) != item.utterance_id
                ):
                    speech_end_utterance[item.stream_id] = item.utterance_id
                    speech_end_ms = (sent_t - item.speech_end_wall) * 1000.0
                    SPEECH_END_TO_AUDIO_SENT_MS.observe(speech_end_ms)
                    if item.preproc_delay_s > 0:
                        speech_end_preproc_ms = speech_end_ms + item.preproc_delay_s * 1000.0
                        SPEECH_END_TO_AUDIO_SENT_PREPROC_MS.observe(speech_end_preproc_ms)
                if sent_log_enabled and trace_logger.isEnabledFor(sent_log_level):
                    trace_logger.log(
                        sent_log_level,
                        "GRPC_SENT rpc=%s session=%s stream=%s seq=%s final=%s "
                        "ts_start=%s ts_end=%s text_len=%s audio_bytes=%s send_ms=%.1f%s",
                        rpc.rpc_id,
                        item.session_id,
                        item.stream_id,
                        item.seq,
                        item.is_final,
                        item.ts_start,
                        item.ts_end,
                        len(item.text),
                        len(item.audio_payload),
                        send_ms,
                        (
                            ""
                            if speech_end_ms is None
                            else f" speech_end_to_sent_ms={speech_end_ms:.1f}"
                        )
                        + (
                            ""
                            if speech_end_preproc_ms is None
                            else f" speech_end_to_sent_preproc_ms={speech_end_preproc_ms:.1f}"
                        ),
                        extra={
                            "stage": "GRPC_SENT",
                            "rpc_id": rpc.rpc_id,
                            "session_id": item.session_id,
                            "stream_id": item.stream_id,
                            "utterance_id": item.utterance_id,
                            "seq": item.seq,
                            "is_final": item.is_final,
                            "text_len": len(item.text),
                            "audio_bytes": len(item.audio_payload),
                            "send_ms": round(send_ms, 1),
                            "buffer_wait_ms": round(buffer_wait_ms, 1),
                            "first_audio": first_audio,
                            "speech_end_to_sent_ms": (
                                None if speech_end_ms is None else round(speech_end_ms, 1)
                            ),
                            "speech_end_to_sent_preproc_ms": (
                                None
                                if speech_end_preproc_ms is None
                                else round(speech_end_preproc_ms, 1)
                            ),
                        },
                    )
        except asyncio.CancelledError:
            raise
        finally:
            if not receive_task.done():
                receive_task.cancel()

            try:
                await receive_task
            except asyncio.CancelledError:
                pass

            # Normal RPC completion is already handled by Coordinator:
            # input_closed + no active streams + no pending postprocess
            # -> response_buffer.close()
            # -> rpc.finished = True
            #
            # Only cancel when the gRPC handler ended prematurely.
            if not rpc.finished and not rpc.cancelled:
                peer, code_name, time_remaining_s = _call_info(context)
                logger.info(
                    "gRPC call ended early: peer=%s grpc_code=%s time_remaining_s=%s",
                    peer,
                    code_name,
                    time_remaining_s,
                    extra={
                        "rpc_id": rpc.rpc_id,
                        "stage": "GRPC_CALL_ENDED",
                        "peer": peer,
                        "grpc_code": code_name,
                        "time_remaining_s": time_remaining_s,
                    },
                )
                await self.coordinator.cancel_rpc(
                    rpc.rpc_id,
                    "gRPC call ended before engine pipeline completed",
                )
