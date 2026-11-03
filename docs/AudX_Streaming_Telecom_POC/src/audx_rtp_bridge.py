"""Timestamp-preserving RTP ingress adapter for AudXStream.

Apply after RTP jitter handling, packet-loss silence insertion, PCM decoding and
resampling to mono 16 kHz. Every push is per stream and in packet sequence.
Returns 0 or 1 contiguous 16-kHz float32 blocks with aligned RTP start time.
Never use RTP timestamp to order inferred ASR segments or outgoing TTS audio.
"""
from dataclasses import dataclass
import numpy as np
from audx_streaming import AudXStream, AudXCNNLSTM, SAMPLE_RATE


@dataclass(frozen=True)
class EnhancedAudioBlock:
    rtp_timestamp: int
    float_pcm_16k: np.ndarray
    duration_ms: float
    source_rate: int = SAMPLE_RATE


class AudXRtpBridge:
    def __init__(self, model: AudXCNNLSTM, *, device: str = 'cpu', rtp_clock_rate: int = 8000):
        if rtp_clock_rate <= 0: raise ValueError('rtp_clock_rate must be positive')
        self.clock_rate=rtp_clock_rate
        self.processor=AudXStream(model,device=device)
        self.anchor=None
        self._flushed=False

    def push(self,audio_float32_16k:np.ndarray,rtp_timestamp:int):
        if self._flushed: raise RuntimeError('bridge already flushed')
        if self.anchor is None: self.anchor=int(rtp_timestamp)
        output_start=self.processor.emitted_samples
        output=self.processor.push(audio_float32_16k)
        return self._result(output,output_start)

    def flush(self):
        if self._flushed: raise RuntimeError('bridge already flushed')
        self._flushed=True
        start=self.processor.emitted_samples
        audio=self.processor.flush()
        return self._result(audio,start)

    def _result(self,audio:np.ndarray, output_start:int):
        if audio.size==0: return None
        tick_offset=round(output_start*self.clock_rate/SAMPLE_RATE)
        return EnhancedAudioBlock(
            rtp_timestamp=((self.anchor or 0)+tick_offset)&0xffffffff,
            float_pcm_16k=audio,
            duration_ms=audio.size*1000.0/SAMPLE_RATE,
        )
