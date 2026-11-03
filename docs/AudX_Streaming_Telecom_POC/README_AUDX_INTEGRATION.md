# AudX trained checkpoint — Streaming proof of concept for telecom ASR

## Scope and evidence

- **Input artifacts verified:** `noise.ipynb` and `audx_model_full.zip` supplied by the user.
- **Checkpoint:** serialized full `models.cnn_lstm.CNNLSTM` module (Torch), ~37.1 MB, **9,282,753 parameters**.
- **Actual layers (restricted deserialization inspection):** Conv2D `1→16`, kernel `3×3`, padding `1`; ReLU; Conv2D `16→32`, kernel `3×3`, padding `1`; ReLU; LSTM `input_size=8224`, `hidden=256`, `num_layers=2`, **bidirectional=False**, `batch_first=True`; Linear `256→257`.
- **Notebook's audio specification:** 16,000 Hz mono, STFT `n_fft=512` (32 ms), `hop_length=256` (16 ms), magnitude regression + original noisy phase for reconstruction.
- **Notebook's 824-item VoiceBank-DEMAND evaluation:** STOI `0.9210→0.9258`, SDR `8.4461→16.2317`, SI-SDR `8.4451→16.2384`. These are user notebook outputs, **not independently repeated** here.
- **Notebook's 7 WAV inference records:** completed; **no telecom WER, no GPU streaming RTF** measured in the notebook.

## Central finding: a streaming path is feasible without retraining

The LSTM is **unidirectional** and can preserve `(h, c)` per call. Two symmetric temporal `3×3` CNN layers require precisely **two right-context STFT frames** (32 ms) and two left-context frames. With centered 512-point STFT and iSTFT overlap-add at 16 kHz, the earliest first output requires ~64 ms of audio, before any packetization/inference/queue delay. The implementation retains per-stream STFT samples, the last two raw spectrogram frames, recurrent LSTM h/c and iSTFT overlap-add weights. **The streaming state is per RTP media leg, not per VAD utterance.**

This is **not** a native streaming feature already documented in the uploaded AudX notebook. It is an adapter reconstructed from the checkpoint and validated below. Inference compatibility with the original GitHub helper `utils.audio.to_spectrogram/to_waveform` still needs confirmation on the environment that has cloned the repo.

## Files

| File | Role |
|---|---|
| `model/audx_weights_safe.pth` | Weight-only checkpoint converted from full pickle via `weights_only=True` allowlist |
| `src/audx_streaming.py` | Stateful STFT → CNN window → LSTM cache → iSTFT; `AudXStream.push/flush` |
| `src/audx_rtp_bridge.py` | Preserves inbound RTP timestamp anchor after fixed-latency enhancement |
| `scripts/convert_checkpoint.py` | Convert original full-model archive without `weights_only=False` |
| `scripts/benchmark_audx.py` | RTF/latency smoke benchmark on generated synthetic waveform |
| `tests/` | Model and waveform parity, multi-stream separation, packet accounting, RTP timestamp wraparound |
| `etc/audx-engine-fragment.yaml` | Desired config contract, **not yet wired to Settings** |

## Installation and test

Requires PyTorch, NumPy, Python 3.11/3.12 (compatible Torch runtime); no librosa or model-train dependencies needed for this particular adapter. `torch` must be installed with an appropriate CPU/CUDA build in the target environment.

```bash
export PYTHONPATH=src
python -m unittest discover -s tests -v
python scripts/benchmark_audx.py --seconds 10 --packet-ms 80 --device cpu
```

Convert original full-model checkpoint if needed:

```bash
PYTHONPATH=src python scripts/convert_checkpoint.py \
  --input /path/to/audx_model_full.zip \
  --output model/audx_weights_safe.pth
```

### Example

```python
import numpy as np
from audx_streaming import load_safe_weights, AudXStream
model = load_safe_weights('model/audx_weights_safe.pth', 'cpu')  # once
stream = AudXStream(model, 'cpu')  # once per media leg
chunk = np.zeros(1280, dtype=np.float32)  # 80 ms at 16 kHz
processed = stream.push(chunk)  # may return zero or fewer/more samples
last = stream.flush()  # only on media leg close
```

**Do not** assume `push` returns the same number of samples as its input chunk; outputs are delayed until the appropriate future frames are available. Never reset on VAD END. Be sure to map output RTP timestamps from the cumulative number of **emitted** samples and correctly flush on stream close.

## Local CPU smoke measurements (not production/telecom figures)

Trained checkpoint; synthetic 10-second 16-kHz signal, CPU Torch `2.10.0+cpu`, 2 inference threads; no live RTP, network, GPU, ASR or translation:

| RTP-like input chunk | Compute RTF | p95 process() wall time | Max absolute waveform diff from internal full-file reference |
|---|---:|---:|---:|
| 20 ms | 0.6067 | 14.90 ms | 1.27e-08 |
| 80 ms | 0.1758 | 15.98 ms | 1.05e-08 |
| 160 ms | 0.0897 | 17.64 ms | 1.05e-08 |

RTF includes synthetic streaming loop overhead and flush, excludes audio capture pacing. Values are machine-dependent, not portable SLAs. They demonstrate **why inferencing each small 20-ms packet is wasteful even for a true stateful backend**. 80-ms chunks are the first configuration to test in the actual engine.

## Mandatory parity check against ORIGINAL notebook before adoption

Run this **inside Kaggle** after the original notebook has imported `from utils.audio import ...`, loaded the real checkpoint, and defined `enhance_waveform(noisy)`:

```python
# Put src/audx_streaming.py beside the notebook or on PYTHONPATH.
from audx_streaming import offline_reference, AudXStream
import numpy as np

x = load_audio('/path/to/a_test_audio.wav')  # original utils.audio.load_audio
expected = enhance_waveform(x)               # notebook's original inference
ours = offline_reference(model, x, device=str(DEVICE))
print('len original/ours:', len(expected), len(ours))
L = min(len(expected), len(ours))
print('full-file parity MAE:', np.mean(np.abs(expected[:L] - ours[:L])))
print('full-file parity max abs:', np.max(np.abs(expected[:L] - ours[:L])))
# Do not deploy if preprocessing/reconstruction semantics differ significantly.

sess = AudXStream(model, device=str(DEVICE))
parts = []
for p in range(0, len(x), 1280):
    parts.append(sess.push(x[p:p+1280]))
parts.append(sess.flush())
y = np.concatenate(parts)
print('stream/full reference diff:', np.max(np.abs(y - ours)))
assert len(y) == len(x)
```

If the original helper normalizes the magnitude, pads with a different policy, resamples or clips differently, update the streaming DSP before considering the implementation equivalent.

## Engine v0.3.3/QUAN insertion point

Current source (uploaded previously) already calls `decode_and_resample()` and `_apply_rtp_policy()` then `_process_audio()` in `EngineCoordinator.push_audio`. **Attach the enhancer AFTER RTP gap repair and BEFORE `_process_audio()`**, so VAD and ASR receive the same enhanced PCM. Store `AudXRtpBridge` in `StreamContext`, one instance per stream, and keep the shared immutable model in `EngineCoordinator`.

Required integration behavior:

1. `audio_enhancement.mode=off`: preserve byte-level existing behavior and avoid importing/loading AudX.
2. `mode=shadow`: run separately so enhanced audio is measured but **does not delay or replace** ASR input. A synchronous call in the main event loop is **not** valid shadow mode.
3. `mode=on`: `AudXRtpBridge.push()` produces zero or one enhanced block; call `_process_audio()` with `block.float_pcm_16k`, `block.rtp_timestamp`, `block.source_rate=16000`, `source_samples=len(block.float_pcm_16k)`, `duration_ms=block.duration_ms`. Await an executor/dedicated processing worker rather than synchronously blocking the asyncio event loop. Maintain input ordering and keep bounded queue/backpressure.
4. `RTP` time continuity: bridge anchors output start to first received RTP timestamp and maps by cumulative enhanced sample count; RTP clock may be 8 kHz even if ASR audio is 16 kHz. The original engine's source timestamp continuity logic remains the ingress owner.
5. Ensure `END_OF_STREAM` and `close_input()` flush outstanding enhancement frames **before** finalizing ASR. VAD END must **not** flush/reset the enhancer. For a lengthy media stream, keep the recurrent hidden state across utterances.
6. On runtime/model error, don't silently replace a delayed enhanced block with current raw packet (which would shift timestamps). Use an aligned bypass buffer and documented failover, or fail RPC until safe fallback is implemented.
7. Measure enhancer stage delay, queue wait, CPU/GPU memory, VAD trigger drift, and E2E latency; compare raw and enhanced output under identical ASR model/settings.

This package intentionally does **not** edit the production `coordinator.py`, `config.py`, or gRPC proto. Full production integration still requires asynchronous queueing, cancellation/flush handling, robust resampling for non-16-kHz sources, and regression tests under concurrent RTP load.

## Acceptance before production

- Verified parity with notebook original enhancement utilities on real 16-kHz WAV.
- Continuous stateful processing across arbitrary packets and multiple utterances, no missing/duplicated samples.
- RTF and p95/p99 within the input packet interval **at target call concurrency**, without rising backlog.
- No decrease in Vietnamese/English ASR WER/CER on representative telecom audio.
- VAD onset/offset and EOU are not adversely shifted by speech suppression.
- Backpressure/fallback and media-leg close behavior verified in actual engine.
- Checkpoint version, model training provenance and dataset license verified for intended deployment.

**Status: CPU parity-tested POC. Not yet an integration into the live ASR engine or an end-to-end telecom quality validation.**
