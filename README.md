# Nemotron Streaming ASR + EnViT5 + Piper TTS Engine

Production-oriented testbed for the bidirectional gRPC contract in `proto/speech.proto`.

The engine supports two tasks per logical stream:

- `task=transcribe`: Nemotron streaming ASR -> partial/final text.
- `task=translate`: Nemotron streaming ASR -> final EnViT5 translation -> Piper TTS -> raw G.711 payload.

## Documentation

| Document | Content |
|---|---|
| [docs/system-design.md](docs/system-design.md) | Full system design: modules, pipeline, config, deployment |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Runtime pool ownership and sequence diagrams |
| [docs/models-and-pipeline.md](docs/models-and-pipeline.md) | Models and pipeline per task |
| [docs/pipeline-trace-guide.md](docs/pipeline-trace-guide.md) | Debugging a call with pipeline trace logs |
| [docs/scripts-guide.md](docs/scripts-guide.md) | Every script in `scripts/` |
| [docs/build-guide.md](docs/build-guide.md) | Docker build (base and app image) |
| [docs/server-setup-guide.md](docs/server-setup-guide.md) | Bare-metal GPU server setup and update |
| [docs/design-notes/resemble-denoiser-integration-design.md](docs/design-notes/resemble-denoiser-integration-design.md) | Optional audio preprocessing (resemble-enhance): design, latency model, rollout gates |
| [docs/audx-stateful-design.md](docs/audx-stateful-design.md) | Optional audio preprocessing model `audx` (stateful AudX noise reduction with AGC; not the resemble-enhance `denoiser` mode): design and measurements (Vietnamese) |
| [docs/telecom_realtime_utterance_segmentation_design.md](docs/telecom_realtime_utterance_segmentation_design.md) | Utterance/sentence segmentation design (`segmentation.mode`) |
| [docs/design-notes/](docs/design-notes/) | Earlier design and research notes, metrics definitions, incident analysis |

## Important semantics

### `task=transcribe`

`RecognitionResult.text` is the source-language transcript. Partial and final results are emitted. `audio_payload` is empty.
`RecognitionResult.confidence` is the ASR utterance confidence in `[0,1]`. The runtime prefers native NeMo token/word/non-blank-frame confidence and otherwise uses a length-normalized RNNT hypothesis-score fallback.

### `task=translate`

The first implementation intentionally performs translation and TTS on **ASR final utterances**, not unstable ASR partials. This avoids speaking text that later needs rollback.

By default no source ASR partial is emitted for translate mode. The final result contains:

- `text`: translated text.
- `language`: target language (the one requested in `AudioChunk.language`).
- `audio_payload`: raw G.711 bytes with no RTP header.
- `audio_pt`: `0` for PCMU or `8` for PCMA.

The Gateway remains responsible for RTP packetization, sequence numbers, SSRC, RTP timestamps, pacing, and egress to MF.

Because proto3 `uint32 audio_pt` defaults to `0`, Gateway must use `len(audio_payload) > 0` to decide whether audio exists. `audio_pt == 0` is a valid PCMU payload type.

## Language routing

For `task=translate` / `realtime_translation`, `AudioChunk.language` is the **target** language (`vi` or `en`, regional variants accepted). The ASR always runs in `auto` mode and the source language is detected per utterance:

```text
language=en, speaker says Vietnamese -> vi-VN -> en-US (Piper lessac)
language=vi, speaker says English    -> en-US -> vi-VN (Piper vais1000)
```

When the gateway also sends `AudioChunk.language_source` (`vi` or `en`) and `asr.use_language_source` is on (`ASR_USE_LANGUAGE_SOURCE=true`; the shipped configs turn it on, the code default is off), the ASR runs with that fixed source instead of `auto`, and a source equal to the target is passed through from the first utterance. An empty value keeps `auto`; a value other than vi/en is ignored with a warning (`auto`). For `task=transcribe` it overrides `language`. It must not change within a stream.

Rules:

- A target other than vi/en is rejected when the stream opens. The target must not change within a stream.
- If the ASR detects neither vi nor en, the stream's last detected vi/en language is used. Without one, the utterance is not translated.
- Not translated means passthrough: the ASR text and the original audio (G.711, same codec as TTS output) are returned, with no translation and no TTS. Reasons (`PASSTHROUGH` trace, `translate_passthrough_total{reason}`): `source_equals_target` (detected source is already the target), `no_target_language` (empty `language`), `language_undetected`.
- Empty ASR text returns nothing.
- The original audio is returned after the utterance ends plus the FINAL latency, in order with translated utterances.

`task=transcribe` is unchanged: `language` is the source (empty = auto).

## Async I/O + runtime-pool architecture

```text
grpc.aio / asyncio event loop
    |
    +-- EngineCoordinator (single owner of RPC/stream lifecycle)
    |
    +-- NemotronRuntimePool
    |      +-- ASR worker 0 -> private Nemotron model + stream caches
    |      +-- ASR worker 1 -> private Nemotron model + stream caches
    |      `-- ...
    |
    +-- TranslationRuntimePool
    |      +-- worker 0 -> private CTranslate2 Translator/tokenizer
    |      `-- ...
    |
    `-- TtsRuntimePool
           +-- worker 0 -> private Piper EN/VI voices
           `-- ...
```

Defaults remain conservative: `ASR_WORKERS=1`, `TRANSLATION_WORKERS=1`,
`TTS_WORKERS=1`. No gRPC task calls a model directly.

ASR is stateful, so the pool enforces **stream affinity**: after `START`, all
`STEP`, `FINALIZE`, and `CANCEL` commands for that active utterance always go
to the same Nemotron worker. Affinity is released only after `FINAL` or
`CANCELLED`. Translation and TTS jobs are utterance-level and may be sent to
any free worker.

Each additional worker owns another model instance. Increase worker counts only
after measuring RAM/VRAM and CPU contention. In particular, `ASR_WORKERS>1`
replicates the Nemotron model.

The critical rule for translate mode is:

```text
ASR FINAL
  -> enqueue TranslationJob
  -> immediately release/reset ASR utterance
  -> next utterance may start

Translation/TTS continue in background.
```

Therefore Piper or EnViT5 latency does not keep the ASR stream in `FINALIZING` and does not extend the ASR deferred-audio window.

## Pipeline sequence

```text
AudioChunk
  -> decode PCM S16LE
  -> resample 16 kHz
  -> RTP continuity policy
  -> VAD endpointing
  -> Nemotron cache-aware ASR

transcribe:
  -> PARTIAL RecognitionResult
  -> FINAL RecognitionResult

translate:
  -> internal ASR FINAL
  -> EnViT5 CTranslate2
  -> translated text
  -> Piper target-language voice
  -> resample 8 kHz
  -> G.711 PCMU/PCMA
  -> FINAL RecognitionResult(text + audio_payload)
```

## Per-stream isolation

A logical stream is keyed by:

```text
rpc_id + session_id + stream_id
```

MO and MT therefore have independent:

- VAD state.
- pre-roll.
- PCM backlog.
- RTP continuity state.
- Nemotron encoder/RNNT state.
- utterance IDs.
- translation/TTS ordering.

`seq` remains monotonically increasing per RPC/session and is assigned when a result is actually emitted.

## Output ordering

Translation/TTS results are committed in utterance order per stream. If future runtime parallelism causes utterance `N+1` to finish before `N`, `N+1` waits in the coordinator until `N` is emitted.

## RTP continuity

The engine now distinguishes forward gaps from late/out-of-order chunks.

Defaults:

```text
RTP_GAP_POLICY=insert_silence
RTP_MAX_GAP_MS=200
RTP_OUT_OF_ORDER_POLICY=drop
```

For a small forward timestamp gap the engine inserts equivalent silence before ASR so the acoustic timeline is not compressed. Large gaps are logged. Out-of-order chunks are dropped by default because appending them after newer PCM would corrupt the audio order.

For exact RTP packet-loss diagnostics, Gateway should eventually add a monotonically increasing chunk sequence or RTP sequence metadata to the protobuf contract.

## Models expected locally

Example layout:

```text
models/
├── nemotron-3.5-asr-streaming-0.6b/
│   └── nemotron-3.5-asr-streaming-0.6b.nemo
├── envit5_ctranslate2/
│   ├── model.bin
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── spiece.model
│   └── ...
└── piper/
    ├── en_US-lessac-medium.onnx
    ├── en_US-lessac-medium.onnx.json
    ├── vi_VN-vais1000-medium.onnx
    └── vi_VN-vais1000-medium.onnx.json
```

If the tokenizer files are already inside the CTranslate2 model directory, `TRANSLATION_TOKENIZER_PATH` can point to the same directory.

Optional audio preprocessing keeps its assets in `models/resemble-enhance/` (`denoiser.onnx`, `enhancer_stage2/`, `resemble-enhance/`); see [Audio preprocessing](#audio-preprocessing-optional).

## Offline Python dependencies

In addition to the working Nemotron/NeMo wheelhouse, add wheels for:

```text
ctranslate2
transformers
sentencepiece
piper-tts
onnxruntime       # normally pulled by Piper
```

The exact versions should be frozen from the VDI environment that passes the probes. Do not re-introduce `grpcio-tools` into the runtime image; generated protobuf files are already included.

## Configure

```bash
cp .env.example .env
vi .env
set -a
source .env
set +a
```

For a transcribe-only deployment:

```bash
TRANSLATION_ENABLED=false
TTS_ENABLED=false
```

For translate mode both should be enabled.


## Audio preprocessing (optional)

Optional streaming noise reduction with [`resemble-enhance`](https://github.com/resemble-ai/resemble-enhance) or a user-trained AudX CNN-LSTM (`model: audx`), applied to each stream after the RTP policy and before VAD/ASR. It is **off by default** in code and in the ConfigMaps: with `audio_preprocessing.enabled: false` the pipeline is unchanged (covered by a test). The shipped `etc/nemotron-engine.yaml` currently turns on `model: audx` for every stream (trial since 2026-10-30; Vietnamese WER not measured yet).

| Key (ENV) | Default | Meaning |
|---|---|---|
| `enabled` (`AUDIO_PREPROCESSING_ENABLED`) | `false` | Master switch |
| `model` (`AUDIO_PREPROCESSING_MODEL`) | `resemble` | `resemble` (resemble-enhance, block `resemble:`) or `audx` (AudX with its LSTM state kept per stream, block `audx:`, D = 244 ms). Never chained; each model has its own `window`, so switching the model needs no other change |
| `resemble.mode` (`AUDIO_PREPROCESSING_MODE`) | `denoiser` | `denoiser` (ONNX, noise reduction only) or `enhancer` (generative: can change words, much heavier) |
| `devices` (`AUDIO_PREPROCESSING_DEVICES`) | `cpu` in code, `cuda:N` in the YAML/ConfigMaps | Worker i runs on `devices[i % len]`; CPU cannot keep up with realtime audio for resemble-enhance. There is no single `device` key (removed; a stale one is reported as `CONFIG_DEPRECATED`) |
| `resemble.assets_dir` (`AUDIO_PREPROCESSING_ASSETS_DIR`) | `models/resemble-enhance` | resemble-enhance weights (modes `denoiser` / `enhancer`); nothing is downloaded at runtime |
| `sample_percent` (`AUDIO_PREPROCESSING_SAMPLE_PERCENT`) | `100` | Random share of streams that are preprocessed (canary) |
| `max_streams` (`AUDIO_PREPROCESSING_MAX_STREAMS`) | `0` | Cap on concurrently preprocessed streams (0 = unlimited); extra streams use the raw flow |
| `resemble.window.compute_budget_ms` / `audx.window.compute_budget_ms` (`AUDIO_PREPROCESSING_COMPUTE_BUDGET_MS` / `AUDIO_PREPROCESSING_AUDX_COMPUTE_BUDGET_MS`) | per mode (shipped yaml and GPU ConfigMaps: 200 for `denoiser`, 100 for `audx`) | Set to about 1.5x the p99 of `audio_preproc_window_seconds` |
| `vad_input` (`AUDIO_PREPROCESSING_VAD_INPUT`) | `enhanced` | `raw` lets VAD see the original audio |
| `audx.checkpoint` (`AUDIO_PREPROCESSING_AUDX_CHECKPOINT`) | `models/audx/audx_weights_safe.pth` | Model `audx` only: AudX state_dict (bare or the POC's `{'state_dict': ...}`), loaded with `weights_only` |
| `audx.agc_target_dbfs` / `audx.agc_max_gain_db` | `-28` / `30` | Model `audx` only: boost-only AGC before the model, divided back out after it (0 dB = off); AudX damages speech at -38 dBFS and below without it |

The remaining keys (`zero_run_min_samples`, `fallback.*`, `workers`, `queue_max`, the resemble-enhance block `resemble.*` with `window.*`, `enhancer.*` and `dither.*`, and the AudX block `audx.*` with `window.*`) are documented next to their defaults in `etc/nemotron-engine.yaml`. The old paths `assets_dir`, `enhancer.*`, `dither.*` directly under `audio_preprocessing` are still read; the old `mode` and shared `window` are not read any more. Both cases log `CONFIG_DEPRECATED` at start.

Behaviour to know before enabling:

- **Constant added latency D** = `hop_ms + right_context_ms + compute_budget_ms`: 1.5 s for `denoiser` with the mode defaults (1000/500/500/500 ms left/hop/right/budget), 1.2 s with `compute_budget_ms: 200` (GPU ConfigMaps), 4.5 s for `enhancer`, and 244 ms for `audx` (hop 80 + model look-ahead 64 + budget 100 ms). It is added to VAD START, first partial, final and first audio. Timestamps keep their formulas, so there is a constant offset D between `ts_start/ts_end` and the audio content.
- **No call is lost to the denoiser:** a region whose result is late, whose queue is full or whose model fails plays the raw audio instead; after `fallback.consecutive_bad_regions` bad regions in a row the stream runs raw for `fallback.bypass_seconds`. If the pool fails to start the engine runs without preprocessing (`AUDIO_PREPROCESSING_DISABLED` in the log).
- **Digital silence:** exact-zero runs get Gaussian dither (`resemble.dither.sigma_lsb`) before the model and are forced back to zero after it, because the model otherwise removes onsets after digital zero (`audx`: no dither, the zero runs are only forced to zero).
- **`audx` is stateful:** at most one job per stream is in flight and it carries the stream's model state, so jobs run in order on any worker; a late job still completes to keep the state, a failed one restarts it (`audio_preproc_state_resets_total`).
- **End of stream:** the held-back D samples are flushed into ASR, so ASR receives `total_in + D` samples.
- Translation passthrough (`source_audio`) always uses the raw audio aligned with the processed stream.
- The models are English-only, non-causal and unmaintained; the benefit on Vietnamese or G.711 calls is not proven. Enable per canary (`sample_percent`) and compare WER with `scripts/client_wav.py --profile rtpgw` with the flag on and off.

Get the assets once on a machine with internet and copy them next to the other models (the Docker image copies `models/` as a whole; bare-metal uses `scripts/copy-models.sh`, which copies `resemble-enhance/` when the source model store has it):

```bash
python scripts/kaggle/enhance_wavs.py fetch --assets-dir models/resemble-enhance            # denoiser + enhancer
python scripts/kaggle/enhance_wavs.py fetch --assets-dir models/resemble-enhance --what onnx # denoiser only
```

Deployment and first-measurement checklist: [docs/server-setup-guide.md](docs/server-setup-guide.md), section B.6 ("Bật tiền xử lý audio", in Vietnamese). GPU timings have not been measured yet; the first deployment is meant to produce them.

## Worker-pool tuning

Start with:

```bash
ASR_WORKERS=1
TRANSLATION_WORKERS=1
TTS_WORKERS=1
```

Then scale one bottleneck at a time. Translation is the easiest stage to scale.
TTS workers are also safe because every worker loads private Piper voice
instances. ASR workers require affinity and a full model replica per worker;
the pool implements the affinity, but capacity still depends on available
RAM/VRAM and native compute threads.

Queue sizes are **per worker** in this revision. For example,
`TRANSLATION_WORKERS=2` and `TRANSLATION_QUEUE_SIZE=64` provide up to 128 queued
translation jobs across the pool.

On CPU, avoid oversubscription. `CPU_THREADS` controls PyTorch intra-op threads,
while `TRANSLATION_INTRA_THREADS` controls each CTranslate2 worker. More Python
workers multiplied by large native thread counts can reduce throughput instead
of improving it.

## File logging

The engine logs JSON to both stdout and a daily-rotating file when `LOG_FILE` is set:

```bash
LOG_LEVEL=INFO
LOG_FILE=/app/logs/nemotron-engine.log
LOG_BACKUP_COUNT=10        # days of history kept
```

The file rotates at local midnight; previous days are kept as
`nemotron-engine.log.YYYY-MM-DD`. There is no size cap, so avoid running DEBUG
for long periods.

For Docker/Kubernetes, mount `/app/logs` to persistent storage if file logs must
survive container or Pod replacement. Stdout logging remains enabled for the
normal container log collector.

### Pipeline trace (debugging a call stage by stage)

With `logging.pipeline_trace: true` (`LOG_PIPELINE_TRACE`, default) the engine
writes one line per stage of every utterance at the level set by
`logging.pipeline_trace_level` (`LOG_PIPELINE_TRACE_LEVEL`, default `DEBUG`;
independent of `logging.level`, so traces reach the file with `logging.level:
INFO` while `VAD step`, `GRPC_IN`/`GRPC_OUT` and other DEBUG lines still need
`logging.level: DEBUG`) — `ASR_FINAL`, `TRANSLATE`,
`TTS_PHRASE`, `EMIT` — each with `rpc_id`, `session_id`, `stream_id` and
`utterance_id`, plus a WARNING `DEGRADE` line (with `reason`) whenever an
utterance falls back to text-only. Set `LOG_PIPELINE_TRACE_TEXT=false` to log
only text lengths instead of the spoken content.

```bash
python scripts/trace_stream.py logs/nemotron-engine.log --stream-id MO
python scripts/trace_stream.py logs/nemotron-engine.log --session-id call-1 --only-problems
```

The script prints a per-utterance timeline and where it stopped
(`INCOMPLETE: no TRANSLATE (stopped after ASR_FINAL)`, `DEGRADED (translate_empty)`,
`EMPTY_ASR`, `NO_AUDIO`, ...). Trace lines go to the log file only; the console
stays at `CONSOLE_LOG_LEVEL`.

To judge whether partial ASR text is stable enough to start translation before
the utterance ends, set `LOG_PIPELINE_TRACE_PARTIAL=true` temporarily (adds an
`ASR_PARTIAL` line per ASR step, about 2 lines/s/stream) and replay the log:

```bash
python scripts/check_segmentation.py logs/nemotron-engine.log --session-id call-1 --verbose
python scripts/check_segmentation.py logs/nemotron-engine.log --sweep
```

It reports premature-commit rate, revision rate and time gained versus
`ASR_FINAL` for the Stable Punctuation Sentence Segmentation rule. It only
simulates; engine behavior is unchanged. The sweep compares `stable_n`/`stable_ms`
and `min_tokens` combinations. See `docs/pipeline-trace-guide.md` §6.

The same segmenter can run live in **shadow mode** (`segmentation.mode: shadow`,
`SEGMENTATION_MODE`, default `off`): it logs `SEGMENT_COMMIT` /
`SEGMENT_REVISED` / `SEGMENT_SUMMARY` / `SEGMENT_ALIGN` trace lines and updates the
`segment_*` Prometheus metrics, but never changes the output (no early
translation, no audio cut). See `docs/pipeline-trace-guide.md` §6b.

With `segmentation.mode: on` the engine translates, synthesizes and emits each
committed sentence as soon as it is stable (each segment's last message has
`is_final=True`; the FINAL remainder is the last segment). It only applies to
`realtime_translation` with a known source language. Default is `off`: enable it
only after validating shadow data (`--sweep`, `off` vs `on` comparison). Changing
the mode needs a restart; set `off` or `shadow` to roll back.

When ASR later revises words that were already committed (`I'm` -> `I am`,
`twenty five` -> `25`), the remainder is cut by word alignment against the
committed text (`segmenter.committed_end`), so no word is repeated or lost; a
committed segment is never re-sent. Each such cut logs `SEGMENT_ALIGN` and counts
`segment_align_total{how,phase}`.

## Scripts

Every script in `scripts/` (log analysis, test clients, probes, deploy/build
helpers) is described in [docs/scripts-guide.md](docs/scripts-guide.md): when to
use it, arguments, exit codes and how to read the output. `scripts/kaggle/` runs
the engine on a Kaggle GPU notebook and compares WER between configurations
(`language_source`, VAD hysteresis); see section 8 of the guide.

## Probe each model independently

ASR:

```bash
PYTHONPATH=src python scripts/probe_model.py "$MODEL_PATH" \
  --device cpu --language vi-VN --lookahead 6
```

Translation:

```bash
PYTHONPATH=src python scripts/probe_translation.py \
  "$TRANSLATION_MODEL_PATH" "$TRANSLATION_TOKENIZER_PATH" \
  --source vi-VN --target en-US \
  --text 'Xin chào, tôi muốn kiểm tra tài khoản.'
```

TTS:

```bash
PYTHONPATH=src python scripts/probe_tts.py \
  --en-model "$TTS_EN_MODEL_PATH" \
  --en-config "$TTS_EN_CONFIG_PATH" \
  --vi-model "$TTS_VI_MODEL_PATH" \
  --vi-config "$TTS_VI_CONFIG_PATH" \
  --language en-US --pt 0 \
  --text 'Hello, I would like to check my account.'
```

`probe_tts.py` writes raw G.711 bytes, not a WAV file.

## Run server

```bash
PYTHONPATH=src python -m nemotron_testbed.main
```

The gRPC port is opened only after all enabled runtimes have loaded successfully.

## Test transcribe

```bash
PYTHONPATH=src python scripts/client_wav.py sample_vi.wav \
  --address 127.0.0.1:50052 \
  --language vi \
  --task transcribe
```

## Test translate + TTS

Vietnamese -> English (`--language` is the target):

```bash
PYTHONPATH=src python scripts/client_wav.py sample_vi.wav \
  --address 127.0.0.1:50052 \
  --language en \
  --task translate \
  --audio-out translated_en.pcmu
```

English -> Vietnamese:

```bash
PYTHONPATH=src python scripts/client_wav.py sample_en.wav \
  --address 127.0.0.1:50052 \
  --language vi \
  --task translate \
  --audio-out translated_vi.pcmu
```

Dual-direction call simulation:

```bash
PYTHONPATH=src python scripts/client_dual_wav.py \
  vi_side.wav en_side.wav \
  --address 127.0.0.1:50052 \
  --task translate \
  --mo-language en \
  --mt-language vi
```

Replay a capture taken at rtpgw (classic pcap, RTP over UDP, G.711 PCMU/PCMA; one flow per stream). `client_wav.py` rebuilds the `AudioChunk`s the real gateway would send (jitter buffer, 8 -> 16 kHz resampler, 80 ms chunks, original RTP timestamps minus the resampler delay). `--gw-resampler sinc|linear` picks the rtpgw `audio.resampler` to emulate, for `--pcap` and `--narrowband` alike: `sinc` (default, rtpgw default since 2026-11) is a half-band FIR with a 3.25 ms delay, `linear` the older per-packet interpolation:

```bash
python scripts/client_wav.py --pcap capture.pcap --list-flows
PYTHONPATH=src python scripts/client_wav.py --pcap capture.pcap --flow '#0' \
  --address 127.0.0.1:50052 \
  --language en \
  --task translate \
  --timeline
```

`--jitter off` sends packets in file order (like the gateway's `pcap_replay`), `--pace capture` keeps the capture's own timing. Details and limits: `docs/scripts-guide.md` (2.1).

For translate tasks `--language` / `--mo-language` / `--mt-language` are the **target** language (the source is detected). Here MO carries Vietnamese speech and wants English, so `--mo-language en`; MT is the reverse. To send the SOURCE hint (`AudioChunk.language_source`) use `--language-source` (`client_wav.py`) or `--mo-language-source` / `--mt-language-source` (`client_dual_wav.py`).

## Convert raw PCMU probe output to WAV for listening

With ffmpeg:

```bash
ffmpeg -f mulaw -ar 8000 -ac 1 -i translated_en.pcmu translated_en.wav
```

For `TTS_OUTPUT_PT=8`, use `-f alaw` instead.

## Metrics

Important ASR metrics:

```text
asr_runtime_queue_depth{worker="..."}
asr_pool_workers
asr_stream_affinities
asr_model_step_seconds
asr_ingress_backlog_ms
asr_deferred_backlog_ms
asr_rtp_timestamp_gaps_total
asr_rtp_out_of_order_total
```

Translation/TTS:

```text
translation_queue_depth{worker="..."}
translation_pool_workers
translation_seconds
tts_queue_depth{worker="..."}
tts_pool_workers
tts_seconds
translate_pipeline_pending
translate_pipeline_results_total
```

Audio preprocessing (only when enabled; the `STATS` log line gains an `audio_preprocessing` block):

```text
audio_preproc_pool_workers{backend}        # 0 = disabled or pool failed to start
audio_preproc_streams_active{backend}
audio_preproc_queue_depth{worker}
audio_preproc_window_seconds{backend}      # compute time per window; tune compute_budget_ms from its p99
audio_preproc_queue_wait_seconds{backend}
audio_preproc_late_seconds{backend}        # late region whose result still arrived: arrival minus the moment it was needed
audio_preproc_slack_seconds{backend}       # enhanced region: moment needed minus arrival; read the low tail (p1)
audio_preproc_flush_seconds{backend}
audio_preproc_regions_total{backend,outcome}   # enhanced, silent, late, shed, error, bypass
audio_preproc_fallback_total{backend,reason}   # resemble-enhance: DENOISER_LATE, DENOISER_QUEUE_FULL, DENOISER_ERROR; audx: AUDX_LATE, AUDX_QUEUE_FULL, AUDX_ERROR
audio_preproc_stream_bypass_total{backend}
audio_preproc_shed_streams_total{backend}
```

End-to-end latency (histograms, available at `logging.level: INFO`):

```text
first_audio_ms              # utterance start -> first audio emitted (includes the speech itself)
post_speech_latency_ms      # VAD END -> first audio emitted (excludes the speech)
speech_end_to_audio_sent_ms # estimated end of speech -> first audio accepted by gRPC (assumes realtime pace)
speech_end_to_audio_sent_preproc_ms # same, plus the audio_preprocessing delay D; only streams with a preprocessor
grpc_buffer_wait_ms         # emit -> gRPC handler picks the result up
grpc_send_ms                # time gRPC took to accept one result for sending
segment_phase_ms{phase}     # per segment, up to its first audio: asr, translate_queue, translate,
                            # tts_queue, tts (first phrase), order_hold, total (see docs/pipeline-trace-guide.md)
```

With `logging.pipeline_trace: true` the same breakdown is written as one `SEGMENT_LATENCY`
line per segment, just before its first `EMIT`.

If Nemotron cannot keep up with realtime audio, increasing `MAX_INGRESS_AUDIO_MS` only delays failure. Capacity must be increased or concurrency limited.

If Translation/TTS cannot keep up, `MAX_POSTPROCESS_PENDING_PER_RPC` bounds how far translated speech can lag behind the conversation.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The unit tests do not load the real models; they validate state flow, runtime-pool routing/ASR affinity, protobuf fields, ordering primitives, buffers, VAD, and G.711 output shape.

## Current deliberate limitation

This version is **utterance-final translation/TTS**, not unstable word-by-word speech translation. The next phase for lower latency is a stable-prefix committer:

```text
Nemotron partial snapshots
 -> detect a text prefix stable for N updates / punctuation / pause
 -> permanently commit phrase
 -> translate committed phrase
 -> TTS committed phrase only
```

Audio that has already been spoken cannot be rolled back, so raw ASR partials must never be sent directly to TTS.

### Quiet console + file logging

Recommended production defaults:

```bash
LOG_LEVEL=INFO
CONSOLE_LOG_LEVEL=ERROR
THIRD_PARTY_LOG_LEVEL=ERROR
LOG_FILE=/app/logs/nemotron-engine.log
SUPPRESS_MODEL_INIT_OUTPUT=true
```

With this setup, engine INFO diagnostics are retained in the rotating log file,
while the console only shows ERROR messages. NeMo/Lightning/Transformers startup
chatter is suppressed, and direct stdout/stderr from Nemotron restore/prompt
initialization is discarded. Set `CONSOLE_LOG_LEVEL=WARNING` when operational
warnings should also be visible in the console. Set
`SUPPRESS_MODEL_INIT_OUTPUT=false` temporarily when debugging model startup.

## Confidence semantics

`RecognitionResult.confidence` is always the source ASR utterance confidence in `[0,1]`. For translate mode the value is carried unchanged through EnViT5 and Piper; it does not claim to measure translation or TTS quality. Native NeMo confidence is preferred, with a length-normalized RNNT hypothesis-score fallback when needed.
