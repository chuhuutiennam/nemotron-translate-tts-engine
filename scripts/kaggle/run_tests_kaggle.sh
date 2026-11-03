#!/usr/bin/env bash
# Run the Kaggle test matrix after setup_kaggle.sh. Results go to $OUT_DIR (default /kaggle/working/out).
#
# Steps (STEPS="A B C T V D E" by default; any subset, in any order of the letters):
#   V  probe_vad.py on 3 clips, narrowband (real Silero; the probe uses default thresholds, not the yaml)
#   A  transcribe, language auto, no language_source                  (baseline)
#   B  transcribe, language auto, language_source=vi                  (the new field)
#   C  transcribe, language=vi                                        (old explicit path; B should match C)
#   T  translate vi -> en on 5 clips, with and without language_source (qualitative)
#   D  like B with vad.neg_threshold=0.5 (engine restarted)           (hysteresis 0.35 vs 0.5, compare to B)
#   E  vi/en interleaved, asr.workers=1: auto vs language_source=@lang (cost of changing the ASR prompt)
#   S  like B with vad.end_silence_ms = each value of SILENCE_SWEEP, one engine per value (not in the default
#      STEPS; run as STEPS="B S"): does ending utterances at shorter pauses cost WER? Base = B (production 250 ms).
#      The summary adds utterances per clip and empty ASR finals per run (eval_wer.py logstats).
#
# Gate G0 of docs/design-notes/resemble-denoiser-integration-design.md (not in the default STEPS; needs setup_enhance.sh):
#   N  clips -> G.711 path -> resemble denoiser (ONNX) -> engine ASR      (compare to A, the raw branch)
#   H  clips -> G.711 path -> resemble enhancer, nfe ENH_NFE -> engine ASR (compare to A; run with STEPS="A N H")
#   G  GPU timing of the denoiser and the enhancer (nfe 32 and 64, 2 s and 10 s chunks) -> $OUT_DIR/enhance_bench.json
# N and H pre-process the WAVs in a separate venv BEFORE the engine starts (the GPU is used one stage at a time),
# then stream them with --no-narrowband (the G.711 round trip is already in the processed WAVs).
#
# Engine launches: main (A B C T N H), neg05 (D), w1 (E), es<ms> (S, one per value). Each loads all models again (minutes).
#
# Environment: ROOT_DIR VENV OUT_DIR DATA_DIR STEPS LIMIT (0 = all 200 clips) CONCURRENCY MIX_PAIRS
#              EN_WAV ENGINE_OFFLINE (1 = HF_HUB_OFFLINE like production) READY_TIMEOUT_S
#              VENV_ENH ASSETS_DIR ENH_DIR ENH_NFE (steps N H G, see setup_enhance.sh)
#              SILENCE_SWEEP (step S, vad.end_silence_ms values in ms, default "400 600 800")
#              CLIP_TIMEOUT_S (gRPC timeout per clip, default 300)
#              DEVICE (cuda, the default | cpu: a CPU-only Linux machine, docs/local-cpu-test-design.md; N H G need a GPU)
#              ASR_WORKERS (default 2) CPU_THREADS (default nproc / ASR_WORKERS): DEVICE=cpu only, written to the yaml
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="${VENV:-/kaggle/working/venv}"
OUT_DIR="${OUT_DIR:-/kaggle/working/out}"
DATA_DIR="${DATA_DIR:-/kaggle/working/testdata}"
VENV_ENH="${VENV_ENH:-/kaggle/working/venv_enh}"
ASSETS_DIR="${ASSETS_DIR:-/kaggle/working/resemble_assets}"
ENH_DIR="${ENH_DIR:-/kaggle/working/enh}"
ENH_NFE="${ENH_NFE:-32}"
STEPS="${STEPS:-A B C T V D E}"
LIMIT="${LIMIT:-0}"
CONCURRENCY="${CONCURRENCY:-4}"
MIX_PAIRS="${MIX_PAIRS:-20}"
SILENCE_SWEEP="${SILENCE_SWEEP:-400 600 800}"
CLIP_TIMEOUT_S="${CLIP_TIMEOUT_S:-300}"
DEVICE="${DEVICE:-cuda}"
ASR_WORKERS="${ASR_WORKERS:-2}"
CPU_THREADS="${CPU_THREADS:-}"
EN_WAV="${EN_WAV:-$ROOT_DIR/asr_data/OSR_us_000_0017_8k.wav}"
ENGINE_OFFLINE="${ENGINE_OFFLINE:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-2400}"
ADDRESS="127.0.0.1:50052"
PY="$VENV/bin/python"
EVAL=("$PY" "$ROOT_DIR/scripts/kaggle/eval_wer.py")
ENGINE_PID=""

log() { printf '\n== %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

case "$DEVICE" in
  cuda | cpu) ;;
  *) die "DEVICE must be cuda or cpu, got '$DEVICE'" ;;
esac
[[ "$CLIP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || die "CLIP_TIMEOUT_S must be a positive integer (seconds), got '$CLIP_TIMEOUT_S'"
if [ "$DEVICE" = cpu ]; then
  for step in N H G; do
    if has "$step"; then die "step $step needs a GPU: remove it from STEPS when DEVICE=cpu"; fi
  done
  [[ "$ASR_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "ASR_WORKERS must be a positive integer, got '$ASR_WORKERS'"
  if [ -z "$CPU_THREADS" ]; then
    CPU_THREADS=$(( $(nproc) / ASR_WORKERS ))
    if [ "$CPU_THREADS" -lt 1 ]; then CPU_THREADS=1; fi
  fi
  [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || die "CPU_THREADS must be a positive integer, got '$CPU_THREADS'"
fi

[ -x "$PY" ] || die "no venv at $VENV (run setup_kaggle.sh first)"
[ -L "$ROOT_DIR/models" ] || [ -d "$ROOT_DIR/models" ] || die "$ROOT_DIR/models is missing (run setup_kaggle.sh)"
ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
# shellcheck disable=SC1091
source "$VENV/engine-env.sh"
mkdir -p "$OUT_DIR/logs"

limit_args=()
if [ "$LIMIT" -gt 0 ]; then limit_args=(--limit "$LIMIT"); fi

silence_runs=()
if has S; then
  for ms in $SILENCE_SWEEP; do
    [[ "$ms" =~ ^[1-9][0-9]*$ ]] || die "SILENCE_SWEEP must hold positive integers (ms), got '$ms'"
    silence_runs+=("S${ms}_source_vi")
  done
  [ "${#silence_runs[@]}" -gt 0 ] || die "step S needs SILENCE_SWEEP values"
  if ! has B && [ ! -f "$OUT_DIR/B_source_vi.jsonl" ]; then
    echo "WARNING: step S is scored against B_source_vi (production end_silence_ms); add B to STEPS (STEPS=\"B S\")"
  fi
fi

device_sets=()
if [ "$DEVICE" = cpu ]; then
  # same devices as k8s-configmap-cpu.yaml; every other value comes from the yaml
  echo "DEVICE=cpu: ASR_WORKERS=$ASR_WORKERS CPU_THREADS=$CPU_THREADS (nproc $(nproc)); translation (int8) and TTS on the CPU"
  device_sets=(asr.device=cpu "asr.devices=[cpu]" "asr.workers=$ASR_WORKERS" "asr.cpu_threads=$CPU_THREADS"
               translation.device=cpu "translation.devices=[cpu]" translation.compute_type=int8
               tts.device=cpu "tts.devices=[cpu]")
else
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [ "$gpu_count" -ge 1 ] || die "no GPU visible (DEVICE=cpu runs on the CPU)"
  if [ "$gpu_count" -lt 2 ]; then
    echo "WARNING: one GPU: ASR, translation and TTS share cuda:0 (production uses cuda:0 and cuda:1)"
    device_sets=(asr.device=cuda:0 "asr.devices=[cuda:0]" translation.device=cuda:0 "translation.devices=[cuda:0]"
                 tts.device=cuda:0 "tts.devices=[cuda:0]")
  fi
fi

# make_config <out.yaml> <engine log file> [dotted.key=yaml-value ...]
make_config() {
  "$PY" - "$ROOT_DIR/etc/nemotron-engine.yaml" "$ROOT_DIR/scripts/kaggle/kaggle-engine.yaml" "$@" <<'PYEOF'
import sys

import yaml

base_path, override_path, out_path, log_file, *settings = sys.argv[1:]


def merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = value


with open(base_path, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
with open(override_path, encoding="utf-8") as fh:
    merge(cfg, yaml.safe_load(fh) or {})
cfg.setdefault("logging", {})["file"] = log_file
for item in settings:
    key, _, value = item.partition("=")
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(value)
with open(out_path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
PYEOF
}

engine_stop() {
  [ -n "$ENGINE_PID" ] || return 0
  if kill -0 "$ENGINE_PID" 2>/dev/null; then
    kill -TERM "$ENGINE_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$ENGINE_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$ENGINE_PID" 2>/dev/null || true
  fi
  wait "$ENGINE_PID" 2>/dev/null || true
  ENGINE_PID=""
}
trap engine_stop EXIT

# engine_start <label> [dotted.key=yaml-value ...]
engine_start() {
  local label="$1"; shift
  local logdir="$OUT_DIR/logs/$label" cfg="$OUT_DIR/config-$label.yaml"
  mkdir -p "$logdir"
  rm -f "$logdir/nemotron-engine.log"
  make_config "$cfg" "$logdir/nemotron-engine.log" "${device_sets[@]}" "$@"
  local offline=()
  if [ "$ENGINE_OFFLINE" = 1 ]; then
    offline=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "HUGGINGFACE_HUB_CACHE=$ROOT_DIR/models/hf_cache")
  fi
  ( cd "$ROOT_DIR" && exec env ENGINE_CONFIG_FILE="$cfg" "${offline[@]}" "$PY" -m nemotron_testbed.main ) \
    > "$logdir/stdout.log" 2>&1 &
  ENGINE_PID=$!
  log "engine[$label] pid=$ENGINE_PID, loading models (timeout ${READY_TIMEOUT_S}s)"
  local waited=0
  until grep -qs "gRPC server ready" "$logdir/nemotron-engine.log"; do
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
      tail -n 40 "$logdir/stdout.log" >&2 || true
      die "engine[$label] exited before it was ready (see $logdir)"
    fi
    [ "$waited" -lt "$READY_TIMEOUT_S" ] || die "engine[$label] not ready after ${READY_TIMEOUT_S}s"
    sleep 5
    waited=$((waited + 5))
  done
  echo "engine[$label] ready after ${waited}s"
}

# run_eval <name> [eval_wer.py run options ...]
run_eval() {
  local name="$1"; shift
  log "run $name"
  "${EVAL[@]}" run --address "$ADDRESS" --tag "${name%%_*}" --out "$OUT_DIR/$name.jsonl" --timeout-s "$CLIP_TIMEOUT_S" "$@"
}

source_modes() {
  local counts
  counts="$(grep -ho '"source_mode": "[a-z]*"' "$1" 2>/dev/null | sort | uniq -c || true)"
  if [ -n "$counts" ]; then echo "$counts"; else echo "(no source_mode in $1)"; fi
}

# enh_run <enhance_wavs.py args ...>: the side venv, with its own LD_LIBRARY_PATH (the engine's nvidia libs must not leak in)
enh_run() {
  (
    export LD_LIBRARY_PATH="$ORIG_LD_LIBRARY_PATH"
    # shellcheck disable=SC1091
    source "$VENV_ENH/enh-env.sh"
    exec "$VENV_ENH/bin/python" "$ROOT_DIR/scripts/kaggle/enhance_wavs.py" "$@"
  )
}

# enhance_branch <name> <denoiser|enhancer> [enhance_wavs.py run options ...]
enhance_branch() {
  local name="$1" branch="$2"
  shift 2
  log "$name: $branch -> $ENH_DIR/$name"
  enh_run run --assets-dir "$ASSETS_DIR" --manifest "$MANIFEST" --out "$ENH_DIR/$name" --branch "$branch" \
    "${limit_args[@]}" "$@" | tee "$OUT_DIR/enhance_$name.txt"
  cp "$ENH_DIR/$name/stats.jsonl" "$OUT_DIR/enhance_${name}_stats.jsonl"
  cp "$ENH_DIR/$name/config.json" "$OUT_DIR/enhance_${name}_config.json"
}

log "Test data"
if [ ! -f "$DATA_DIR/vi.tsv" ]; then
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "${EVAL[@]}" prepare --out "$DATA_DIR"
fi
MANIFEST="$DATA_DIR/vi.tsv"

if has V; then
  log "V: probe_vad.py (narrowband)"
  {
    for n in 000 001 002; do
      echo "### vi-$n"
      "$PY" "$ROOT_DIR/scripts/probe_vad.py" "$DATA_DIR/wav/vi-$n.wav" --narrowband --end-silence-ms 250 \
        --silero-model-path "$ROOT_DIR/models/silero/silero_vad.jit"
    done
  } 2>&1 | tee "$OUT_DIR/probe_vad.txt" | tail -n 40 || echo "WARNING: probe_vad failed (see $OUT_DIR/probe_vad.txt)"
fi

if has N || has H || has G; then
  [ -x "$VENV_ENH/bin/python" ] || die "no venv at $VENV_ENH (run setup_enhance.sh first)"
  [ -f "$ASSETS_DIR/denoiser.onnx" ] || die "no assets in $ASSETS_DIR (run setup_enhance.sh first)"
  if has N; then
    enhance_branch N_denoiser denoiser --denoiser-backend onnx
  fi
  if has H; then
    enhance_branch H_enhancer enhancer --nfe "$ENH_NFE"
  fi
  if has G; then
    log "G: GPU timing"
    enh_run bench --assets-dir "$ASSETS_DIR" --manifest "$MANIFEST" --out "$OUT_DIR/enhance_bench.json" 2>&1 \
      | tee "$OUT_DIR/enhance_bench.txt" || echo "WARNING: bench failed (see $OUT_DIR/enhance_bench.txt)"
  fi
fi

if has A || has B || has C || has T || has N || has H; then
  engine_start main
  warm=(--warmup 2)
  if has A; then
    run_eval A_auto --manifest "$MANIFEST" --language "" --language-source "" --concurrency "$CONCURRENCY" "${warm[@]}" "${limit_args[@]}"
    warm=()
  fi
  if has N; then
    run_eval N_denoiser --manifest "$ENH_DIR/N_denoiser/$(basename "$MANIFEST")" --no-narrowband --language "" --language-source "" \
      --concurrency "$CONCURRENCY" "${warm[@]}" "${limit_args[@]}"
    warm=()
  fi
  if has H; then
    run_eval H_enhancer --manifest "$ENH_DIR/H_enhancer/$(basename "$MANIFEST")" --no-narrowband --language "" --language-source "" \
      --concurrency "$CONCURRENCY" "${warm[@]}" "${limit_args[@]}"
    warm=()
  fi
  if has B; then
    run_eval B_source_vi --manifest "$MANIFEST" --language "" --language-source vi --concurrency "$CONCURRENCY" "${limit_args[@]}"
  fi
  if has C; then
    run_eval C_language_vi --manifest "$MANIFEST" --language vi --language-source "" --concurrency "$CONCURRENCY" "${limit_args[@]}"
  fi
  if has T; then
    run_eval T_vi2en_source --manifest "$MANIFEST" --task realtime_translation --language en --language-source vi --concurrency 1 --limit 5
    run_eval T_vi2en_auto --manifest "$MANIFEST" --task realtime_translation --language en --language-source "" --concurrency 1 --limit 5
    "${EVAL[@]}" show "$OUT_DIR/T_vi2en_source.jsonl" -n 5
    "${EVAL[@]}" show "$OUT_DIR/T_vi2en_auto.jsonl" -n 5
  fi
  log "STREAM_OPEN source_mode counts (main engine)"
  source_modes "$OUT_DIR/logs/main/nemotron-engine.log"
  engine_stop
fi

if has D; then
  engine_start neg05 vad.neg_threshold=0.5
  run_eval D_neg05_source_vi --manifest "$MANIFEST" --language "" --language-source vi --concurrency "$CONCURRENCY" "${limit_args[@]}"
  engine_stop
fi

if has S; then
  for ms in $SILENCE_SWEEP; do
    engine_start "es$ms" vad.end_silence_ms="$ms"
    run_eval "S${ms}_source_vi" --manifest "$MANIFEST" --language "" --language-source vi --concurrency "$CONCURRENCY" "${limit_args[@]}"
    engine_stop
  done
fi

if has E; then
  "${EVAL[@]}" mix --manifest "$MANIFEST" --en-wav "$EN_WAV" --count "$MIX_PAIRS" --out "$OUT_DIR/mix.tsv"
  engine_start w1 asr.workers=1
  run_eval E_auto --manifest "$OUT_DIR/mix.tsv" --language "" --language-source "" --concurrency 1 --warmup 3
  run_eval E_explicit --manifest "$OUT_DIR/mix.tsv" --language "" --language-source @lang --concurrency 1 --warmup 3
  run_eval E_auto_repeat --manifest "$OUT_DIR/mix.tsv" --language "" --language-source "" --concurrency 1
  log "STREAM_OPEN source_mode counts (w1 engine)"
  source_modes "$OUT_DIR/logs/w1/nemotron-engine.log"
  engine_stop
fi

# score_group <csv name> <title> <base> [other ...]  (run names without .jsonl; skipped unless the base exists)
score_group() {
  local csv="$1" title="$2" base="$3"; shift 3
  local files=("$OUT_DIR/$base.jsonl")
  local name
  [ -f "${files[0]}" ] || return 0
  for name in "$@"; do
    [ -f "$OUT_DIR/$name.jsonl" ] && files+=("$OUT_DIR/$name.jsonl")
  done
  printf '\n### %s\n' "$title"
  "${EVAL[@]}" score --base "${files[0]}" "${files[@]:1}" --csv "$OUT_DIR/$csv"
}

log "Summary"
{
  score_group score_A_auto.csv "language_source vs auto (base = A_auto; C = language=vi)" A_auto B_source_vi C_language_vi
  score_group score_B_source_vi.csv "VAD hysteresis (base = B_source_vi with neg_threshold 0.35; D = 0.5)" B_source_vi D_neg05_source_vi
  score_group score_E_auto.csv "prompt switching, vi/en interleaved, 1 ASR worker (base = E_auto; E_auto_repeat = noise floor)" E_auto E_explicit E_auto_repeat
  if [ -f "$OUT_DIR/N_denoiser.jsonl" ] || [ -f "$OUT_DIR/H_enhancer.jsonl" ]; then
    score_group score_preprocessing.csv "G0 pre-processing (base = A_auto, raw; N = ONNX denoiser; H = enhancer nfe $ENH_NFE)" A_auto N_denoiser H_enhancer
  fi
  if [ "${#silence_runs[@]}" -gt 0 ]; then
    score_group score_S_end_silence.csv \
      "VAD end_silence_ms (base = B_source_vi, production 250 ms; S<ms> = vad.end_silence_ms; not paced, so no latency: the last segment of each turn comes <ms> - 250 ms later)" \
      B_source_vi "${silence_runs[@]}"
    silence_tags=(B)
    for name in "${silence_runs[@]}"; do silence_tags+=("${name%%_*}"); done
    printf '\n### utterances per run (engine logs; B = main engine)\n'
    "${EVAL[@]}" logstats --tags "${silence_tags[@]}" --csv "$OUT_DIR/logstats_S.csv" "$OUT_DIR"/logs/*/nemotron-engine.log \
      || echo "WARNING: logstats failed"
  fi
} 2>&1 | tee "$OUT_DIR/summary.txt"
echo
echo "Done. Results in $OUT_DIR (summary.txt, *.jsonl, score_*.csv, enhance_*, logs/<engine>/)."
