#!/bin/bash
# =============================================================================
# Validate a data_type=label ASR dataset BEFORE training (no model / no GPU).
#   Wraps tools/check_label_dataset.py — see docs/check-label-dataset.md.
#
# Checks: .label index loads, audio files exist & decode, sample-rate/duration,
# token_len fits ctx_len, and <|image_pad|> placeholder / transcript extraction
# aligns 1:1 with what the trainer builds. A green VERDICT == the dataloader
# should also be happy (this is the first thing to run when you hit
# "num_samples=0" on a label dataset).
#
# Usage:
#   ROOT=/share/voice-dataset ./scripts/check_label.sh            # quick scan
#   ROOT=/share/voice-dataset CHECK_ALL=1 ./scripts/check_label.sh  # exhaustive
#   ROOT=/share/voice-dataset LABEL_EXCLUDE=misc,noise ./scripts/check_label.sh
#   ROOT=/share/voice-dataset STRICT=1 ./scripts/check_label.sh    # CI (exit!=0 on problems)
#
# Env knobs (all optional except ROOT):
#   ROOT            root folder containing *.label files   (REQUIRED)
#   CTX_LEN        context length, must match training      (default 1024)
#   LABEL_EXCLUDE  comma-separated keywords to skip         (default empty)
#   FRAC           stratified spot-check: keep this fraction of EACH .label
#                  file (e.g. 0.05 = 5%); the rest is never touched. Best for
#                  multi-TB data. Sampled entries are fully decoded.  (default: off)
#   MAX_CHECK      entries to fully decode (random sample)  (default 300)
#   CHECK_ALL      1 = fully decode EVERY entry (slow)      (default 0)
#   WORKERS        decode threads                           (default 8)
#   SHOW           fully-built training samples to print    (default 3)
#   EXAMPLES       problem examples printed per category    (default 8)
#   REPORT         path for the FULL problem list (JSONL)   (default label_check_report.jsonl)
#   SUMMARY        path for the run summary (JSON)          (default label_check_summary.json)
#   NO_REPORT      1 = do not write a report file           (default 0)
#   DECODE_ERRORS  non-UTF-8 .label files: skip | ignore     (default skip)
#   AUDIO_ROOT     explicit base dir for relative audio paths (tried first; default: off)
#   MAX_UP         parent levels tried for off-by-one paths   (default 2)
#   STRICT         1 = exit non-zero if any problem found   (default 0)
#
# For a huge dataset use CHECK_ALL=1: the terminal shows only streamed counters
# + histograms (O(1) memory); every problem entry is written to $REPORT.
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # repo root

ROOT=${ROOT:?set ROOT=/path/to/voice-dataset (folder with *.label files)}
CTX_LEN=${CTX_LEN:-1024}
LABEL_EXCLUDE=${LABEL_EXCLUDE:-}
MAX_CHECK=${MAX_CHECK:-300}
WORKERS=${WORKERS:-8}
SHOW=${SHOW:-3}
EXAMPLES=${EXAMPLES:-8}
REPORT=${REPORT:-label_check_report.jsonl}
SUMMARY=${SUMMARY:-label_check_summary.json}
DECODE_ERRORS=${DECODE_ERRORS:-skip}     # skip = drop non-UTF-8 .label files | ignore = drop bad bytes
MAX_UP=${MAX_UP:-2}                       # parent levels tried for off-by-one relative paths

ARGS=(--root "$ROOT" --ctx_len "$CTX_LEN" --max_check "$MAX_CHECK" \
      --workers "$WORKERS" --show "$SHOW" --examples "$EXAMPLES" \
      --report "$REPORT" --summary "$SUMMARY" \
      --decode_errors "$DECODE_ERRORS" --max_up "$MAX_UP")
[ -n "${AUDIO_ROOT:-}" ] && ARGS+=(--audio_root "$AUDIO_ROOT")
[ -n "$LABEL_EXCLUDE" ] && ARGS+=(--label_exclude "$LABEL_EXCLUDE")
[ -n "${FRAC:-}" ] && ARGS+=(--frac "$FRAC")
[ "${CHECK_ALL:-0}" = "1" ] && ARGS+=(--check_all)
[ "${NO_REPORT:-0}" = "1" ] && ARGS+=(--no_report)
[ "${STRICT:-0}" = "1" ]  && ARGS+=(--strict)

echo "=== check_label | root=$ROOT | ctx_len=$CTX_LEN | exclude='${LABEL_EXCLUDE:-none}' | frac=${FRAC:-off} check_all=${CHECK_ALL:-0} ==="
exec python tools/check_label_dataset.py "${ARGS[@]}"
