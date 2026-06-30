#!/bin/bash
# =============================================================================
# Staged spot-check for a multi-TB `data_type=label` corpus.
#   Wraps scripts/check_label.sh (FRAC = stratified per-.label-file sampling)
#   so you never touch the 95%+ you don't sample.  See docs/check-label-dataset.md
#
#   Stage 1: FRAC=1%  -> fast whole-corpus shape (missing / too_long / SR / dur)
#   Stage 2: FRAC=5%  -> deeper pass; full audio + token_len + label alignment
#
# Edit ROOT below (or pass it in), then just run:
#     ./scripts/spotcheck_label_3tb.sh
#     ROOT=/share/voice-dataset ./scripts/spotcheck_label_3tb.sh
#
# Override anything check_label.sh understands, e.g.:
#     ROOT=/data/voice CTX_LEN=2048 WORKERS=16 LABEL_EXCLUDE=misc,noise \
#         ./scripts/spotcheck_label_3tb.sh
#     STAGE1_FRAC=0.005 STAGE2_FRAC=0.1 ./scripts/spotcheck_label_3tb.sh
#     STAGE=1 ./scripts/spotcheck_label_3tb.sh     # run only stage 1
# =============================================================================
set -e
cd "$(dirname "$0")/.."          # repo root

# ---- EDIT ME: root folder that contains the *.label files ----
ROOT=${ROOT:-/share/voice-dataset}

CTX_LEN=${CTX_LEN:-1024}                 # must match training
WORKERS=${WORKERS:-16}                   # decode threads (bump on fast storage)
STAGE1_FRAC=${STAGE1_FRAC:-0.01}         # quick whole-corpus shape
STAGE2_FRAC=${STAGE2_FRAC:-0.05}         # deeper precision pass
STAGE=${STAGE:-both}                     # 1 | 2 | both
REPORT_DIR=${REPORT_DIR:-out/label-check}

if [ ! -d "$ROOT" ]; then
    echo "[fatal] ROOT is not a directory: $ROOT"
    echo "        edit ROOT at the top of this script, or:  ROOT=/path ./scripts/spotcheck_label_3tb.sh"
    exit 2
fi
mkdir -p "$REPORT_DIR"

run_stage() {
    local frac=$1 tag=$2
    echo
    echo "################################################################"
    echo "# Stage $tag : FRAC=$frac  (root=$ROOT)"
    echo "################################################################"
    ROOT="$ROOT" CTX_LEN="$CTX_LEN" WORKERS="$WORKERS" \
        FRAC="$frac" \
        LABEL_EXCLUDE="${LABEL_EXCLUDE:-}" \
        AUDIO_ROOT="${AUDIO_ROOT:-}" MAX_UP="${MAX_UP:-2}" \
        REPORT="$REPORT_DIR/problems-frac$frac.jsonl" \
        SUMMARY="$REPORT_DIR/summary-frac$frac.json" \
        bash scripts/check_label.sh
}

# Compact digest from a stage's summary JSON (resolution rate + status counts).
digest() {
    local frac=$1 sumfile="$REPORT_DIR/summary-frac$1.json"
    [ -f "$sumfile" ] || return 0
    python - "$sumfile" "$frac" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  FRAC={sys.argv[2]}: checked={s['entries_checked']}  "
      f"resolved={s['resolved_pct']}%  missing={s['missing']}  "
      f"verdict={s['verdict']}")
print(f"     status      : {s['status_counts']}")
print(f"     resolution  : {s['path_resolution']}")
tl = s.get('token_len') or {}
if tl:
    print(f"     token_len   : mean={tl['mean']} max={tl['max']} over_ctx={tl['over_limit']}")
print(f"     problems    : {s['problems_written']} -> {s['report_path']}")
PY
}

if [ "$STAGE" = "both" ] || [ "$STAGE" = "1" ]; then
    run_stage "$STAGE1_FRAC" "1"
fi
if [ "$STAGE" = "both" ] || [ "$STAGE" = "2" ]; then
    run_stage "$STAGE2_FRAC" "2"
fi

echo
echo "================================================================"
echo "=== DIGEST (reports + summaries under: $REPORT_DIR/) ==="
echo "================================================================"
if [ "$STAGE" = "both" ] || [ "$STAGE" = "1" ]; then digest "$STAGE1_FRAC"; fi
if [ "$STAGE" = "both" ] || [ "$STAGE" = "2" ]; then digest "$STAGE2_FRAC"; fi
echo
echo "  full summaries : $REPORT_DIR/summary-frac*.json"
echo "  problem lists  : $REPORT_DIR/problems-frac*.jsonl"
