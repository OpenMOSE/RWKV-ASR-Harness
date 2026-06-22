# -*- coding: utf-8 -*-
"""Materialize a local subset of japanese-asr/ja_asr.reazon_speech_all.

The full dataset is ~19.2M clips (>2 TB) across 16 streaming configs
(subset_0..15). HuggingFace streaming downloads shards ON DEMAND, so taking the
first K clips of a config only fetches the shards it needs — we never download
the whole 2 TB. We keep raw WAV bytes (avoids the torchcodec dependency that
datasets>=4 needs for on-the-fly decoding), filter by duration, and pre-compute
the speech token length so dataloader workers never touch the WavLM backbone.

Robustness (important for multi-hour streaming jobs):
  * Each config is collected and saved to its own part dir under <out>/_parts/.
    A network failure (e.g. HF CDN 408) only affects the current config — earlier
    configs are already on disk.
  * Per-config retry: each config is attempted up to --retries times (a fresh
    stream each attempt, so no duplicates); whatever was collected on the last
    attempt is kept.
  * Resume: a part dir that already has >= 95% of its target is skipped.
  * After all configs, parts are concatenated into <out> (memory-mapped, low RAM).

Sizing:
  --frac 0.01            ~1% of EVERY config (spread across all 16) -> ~192k clips
  --max_samples N        up to N clips total, spread across the listed --configs

Output columns (consumed by data_type='asr' in lg_train/dataset.py):
    audio_bytes  : binary  (original WAV container, 16 kHz)
    transcription: string  (Japanese ground truth)
    token_len    : int32   (SpeechProjector output length == #<|image_pad|>)

Examples
--------
python -m lg_train.prepare.build_reazon_subset --out ./localdataset --frac 0.01
python -m lg_train.prepare.build_reazon_subset --out ./localdataset_small \
    --configs subset_0 --max_samples 5000
"""

import argparse
import io
import os

# Route shard downloads through the standard resolve CDN; the xet bridge has
# been returning 408 timeouts for this dataset. Set before importing hf libs.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import soundfile as sf
from datasets import (Dataset, Features, Value, concatenate_datasets,
                      get_dataset_config_names, load_dataset,
                      load_dataset_builder, load_from_disk, Audio)

from lg_train.encoder.speech_encoder import speech_token_len

REPO = "japanese-asr/ja_asr.reazon_speech_all"

FEATURES = Features({
    "audio_bytes": Value("binary"),
    "transcription": Value("string"),
    "token_len": Value("int32"),
})


def collect_config(cfg, target, max_sec, min_sec, text_field, retries):
    """Stream one config, return up to `target` filtered rows (resilient)."""
    best = []
    for attempt in range(1, retries + 1):
        rows = []
        kept = skipped = 0
        try:
            ds = load_dataset(REPO, cfg, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            for ex in ds:
                if kept >= target:
                    break
                b = ex["audio"]["bytes"]
                text = ex.get(text_field)
                if not b or not text:
                    skipped += 1
                    continue
                try:
                    info = sf.info(io.BytesIO(b))
                except Exception:
                    skipped += 1
                    continue
                if info.samplerate != 16000:
                    skipped += 1
                    continue
                dur = info.frames / info.samplerate
                if dur < min_sec or dur > max_sec:
                    skipped += 1
                    continue
                rows.append({
                    "audio_bytes": b,
                    "transcription": text,
                    "token_len": int(speech_token_len(info.frames)),
                })
                kept += 1
                if kept % 2000 == 0:
                    print(f"    {cfg}: kept={kept}/{target} (skipped={skipped})", flush=True)
            print(f"[ok] {cfg}: kept={kept} (attempt {attempt})", flush=True)
            return rows
        except Exception as e:
            if len(rows) > len(best):
                best = rows
            print(f"[warn] {cfg} attempt {attempt}/{retries} failed at kept={kept}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
    print(f"[partial] {cfg}: keeping {len(best)} rows after {retries} attempts", flush=True)
    return best


def build(out, configs, frac, max_samples, max_sec, min_sec, text_field, retries, num_proc):
    if frac is not None:
        targets = {c: round(frac * load_dataset_builder(REPO, c).info.splits["train"].num_examples)
                   for c in configs}
        print(f"[plan] frac={frac} over {len(configs)} configs -> ~{sum(targets.values())} clips", flush=True)
    else:
        per = max(1, max_samples // len(configs))
        targets = {c: per for c in configs}
        targets[configs[0]] += max_samples - per * len(configs)
        print(f"[plan] max_samples={max_samples} over {len(configs)} configs", flush=True)

    parts_dir = os.path.join(out, "_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_paths = []

    for cfg, target in targets.items():
        part = os.path.join(parts_dir, cfg)
        if os.path.isdir(part):
            try:
                n = len(load_from_disk(part))
                if n >= 0.95 * target:
                    print(f"[skip] {cfg}: already have {n} (>=95% of {target})", flush=True)
                    part_paths.append(part)
                    continue
            except Exception:
                pass
        rows = collect_config(cfg, target, max_sec, min_sec, text_field, retries)
        if rows:
            Dataset.from_list(rows, features=FEATURES).save_to_disk(part)
            part_paths.append(part)

    print("[merge] concatenating parts...", flush=True)
    parts = [load_from_disk(p) for p in part_paths]
    final = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    final.save_to_disk(out, num_proc=num_proc)
    print(f"[saved] {len(final)} samples -> {out}", flush=True)
    print(f"[note] per-config parts kept under {parts_dir} (delete to reclaim space)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build a local ReazonSpeech subset")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--configs", nargs="+", default=None,
                   help="configs to use; default = all 16 subsets")
    p.add_argument("--frac", type=float, default=None,
                   help="fraction of EACH config to keep, e.g. 0.01 for 1%%")
    p.add_argument("--max_samples", type=int, default=50000,
                   help="used only when --frac is not given")
    p.add_argument("--max_sec", type=float, default=20.0)
    p.add_argument("--min_sec", type=float, default=0.3)
    p.add_argument("--text_field", type=str, default="transcription",
                   help="'transcription' (ja) or 'transcription/en_gpt3.5' (en)")
    p.add_argument("--retries", type=int, default=5, help="per-config stream retries")
    p.add_argument("--num_proc", type=int, default=4, help="procs for final save_to_disk")
    args = p.parse_args()

    configs = args.configs or get_dataset_config_names(REPO)
    build(args.out, configs, args.frac, args.max_samples, args.max_sec,
          args.min_sec, args.text_field, args.retries, args.num_proc)
