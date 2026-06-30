# -*- coding: utf-8 -*-
"""Stand-alone validator for a `data_type=label` ASR dataset.

Runs WITHOUT the training stack / model so it can be used on a confidential
data box: it checks that the *.label index loads, that audio files exist and
decode cleanly, and that transcript / placeholder-token extraction matches what
the trainer would build.  It mirrors the logic in lg_train/dataset.py
(_load_label_index, _resolve_audio_path, _process_label) and
lg_train/encoder/speech_encoder.py (speech_token_len) so a green report here
means the real dataloader should also be happy.

Dependencies for the core checks: numpy, soundfile, librosa (same as training).
The optional deep tokenization check (--show / alignment) additionally imports
the repo's RWKV pipeline; if that import fails the rest still runs.

Usage
-----
    # quick scan (index + headers for ALL entries, full decode of a 300 sample)
    python tools/check_label_dataset.py --root /share/voice-dataset

    # decode every clip (slow but exhaustive), exclude a 'misc' folder
    python tools/check_label_dataset.py --root /share/voice-dataset \
        --check_all --label_exclude misc,noise --ctx_len 1024

    # show 5 fully-built training samples (placeholder alignment + decoded label)
    python tools/check_label_dataset.py --root /share/voice-dataset --show 5

Exit code is non-zero when --strict and any problem was found (CI-friendly).
"""

import argparse
import bisect
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

# repo root on path so the optional `lg_train.*` imports resolve when run as
# `python tools/check_label_dataset.py` from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import soundfile as sf
except Exception:                                  # pragma: no cover
    sf = None
try:
    import librosa
except Exception:                                  # pragma: no cover
    librosa = None


# --------------------------------------------------------------------------- #
# token-length geometry — kept identical to lg_train/encoder/speech_encoder.py
# (imported as the source of truth when available; local copy is the fallback)
# --------------------------------------------------------------------------- #
WAVLM_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
WAVLM_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def _wavlm_feat_len(n):
    n = int(n)
    for k, s in zip(WAVLM_CONV_KERNEL, WAVLM_CONV_STRIDE):
        n = (n - k) // s + 1
    return max(n, 1)


def _speech_token_len(n):
    t = _wavlm_feat_len(n)
    return (t + 2 * 2 - 3) // 2 + 1


speech_token_len = _speech_token_len
try:
    from lg_train.encoder.speech_encoder import speech_token_len as _real_stl
    # sanity: our local copy must agree with the trainer's, else geometry drifted
    if all(_real_stl(n) == _speech_token_len(n) for n in (1600, 16000, 32000, 80000)):
        speech_token_len = _real_stl
    else:
        print("[warn] local speech_token_len disagrees with lg_train's — using lg_train's")
        speech_token_len = _real_stl
except Exception as e:
    print(f"[note] using built-in speech_token_len (could not import lg_train's: {e})")


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".wma", ".aiff", ".au"}
IMAGE_PAD = "<|image_pad|>"
IMAGE_TOKEN_ID = 65532


# --------------------------------------------------------------------------- #
# label index — mirrors WorldDataset._load_label_index
# --------------------------------------------------------------------------- #
def load_label_index(root, keywords, frac=None, seed=42, decode_errors="skip",
                     progress_every=5000):
    """Return list of (label_file, lineno, audio_path, text). `keywords` is a
    lower-cased list; entries / files whose path contains one are skipped.

    frac in (0,1]: STRATIFIED sample — keep a random ceil(frac*N) of the valid
    entries of EACH .label file (so the 95% you don't sample are never touched
    downstream: no path resolution, no header read, no decode). This is the way
    to spot-check a multi-TB corpus. A fixed `seed` makes the sample reproducible.

    decode_errors: how to handle a .label file that is not valid UTF-8:
      'skip'   -> warn and skip the WHOLE file (default).
      'ignore' -> drop the offending bytes and keep the file's good lines.
    Index building prints progress every `progress_every` files so a huge corpus
    never looks frozen."""
    do_sample = frac is not None and 0 < frac < 1.0
    rng = random.Random(seed)
    open_errors = "ignore" if decode_errors == "ignore" else "strict"
    index = []
    n_files = n_files_excluded = 0
    n_excluded_entries = 0
    n_bad_lines = 0
    n_before_sample = 0
    n_decode_error = 0          # .label files skipped: not valid UTF-8
    n_read_error = 0            # .label files skipped: OS/read error
    t0 = time.time()
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".label"):
                continue
            fpath = os.path.join(dirpath, fn)
            if keywords and any(kw in fpath.lower() for kw in keywords):
                n_files_excluded += 1
                continue
            n_files += 1
            if progress_every and n_files % progress_every == 0:
                print(f"[index] scanned {n_files} .label files | {len(index)} entries kept "
                      f"| skipped(bad-utf8)={n_decode_error} | {time.time()-t0:.0f}s",
                      flush=True)
            file_entries = []
            try:
                with open(fpath, "r", encoding="utf-8", errors=open_errors) as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.rstrip("\n").rstrip("\r")
                        if not line:
                            continue
                        parts = line.split(None, 1)   # first whitespace (space or TAB)
                        if len(parts) < 2:
                            n_bad_lines += 1
                            continue
                        apath, text = parts[0], parts[1].strip()
                        if not text:
                            n_bad_lines += 1
                            continue
                        if keywords and any(kw in apath.lower() for kw in keywords):
                            n_excluded_entries += 1
                            continue
                        file_entries.append((fpath, lineno, apath, text))
            except UnicodeDecodeError as e:
                # not valid UTF-8 (e.g. byte 0xf1) -> skip the file, keep going
                n_decode_error += 1
                print(f"[warn] skip .label (not valid UTF-8: {e}): {fpath}", flush=True)
                continue
            except OSError as e:
                n_read_error += 1
                print(f"[warn] skip .label (read error: {e}): {fpath}", flush=True)
                continue
            n_before_sample += len(file_entries)
            if do_sample and file_entries:
                k = max(1, math.ceil(frac * len(file_entries)))
                file_entries = rng.sample(file_entries, k)
            index.extend(file_entries)
    return index, dict(n_files=n_files, n_files_excluded=n_files_excluded,
                       n_decode_error=n_decode_error, n_read_error=n_read_error,
                       n_excluded_entries=n_excluded_entries, n_bad_lines=n_bad_lines,
                       n_before_sample=n_before_sample, sampled=do_sample, frac=frac)


# --------------------------------------------------------------------------- #
# path resolution — robust to the messy real world: a .label entry's audio path
# may be absolute, relative to the .label file's own dir, relative to the dataset
# root (or its parent), or off-by-one (an extra/missing leading directory).
# We try a prioritized set of (base, transform) candidates and CACHE the winning
# scheme per .label directory so only the first entry of each file pays the cost.
# resolve() returns (path, scheme, tried): scheme is None when nothing existed.
# --------------------------------------------------------------------------- #
class PathResolver:
    def __init__(self, label_root, audio_root=None, max_up=2):
        self.label_root = os.path.normpath(label_root)
        self.audio_root = os.path.normpath(audio_root) if audio_root else None
        self.max_up = max_up
        self._cache = {}     # label_dir -> winning (base_path, transform_name)

    def _bases(self, label_dir):
        """Candidate base directories, most-likely first."""
        bases = []
        if self.audio_root:
            bases.append(("audio_root", self.audio_root))
        bases.append(("label_dir", label_dir))           # path relative to the .label file
        d = label_dir
        for i in range(1, self.max_up + 1):               # walk up: off-by-one (missing dir)
            d = os.path.dirname(d)
            if not d:
                break
            bases.append((f"label_dir-{i}", d))
        bases.append(("root", self.label_root))
        bases.append(("root_parent", os.path.dirname(self.label_root)))
        # dedup preserving order
        seen, out = set(), []
        for name, b in bases:
            if b and b not in seen:
                seen.add(b); out.append((name, b))
        return out

    @staticmethod
    def _apply(base, tname, rel):
        if tname == "asis":
            r = rel
        elif tname == "strip1":                           # off-by-one (extra leading dir)
            parts = rel.split(os.sep)
            if len(parts) <= 1:
                return None
            r = os.sep.join(parts[1:])
        else:
            return None
        return os.path.join(base, r)

    TRANSFORMS = ("asis", "strip1")

    def resolve(self, rel, label_file):
        if os.path.isabs(rel):
            ok = os.path.exists(rel)
            return rel, ("abs" if ok else None), [rel]

        label_dir = os.path.dirname(label_file)
        tried = []

        # fast path: reuse the scheme that already worked for this .label dir
        plan = self._cache.get(label_dir)
        if plan is not None:
            base, tname, bname = plan
            cand = self._apply(base, tname, rel)
            if cand and os.path.exists(cand):
                return cand, f"{bname}:{tname}", tried

        # full search over (base, transform), most-likely first
        for bname, base in self._bases(label_dir):
            for tname in self.TRANSFORMS:
                cand = self._apply(base, tname, rel)
                if not cand:
                    continue
                tried.append(cand)
                if os.path.exists(cand):
                    self._cache[label_dir] = (base, tname, bname)
                    return cand, f"{bname}:{tname}", tried
        # nothing existed -> best-guess path (label_dir/rel) for the report
        return os.path.join(label_dir, rel), None, tried[:6]


# --------------------------------------------------------------------------- #
# per-entry audio decode — mirrors WorldDataset._process_label
# --------------------------------------------------------------------------- #
def decode_audio_16k(path):
    """Return (wav_float32_mono_16k, orig_sr, n_channels). Raises on failure."""
    try:
        wav, sr = sf.read(path, dtype="float32")
        n_ch = 1 if wav.ndim == 1 else wav.shape[1]
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return wav, sr, n_ch
    except Exception:
        # formats soundfile can't open (mp3 etc.) -> librosa (also resamples)
        wav = librosa.load(path, sr=16000)[0]
        return wav, None, 1   # orig sr unknown via this path


def check_entry(entry, resolver, ctx_len, do_decode):
    """Validate one (label_file, lineno, apath, text). Returns a dict result."""
    label_file, lineno, apath, text = entry
    r = {
        "label_file": label_file, "lineno": lineno, "apath": apath,
        "text": text, "status": "ok", "reason": "",
        "sr": None, "n_ch": None, "dur": None, "token_len": None,
        "resolve_scheme": None,
    }

    # transcript / parsing sanity
    _, ext = os.path.splitext(apath.lower())
    if ext not in AUDIO_EXTS:
        r["status"] = "warn"
        r["reason"] = f"audio path has no known audio extension ({ext or 'none'}); " \
                      f"possible .label parsing issue (path with spaces?)"

    path, scheme, tried = resolver.resolve(apath, label_file)
    r["resolved"] = path
    r["resolve_scheme"] = scheme
    if scheme is None:
        r["status"] = "missing"
        r["reason"] = ("audio file not found; tried: "
                       + ", ".join(tried[:4]) + (" ..." if len(tried) > 4 else ""))
        return r

    # header-only info (cheap) — works for wav/flac; mp3 may need full decode
    if sf is not None:
        try:
            info = sf.info(path)
            r["sr"] = info.samplerate
            r["n_ch"] = info.channels
            r["dur"] = info.frames / info.samplerate if info.samplerate else None
            r["fmt"] = info.format
        except Exception:
            r["fmt"] = "(header unreadable by soundfile; needs librosa)"

    if not do_decode:
        return r

    # full decode + the exact guard the dataloader applies
    try:
        wav, osr, n_ch = decode_audio_16k(path)
    except Exception as e:
        r["status"] = "decode_fail"
        r["reason"] = f"{type(e).__name__}: {str(e)[:120]}"
        return r
    if osr is not None:
        r["sr"], r["n_ch"] = osr, n_ch
    n16k = len(wav)
    r["dur"] = n16k / 16000.0
    if n16k == 0:
        r["status"] = "decode_fail"
        r["reason"] = "decoded to 0 samples"
        return r
    tl = int(speech_token_len(n16k))
    r["token_len"] = tl
    # same guard as _process_label: must leave room for the transcript
    if tl <= 0 or tl > ctx_len - 32:
        r["status"] = "too_long"
        r["reason"] = f"token_len={tl} exceeds ctx_len-32 ({ctx_len - 32}); " \
                      f"clip ~{r['dur']:.1f}s would be dropped (NaN-loss guard)"
    return r


# --------------------------------------------------------------------------- #
# optional deep check: build the real training input_ids/labels
# --------------------------------------------------------------------------- #
def deep_tokenization_check(samples, ctx_len, n_show):
    try:
        from lg_train.utils import build_inputs_and_labels, pipeline
    except Exception as e:
        print(f"\n[deep] skipped (could not import RWKV pipeline: {e})")
        return None

    # verify the placeholder maps to exactly one token == IMAGE_TOKEN_ID
    pad_ids = pipeline.encode(IMAGE_PAD)
    print("\n========== DEEP TOKENIZATION CHECK ==========")
    print(f"pipeline.encode('{IMAGE_PAD}') -> {pad_ids}")
    if len(pad_ids) != 1:
        print(f"[FAIL] placeholder does not encode to a single token "
              f"(got {len(pad_ids)}). Audio/text alignment WILL be wrong.")
    elif pad_ids[0] != IMAGE_TOKEN_ID:
        print(f"[FAIL] placeholder token id {pad_ids[0]} != expected {IMAGE_TOKEN_ID}.")
    else:
        print(f"[ok] placeholder -> single token id {IMAGE_TOKEN_ID}")

    n_mismatch = 0
    shown = 0
    n_checked = 0
    for r in samples:
        # skip entries already reported as hard problems (missing/decode_fail/too_long);
        # the alignment count is about clips that SHOULD build cleanly.
        if r["status"] in ("missing", "decode_fail", "too_long") or r.get("token_len") is None:
            continue
        n_checked += 1
        tl = r["token_len"]
        conversations = [
            {"from": "user", "value": IMAGE_PAD * int(tl)},
            {"from": "assistant", "value": r["text"]},
        ]
        input_ids, label_ids = build_inputs_and_labels(conversations, pipeline, ctx_len, -100)
        n_audio = int((input_ids == IMAGE_TOKEN_ID).sum().item())
        n_sup = int((label_ids != -100).sum().item())
        ok = (n_audio == tl) and (n_sup > 0)
        if not ok:
            n_mismatch += 1
        if shown < n_show:
            shown += 1
            sup_ids = label_ids[label_ids != -100].tolist()
            try:
                decoded = pipeline.decode(sup_ids)
            except Exception:
                decoded = "<decode failed>"
            tag = "ok" if ok else "MISMATCH"
            print(f"\n--- sample [{tag}] {os.path.basename(r['resolved'])} ---")
            print(f"  transcript      : {r['text'][:80]}")
            print(f"  token_len(audio): {tl}  | placeholder tokens in input: {n_audio}")
            print(f"  supervised toks : {n_sup}  | seq len: {len(input_ids)} / ctx {ctx_len}")
            print(f"  decoded labels  : {str(decoded)[:80]}")
    print(f"\n[deep] alignment mismatches: {n_mismatch} / {n_checked} buildable samples checked")
    return n_mismatch


# --------------------------------------------------------------------------- #
# streaming aggregation — O(1) memory regardless of dataset size
# --------------------------------------------------------------------------- #
DUR_EDGES = [0.5, 1, 2, 3, 5, 10, 15, 20, 30, 60]   # seconds
PROBLEM_STATUSES = ("missing", "decode_fail", "too_long", "warn")


def _hist_labels(edges, unit=""):
    labels = []
    for i in range(len(edges) + 1):
        if i == 0:
            labels.append(f"< {edges[0]}{unit}")
        elif i == len(edges):
            labels.append(f">= {edges[-1]}{unit}")
        else:
            labels.append(f"{edges[i - 1]}-{edges[i]}{unit}")
    return labels


def print_hist(title, edges, counts, unit=""):
    labels = _hist_labels(edges, unit)
    total = sum(counts) or 1
    width = max((len(l) for l in labels), default=0)
    print(f"  {title}:")
    for label, c in zip(labels, counts):
        bar = "#" * int(round(40 * c / total))
        print(f"    {label:>{width}} | {c:>9} {100*c/total:5.1f}%  {bar}")


class Aggregator:
    """Accumulates per-entry results without storing them all. Problems are
    streamed to an optional JSONL report; only a small sample is kept in RAM
    for the on-screen examples + the deep-tokenization check."""

    def __init__(self, ctx_len, report_path=None, examples=8, deep_cap=200):
        self.ctx_len = ctx_len
        self.room = ctx_len - 32
        self.tl_edges = [max(1, self.room // 4), max(2, self.room // 2),
                         max(3, (3 * self.room) // 4), self.room]
        self.status = Counter()
        self.sr = Counter()
        self.ch = Counter()
        self.fmt = Counter()
        self.scheme = Counter()      # which path-resolution scheme succeeded
        self.dur_hist = [0] * (len(DUR_EDGES) + 1)
        self.tl_hist = [0] * (len(self.tl_edges) + 1)
        self.dur_n = self.dur_sum = 0
        self.dur_min = float("inf"); self.dur_max = 0.0
        self.tl_n = self.tl_sum = 0
        self.tl_min = 1 << 60; self.tl_max = 0
        self.tl_over = 0
        self.examples = {c: [] for c in PROBLEM_STATUSES}
        self.example_cap = examples
        self.deep_samples = []
        self.deep_cap = deep_cap
        self._report = open(report_path, "w", encoding="utf-8") if report_path else None
        self.report_path = report_path
        self.n_problems_written = 0

    def add(self, r):
        st = r["status"]
        self.status[st] += 1
        self.scheme[r.get("resolve_scheme") or "MISSING"] += 1
        if r.get("sr") is not None:
            self.sr[r["sr"]] += 1
        if r.get("n_ch") is not None:
            self.ch[r["n_ch"]] += 1
        if r.get("fmt"):
            self.fmt[r["fmt"]] += 1
        d = r.get("dur")
        if d is not None:
            self.dur_n += 1; self.dur_sum += d
            self.dur_min = min(self.dur_min, d); self.dur_max = max(self.dur_max, d)
            self.dur_hist[bisect.bisect_right(DUR_EDGES, d)] += 1
        tl = r.get("token_len")
        if tl is not None:
            self.tl_n += 1; self.tl_sum += tl
            self.tl_min = min(self.tl_min, tl); self.tl_max = max(self.tl_max, tl)
            self.tl_hist[bisect.bisect_right(self.tl_edges, tl)] += 1
            if tl > self.room:
                self.tl_over += 1
        # problems: keep a few in RAM, stream the rest to the report file
        if st in PROBLEM_STATUSES:
            if len(self.examples[st]) < self.example_cap:
                self.examples[st].append(r)
            if self._report is not None:
                self._report.write(json.dumps({
                    "status": st, "reason": r.get("reason", ""),
                    "label_file": r["label_file"], "lineno": r["lineno"],
                    "apath": r["apath"], "resolved": r.get("resolved"),
                    "text": r["text"], "sr": r.get("sr"), "dur": r.get("dur"),
                    "token_len": r.get("token_len"),
                }, ensure_ascii=False) + "\n")
                self.n_problems_written += 1
        # deep-check sample: bounded set of cleanly-buildable clips
        if st == "ok" and tl is not None and len(self.deep_samples) < self.deep_cap:
            self.deep_samples.append(r)

    def close(self):
        if self._report is not None:
            self._report.close()

    def print_summary(self):
        print(f"\n========== SUMMARY ==========")
        print(f"status counts: {dict(self.status)}")
        # path resolution breakdown — which scheme located the audio (MISSING = none)
        sch = dict(sorted(self.scheme.items(), key=lambda kv: -kv[1]))
        print(f"path resolution: {sch}")
        found = sum(v for k, v in self.scheme.items() if k != "MISSING")
        total = sum(self.scheme.values()) or 1
        print(f"               resolved {found}/{total} ({100*found/total:.1f}%) | "
              f"MISSING {self.scheme.get('MISSING', 0)}")
        print(f"sample rate  : {dict(self.sr)}")
        print(f"channels     : {dict(self.ch)}")
        print(f"format       : {dict(self.fmt)}")
        if self.dur_n:
            print(f"duration (s) : n={self.dur_n} min={self.dur_min:.2f} "
                  f"mean={self.dur_sum/self.dur_n:.2f} max={self.dur_max:.2f}")
            print_hist("duration histogram", DUR_EDGES, self.dur_hist, "s")
        if self.tl_n:
            print(f"token_len    : n={self.tl_n} min={self.tl_min} "
                  f"mean={self.tl_sum/self.tl_n:.1f} max={self.tl_max} "
                  f"| ctx_len-32={self.room} ({self.tl_over} over limit)")
            print_hist("token_len histogram", self.tl_edges, self.tl_hist)
        non16k = sum(v for k, v in self.sr.items() if k != 16000)
        if non16k:
            print(f"[note] {non16k} clips are not 16 kHz (will be resampled on the fly).")

    def to_dict(self):
        """Machine-readable summary (written to --summary)."""
        found = sum(v for k, v in self.scheme.items() if k != "MISSING")
        total = sum(self.scheme.values())
        dur = None
        if self.dur_n:
            dur = {"n": self.dur_n, "min": round(self.dur_min, 3),
                   "mean": round(self.dur_sum / self.dur_n, 3), "max": round(self.dur_max, 3),
                   "hist": dict(zip(_hist_labels(DUR_EDGES, "s"), self.dur_hist))}
        tl = None
        if self.tl_n:
            tl = {"n": self.tl_n, "min": self.tl_min,
                  "mean": round(self.tl_sum / self.tl_n, 1), "max": self.tl_max,
                  "over_limit": self.tl_over, "room": self.room,
                  "hist": dict(zip(_hist_labels(self.tl_edges), self.tl_hist))}
        return {
            "status_counts": dict(self.status),
            "path_resolution": dict(sorted(self.scheme.items(), key=lambda kv: -kv[1])),
            "resolved": found, "checked": total,
            "resolved_pct": round(100 * found / max(total, 1), 2),
            "missing": self.scheme.get("MISSING", 0),
            "sample_rate": dict(self.sr), "channels": dict(self.ch), "format": dict(self.fmt),
            "duration": dur, "token_len": tl,
            "problems_written": self.n_problems_written, "report_path": self.report_path,
        }

    def print_examples(self):
        for cat in PROBLEM_STATUSES:
            n = self.status.get(cat, 0)
            if not n:
                continue
            shown = self.examples[cat]
            print(f"\n---- {cat}: {n} ----")
            for r in shown:
                print(f"  {os.path.basename(r['label_file'])}:{r['lineno']} | "
                      f"{r['apath']} -> {r['reason']}")
            if n > len(shown):
                where = f" (full list in {self.report_path})" if self.report_path else ""
                print(f"  ... and {n - len(shown)} more{where}")


def run_checks(index, resolver, ctx_len, decode_set, workers, agg, progress_every=5000):
    """Stream entries through the threaded checker with bounded in-flight
    futures (so memory stays flat on huge datasets) and live progress."""
    total = len(index)

    def job(i):
        return check_entry(index[i], resolver, ctx_len, do_decode=(i in decode_set))

    inflight = max(workers * 4, workers + 1)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        it = iter(range(total))
        futs = set()
        for _ in range(inflight):
            try:
                futs.add(ex.submit(job, next(it)))
            except StopIteration:
                break
        while futs:
            ready, futs = wait(futs, return_when=FIRST_COMPLETED)
            for fut in ready:
                agg.add(fut.result())
                done += 1
                try:
                    futs.add(ex.submit(job, next(it)))
                except StopIteration:
                    pass
            if done % progress_every < len(ready) or done == total:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                print(f"  {done}/{total} ({100*done/total:4.1f}%) | "
                      f"{rate:6.0f} entries/s | elapsed {el:5.0f}s", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Validate a data_type=label ASR dataset (no training stack).")
    ap.add_argument("--root", required=True, help="root folder containing *.label files")
    ap.add_argument("--ctx_len", type=int, default=1024, help="context length (matches training)")
    ap.add_argument("--label_exclude", default="", help="comma-separated keywords to skip (e.g. misc,noise)")
    ap.add_argument("--decode_errors", choices=["skip", "ignore"], default="skip",
                    help="how to handle a .label file that is not valid UTF-8: "
                         "'skip' the whole file with a warning (default), or 'ignore' bad bytes and keep its good lines")
    ap.add_argument("--frac", type=float, default=None,
                    help="STRATIFIED spot-check: keep a random fraction (e.g. 0.05 = 5%%) of EACH "
                         ".label file's entries; the rest are never touched (best for multi-TB data). "
                         "The sampled entries are then fully decoded.")
    ap.add_argument("--max_check", type=int, default=300, help="how many entries to fully decode (random sample); ignored when --frac/--check_all is set")
    ap.add_argument("--check_all", action="store_true", help="fully decode EVERY entry (slow, exhaustive)")
    ap.add_argument("--workers", type=int, default=8, help="threads for the decode pass")
    ap.add_argument("--show", type=int, default=3, help="fully-built training samples to print (deep check)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed")
    ap.add_argument("--audio_root", default=None,
                    help="explicit base dir for relative audio paths (tried first); use when audio "
                         "lives outside the .label tree")
    ap.add_argument("--max_up", type=int, default=2,
                    help="how many parent levels above each .label dir to try when resolving "
                         "off-by-one relative paths (default 2)")
    ap.add_argument("--examples", type=int, default=8, help="problem examples to print per category")
    ap.add_argument("--report", default="label_check_report.jsonl",
                    help="write the FULL list of problem entries here as JSONL (default: ./label_check_report.jsonl)")
    ap.add_argument("--no_report", action="store_true", help="do not write a report file")
    ap.add_argument("--summary", default="label_check_summary.json",
                    help="write the run summary (status counts, resolution %%, histograms) here as JSON")
    ap.add_argument("--no_summary", action="store_true", help="do not write a summary file")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any problem is found")
    args = ap.parse_args()

    if sf is None or librosa is None:
        print("[fatal] need `soundfile` and `librosa` installed (pip install soundfile librosa).")
        sys.exit(2)
    if not os.path.isdir(args.root):
        print(f"[fatal] root is not a directory: {args.root}")
        sys.exit(2)

    if args.frac is not None and not (0 < args.frac <= 1.0):
        print(f"[fatal] --frac must be in (0, 1], got {args.frac}")
        sys.exit(2)
    keywords = [k.strip().lower() for k in args.label_exclude.split(",") if k.strip()]

    # ---- 1) index ----
    print(f"========== INDEX ==========\nroot: {args.root}")
    if keywords:
        print(f"exclude keywords: {keywords}")
    if args.frac is not None:
        print(f"stratified sample: {args.frac:.1%} per .label file (seed={args.seed})")
    index, idx_stats = load_label_index(args.root, keywords, frac=args.frac, seed=args.seed,
                                        decode_errors=args.decode_errors)
    print(f".label files scanned : {idx_stats['n_files']} "
          f"(excluded by keyword: {idx_stats['n_files_excluded']})")
    if idx_stats.get("n_decode_error") or idx_stats.get("n_read_error"):
        print(f".label files skipped : {idx_stats.get('n_decode_error', 0)} (not valid UTF-8), "
              f"{idx_stats.get('n_read_error', 0)} (read error)")
    if idx_stats.get("sampled"):
        print(f"valid entries (total): {idx_stats['n_before_sample']}")
        print(f"utterances sampled   : {len(index)} "
              f"(~{100*len(index)/max(idx_stats['n_before_sample'],1):.1f}% of valid)")
    else:
        print(f"utterances indexed   : {len(index)}")
    print(f"entries excluded     : {idx_stats['n_excluded_entries']}")
    print(f"malformed/empty lines: {idx_stats['n_bad_lines']}")
    if not index:
        print("\n[FAIL] no usable utterances found — the dataloader would see 0 samples "
              "(this is the 'num_samples=0' cause). Check --root and the .label format "
              "'<audio_path><whitespace><transcript>'.")
        sys.exit(1)

    resolver = PathResolver(args.root, audio_root=args.audio_root, max_up=args.max_up)

    # ---- 2) decide which entries to fully decode ----
    # --frac already subsampled the index, so fully decode all of the sampled set.
    decode_all = args.check_all or (args.frac is not None)
    if decode_all:
        decode_set = set(range(len(index)))
    else:
        rnd = random.Random(args.seed)
        k = min(args.max_check, len(index))
        decode_set = set(rnd.sample(range(len(index)), k))
    report_path = None if args.no_report else args.report
    print(f"\n========== AUDIO CHECK ==========")
    print(f"header check: ALL {len(index)} entries | full decode: {len(decode_set)} entries "
          f"({'all' if decode_all else 'random sample'})")
    if report_path:
        print(f"problem report -> {report_path}")

    # ---- 3) run checks (threaded, streaming aggregation) ----
    agg = Aggregator(args.ctx_len, report_path=report_path, examples=args.examples,
                     deep_cap=max(args.show, 200))
    run_checks(index, resolver, args.ctx_len, decode_set, args.workers, agg)
    agg.close()

    # ---- 4) summary + 5) problem examples ----
    agg.print_summary()
    agg.print_examples()

    # ---- 6) deep tokenization / label-extraction check ----
    n_mismatch = deep_tokenization_check(agg.deep_samples, args.ctx_len, args.show)

    # ---- 7) verdict ----
    by_status = agg.status
    n_problems = (by_status.get("missing", 0) + by_status.get("decode_fail", 0)
                  + by_status.get("too_long", 0))
    passed = (n_problems == 0) and (n_mismatch in (None, 0))
    print(f"\n========== VERDICT ==========")
    if report_path and (n_problems or by_status.get("warn", 0)):
        print(f"full problem list ({agg.n_problems_written} entries): {report_path}")
    if passed:
        print("PASS — index loads, audio decodes, and placeholder/label extraction aligns.")
        code = 0
    else:
        print(f"ISSUES FOUND — hard problems={n_problems}"
              + (f", alignment mismatches={n_mismatch}" if n_mismatch else "")
              + f", warnings={by_status.get('warn', 0)}")
        code = 1 if args.strict else 0

    # ---- 8) persistent summary file (so it doesn't scroll away on huge runs) ----
    if not args.no_summary:
        summary = {
            "root": args.root, "ctx_len": args.ctx_len, "frac": args.frac,
            "audio_root": args.audio_root, "decode_errors": args.decode_errors,
            "label_files_scanned": idx_stats.get("n_files"),
            "label_files_skipped_bad_utf8": idx_stats.get("n_decode_error"),
            "valid_entries_total": idx_stats.get("n_before_sample"),
            "entries_indexed": len(index),
            "entries_checked": len(decode_set),
            "deep_alignment_mismatches": n_mismatch,
            "hard_problems": n_problems,
            "verdict": "PASS" if passed else "ISSUES",
            **agg.to_dict(),
        }
        try:
            with open(args.summary, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nsummary written -> {args.summary}")
        except OSError as e:
            print(f"[warn] could not write summary {args.summary}: {e}")
    sys.exit(code)


if __name__ == "__main__":
    main()
