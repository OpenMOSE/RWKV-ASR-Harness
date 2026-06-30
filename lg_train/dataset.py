import os
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
import jsonlines
import librosa
from .utils import *
os.environ["HF_DATASETS_CACHE"] = "/DATA/disk0/hf"

import PIL.PngImagePlugin
# 增加MAX_TEXT_CHUNK的大小，默认是1MB，可以设置为更大的值，例如10MB
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024
from .prepare.custom_transformers import get_image_processor
class WorldDataset(Dataset):
    def __init__(self, args, processor=None):
        """
        通用多模态数据集：
        支持 data_type = ['hf', 'img', 'arrow', 'jsonl', 'wav', 'state']
        """
        self.args = args
        self.processor = processor
        self.data_type = args.data_type
        self.debug_data = int(getattr(args, "debug_data", 0) or 0)

        # --- 1. 加载数据 ---
        if args.data_type == 'hf':
            self.data = self._load_hf_dataset(args.data_file)
        elif args.data_type == 'arrow':
            self.data = self._load_arrow_dataset(args.data_file)
        elif args.data_type in ['img', 'state']:
            self.data = self._load_vision_text(args.data_file)
            if hasattr(args, "copy"):
                self.data = self.data * args.copy
        elif args.data_type == 'jsonl':
            with jsonlines.open(args.data_file) as f:
                self.data = list(f)
        elif args.data_type == 'wav':
            with jsonlines.open(f'{args.data_file}/answer.jsonl') as f:
                self.data = list(f)
        elif args.data_type == 'asr':
            # Pre-materialized speech dataset (see prepare/build_reazon_subset.py):
            # columns = audio_bytes (binary) | transcription (str) | token_len (int)
            self.data = load_from_disk(args.data_file)
        elif args.data_type == 'label':
            # On-the-fly ASR from scattered *.label files under a root folder.
            # Each .label line: "<rel/audio/path><space|TAB><transcript...>"
            # Audio is decoded lazily in __getitem__ (no preprocessing).
            self._label_root = os.path.normpath(args.data_file)
            self._audio_root = getattr(args, "audio_root", None) or None
            if self._audio_root:
                self._audio_root = os.path.normpath(self._audio_root)
            self._resolve_max_up = int(getattr(args, "resolve_max_up", 2) or 0)
            self._path_cache = {}         # label_dir -> winning (base, transform)
            # skip visibility: silent data loss is the #1 cause of "overfit on
            # 3TB after a few steps" -> count & periodically log what's dropped.
            self._label_reads = 0
            self._label_skips = 0
            self._skip_log_every = 2000
            self.data = self._load_label_index(args.data_file)
        elif args.data_type == 'autoimg':
            self.data = self._load_hf_dataset(args.data_file)
            self.image_processor = get_image_processor(768, 384, True)
        else:
            raise ValueError(f"Unsupported data_type: {args.data_type}")
        print(f"Loaded {len(self.data)} samples for {args.data_type} dataset.")

        # --- 1b. shuffle the index (default ON; --data_shuffle 0 to disable) ---
        # Done BEFORE the epoch_steps trim so the kept subset is a random sample of
        # the whole dataset (not just the first N in scan order). A FIXED seed is
        # used so every DDP rank produces the same order (DistributedSampler then
        # splits a consistent index). Per-epoch access order is still reshuffled by
        # the DataLoader's sampler.
        if int(getattr(args, "data_shuffle", 1) or 0):
            seed = args.random_seed if getattr(args, "random_seed", -1) >= 0 else 42
            if isinstance(self.data, list):
                random.Random(seed).shuffle(self.data)
            elif hasattr(self.data, "shuffle"):   # HF datasets.Dataset (arrow/asr/hf)
                self.data = self.data.shuffle(seed=seed)
            print(f"[shuffle] index shuffled before trim (seed={seed})")

        data_nums = len(self.data)
        if args.epoch_steps < data_nums and args.epoch_steps>0:
            if isinstance(self.data, list):
                self.data = self.data[:args.epoch_steps]
            else:
                self.data = self.data.select(range(args.epoch_steps))
            print(f"Trimmed to {len(self.data)} samples for epoch_steps {args.epoch_steps}.")
    # ------------------------------
    # 数据加载函数
    # ------------------------------

    def _load_hf_dataset(self, path):
        """加载 Hugging Face 格式数据"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        datasets = []
        for subdir in subdirs:
            try:
                ds = load_dataset(subdir, split="train")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ 跳过无效数据目录: {subdir}, 原因: {e}")

        if datasets:
            return concatenate_datasets(datasets)
        else:
            # 说明当前目录本身是dataset根目录
            return load_dataset(path, split="train",cache_dir="/DATA/disk0/hf")

    def _load_arrow_dataset(self, path):
        """加载 Arrow 格式（支持多个子目录）"""
        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        if subdirs:
            datasets = [load_from_disk(sd) for sd in subdirs]
            return concatenate_datasets(datasets)
        return load_from_disk(path)

    def _load_vision_text(self, path):
        """可根据项目自定义 load_vision_text"""
        # 假设格式 [{"image": "xxx.jpg", "conversations": [...]}, ...]
        return load_vision_text(path)
 

    # ------------------------------
    # Dataset 必须方法
    # ------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.data[idx]
                break
            except FileNotFoundError:
                idx = (idx + 1) % len(self.data)
        t = self.data_type
        used_idx, used_sample = idx, sample

        if t == 'img':
            out = self._process_img(sample)
        elif t == 'arrow':
            out = self._process_arrow(sample)
        elif t == 'hf':
            out = self._process_hf(sample)
        elif t == 'wav':
            return self._process_wav(sample)
        elif t == 'asr':
            out = self._process_asr(sample)
        elif t == 'label':
            # on-the-fly: some audio may be missing/corrupt/too-long -> skip to next.
            # Skips are COUNTED and periodically logged: a high skip rate means the
            # effective dataset is far smaller than len(self.data) (silent data loss
            # -> overfitting). Watch for "[label] skip rate" lines in the logs.
            out = None
            last_err = ""
            for _ in range(64):
                try:
                    out = self._process_label(self.data[idx])
                    used_idx, used_sample = idx, self.data[idx]
                    self._label_reads += 1
                    break
                except Exception as e:
                    self._label_skips += 1
                    last_err = f"{type(e).__name__}: {str(e)[:100]}"
                    if self._label_skips % self._skip_log_every == 0:
                        total = self._label_reads + self._label_skips
                        rate = 100.0 * self._label_skips / max(total, 1)
                        print(f"[label] skip rate {rate:.1f}% "
                              f"({self._label_skips} skipped / {self._label_reads} ok) "
                              f"| last: {last_err}", flush=True)
                    idx = (idx + 1) % len(self.data)
            if out is None:
                raise RuntimeError(
                    f"too many unreadable audio entries in 'label' dataset "
                    f"(64 consecutive skips; last error: {last_err}). "
                    f"Check audio paths / --audio_root with scripts/check_label.sh")
        elif t == 'jsonl':
            return sample
        elif t == 'autoimg':
            out = self._process_autoimg(sample)
        else:
            raise ValueError(f"Unsupported data_type in __getitem__: {t}")

        # debug: attach a small per-sample identifier so the trainer can print,
        # step by step, exactly which dataset entries went into each batch.
        if self.debug_data and isinstance(out, tuple) and len(out) == 3:
            return (*out, self._meta(used_idx, used_sample))
        return out

    def _meta(self, idx, sample):
        """Short human-readable id of a dataset entry (for --debug_data)."""
        try:
            if self.data_type == 'label':
                _label_dir, p, txt = sample
                return f"#{idx}:{os.path.basename(p)}|{str(txt)[:24]}"
            if isinstance(sample, dict):
                txt = sample.get('transcription') or sample.get('text') or sample.get('texts') or ''
                return f"#{idx}:{str(txt)[:24]}"
        except Exception:
            pass
        return f"#{idx}"

    # ------------------------------
    # 各类型处理函数
    # ------------------------------

    def _process_img(self, sample):
        images = sample['image']
        if not isinstance(images, list):
            images = [images]
        images = [Image.open(os.path.join(self.args.data_file, "data", img)).convert("RGB") for img in images]

        texts = sample["conversations"]
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        return  images, input_ids, label_ids

    def _process_arrow(self, sample):
        images = [img.convert("RGB") for img in sample["images"]][:3]  # ctx_len limit 3 image
        texts = convert_texts_to_conversations(sample["texts"])
        source = sample['source']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images), source=source)
        return  images, input_ids, label_ids
    def _process_hf(self, sample):
        if 'image' in sample:
            image = sample['image']
            images=[]
            if not isinstance(image, list) and image is not None:
                images = [image]
            images = [img.convert("RGB") for img in images]
        if 'images' in sample:
            images = sample['images']

        images = [img.convert("RGB") for img in images][:3]
        texts = convert_texts_to_conversations(sample["texts"])
        # texts = sample['conversations']
        while texts[0]["value"].startswith("<image>"):
            texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
        for i in range(len(images)):
                texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=[576]*len(images))
        images = images if images else None
        return  images, input_ids, label_ids
    def _process_autoimg(self, sample):
        if 'image' in sample:
            image = sample['image']
            images=[]
            if not isinstance(image, list) and image is not None:
                images = [image]
            images = [img.convert("RGB") for img in images]
        if 'images' in sample:
            images = sample['images']
        images = images[:6]

        texts = convert_texts_to_conversations(sample["texts"])
        texts = placeholder_token(texts, len(images))
        image_token_length = []
        pixel_values = []

        for image in images:
            pixel_value,_ = self.image_processor(image.convert("RGB"))
            b,_,_,_ = pixel_value.shape
            image_token_length.append(b*115)
            pixel_values.append(pixel_value)
        pixel_values = torch.cat(pixel_values, dim=0) if pixel_values else None

        input_ids, label_ids = process_vision_text(texts, max_length=self.args.ctx_len, image_token_length=image_token_length)
        return pixel_values, input_ids, label_ids
    def _process_wav(self, sample):
        audio = librosa.load(sample["path"], sr=16000)[0]
        return {"audio": audio, "text": sample.get("text", "")}

    # ------------------------------
    # on-the-fly ASR from *.label folders
    # ------------------------------
    def _load_label_index(self, root):
        """Recursively scan `root` for *.label files and build a
        (label_dir, audio_path, text) index. Each line:
        '<audio_path><whitespace><transcript>'. Only the text index is held in
        memory (label_dir is interned -> shared across a file's entries); audio
        is decoded on demand in _process_label. Keeping label_dir lets
        _resolve_audio_path try paths relative to the .label file's own folder,
        which is the common real-world layout.

        Non-UTF-8 .label files are skipped with a warning (not fatal).
        --label_exclude "misc,noise" skips any entry whose audio path (or its
        source .label file path) contains one of the comma-separated keywords
        (case-insensitive substring match). Useful to drop e.g. a 'misc/' folder."""
        import sys
        excl = (getattr(self.args, "label_exclude", "") or "")
        keywords = [k.strip().lower() for k in excl.split(",") if k.strip()]
        n_skipped = 0
        n_decode_error = 0

        index = []
        n_files = 0
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".label"):
                    continue
                fpath = os.path.join(dirpath, fn)
                # skip an entire .label file whose path matches an excluded keyword
                if keywords and any(kw in fpath.lower() for kw in keywords):
                    continue
                n_files += 1
                label_dir = sys.intern(dirpath)   # shared by all entries of this file
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.rstrip("\n").rstrip("\r")
                            if not line:
                                continue
                            parts = line.split(None, 1)   # split on first whitespace (space or TAB)
                            if len(parts) < 2:
                                continue
                            apath, text = parts[0], parts[1].strip()
                            # skip a single entry whose audio path matches a keyword
                            if keywords and any(kw in apath.lower() for kw in keywords):
                                n_skipped += 1
                                continue
                            if text:
                                index.append((label_dir, apath, text))
                except UnicodeDecodeError as e:
                    n_decode_error += 1
                    print(f"⚠️ skip .label (not valid UTF-8: {e}): {fpath}")
                except Exception as e:
                    print(f"⚠️ skip label file {fpath}: {e}")
        if keywords:
            print(f"[label] exclude keywords={keywords} -> skipped {n_skipped} entries (+ matching .label files)")
        if n_decode_error:
            print(f"[label] skipped {n_decode_error} .label files (not valid UTF-8)")
        print(f"[label] scanned {n_files} .label files -> {len(index)} utterances under {root}")
        return index

    def _resolve_audio_path(self, rel, label_dir):
        """Resolve a label entry's audio path, robust to mixed conventions:
        absolute / relative to the .label file's own dir / relative to the
        dataset root (or its parent) / off-by-one (an extra or missing leading
        directory). Tries a prioritized list of (base, transform) candidates and
        CACHES the winning scheme per label_dir, so only the first entry of each
        .label file pays the search cost.  Returns (path, found_bool)."""
        if os.path.isabs(rel):
            return rel, os.path.exists(rel)

        # fast path: reuse the scheme that already worked for this .label dir
        plan = self._path_cache.get(label_dir)
        if plan is not None:
            base, strip1 = plan
            cand = self._apply_scheme(base, strip1, rel)
            if cand and os.path.exists(cand):
                return cand, True

        # candidate base dirs, most-likely first
        root = self._label_root
        bases = []
        if self._audio_root:
            bases.append(self._audio_root)
        bases.append(label_dir)                       # relative to the .label file
        d = label_dir
        for _ in range(self._resolve_max_up):          # walk up (missing-dir off-by-one)
            d = os.path.dirname(d)
            if d:
                bases.append(d)
        bases.append(root)
        bases.append(os.path.dirname(root))
        seen = set()
        for base in bases:
            if not base or base in seen:
                continue
            seen.add(base)
            for strip1 in (False, True):               # asis / drop extra leading dir
                cand = self._apply_scheme(base, strip1, rel)
                if cand and os.path.exists(cand):
                    self._path_cache[label_dir] = (base, strip1)
                    return cand, True
        return os.path.join(label_dir, rel), False     # best-guess; caller will skip

    @staticmethod
    def _apply_scheme(base, strip1, rel):
        if strip1:
            parts = rel.split(os.sep)
            if len(parts) <= 1:
                return None
            rel = os.sep.join(parts[1:])
        return os.path.join(base, rel)

    def _process_label(self, sample):
        """(label_dir, audio_path, transcript) -> (waveform, input_ids, label_ids)."""
        import numpy as np
        import soundfile as sf
        from .encoder.speech_encoder import speech_token_len

        label_dir, apath, text = sample
        path, found = self._resolve_audio_path(apath, label_dir)
        if not found:
            raise FileNotFoundError(f"unresolved audio path: {apath}")
        try:
            wav, sr = sf.read(path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        except Exception:
            # formats soundfile can't open (mp3 etc.) -> librosa (also resamples)
            wav = librosa.load(path, sr=16000)[0]

        token_len = speech_token_len(len(wav))
        # guard: drop clips whose audio tokens leave no room for the transcript
        # (otherwise build_inputs_and_labels truncates all labels -> NaN loss)
        if token_len <= 0 or token_len > self.args.ctx_len - 32:
            raise ValueError("audio too long for ctx_len")

        conversations = [
            {"from": "user", "value": "<|image_pad|>" * int(token_len)},
            {"from": "assistant", "value": text},
        ]
        input_ids, label_ids = build_inputs_and_labels(
            conversations, pipeline, self.args.ctx_len, -100
        )
        return wav, input_ids, label_ids

    def _process_asr(self, sample):
        """Speech-to-text sample -> (waveform, input_ids, label_ids).

        Builds the same <|image_pad|> placeholder / masked-scatter contract as
        the vision path: the number of placeholders equals the SpeechProjector
        output token count for this clip, so encoder features align 1:1.
        """
        import io
        import numpy as np
        import soundfile as sf
        from .encoder.speech_encoder import speech_token_len

        wav, sr = sf.read(io.BytesIO(sample["audio_bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        token_len = sample.get("token_len")
        if not token_len:
            token_len = speech_token_len(len(wav))

        conversations = [
            {"from": "user", "value": "<|image_pad|>" * int(token_len)},
            {"from": "assistant", "value": sample["transcription"]},
        ]
        input_ids, label_ids = build_inputs_and_labels(
            conversations, pipeline, self.args.ctx_len, -100
        )
        return wav, input_ids, label_ids

def placeholder_token(texts, img_nums):
    while texts[0]["value"].startswith("<image>"):
        texts[0]["value"] = texts[0]["value"].replace("<image>", "", 1)
    for i in range(img_nums):
            texts[0]["value"] = "<|placeholder|>" + texts[0]["value"]
    return texts

import lightning as L
from torch.utils.data import DataLoader


def world_collate_fn(batch):
    """Module-level collate so it stays picklable under multi-GPU / spawned
    dataloader workers (a nested local function raises
    'can't pickle local object ...train_dataloader.<locals>.custom_collate_fn')."""
    cols = list(zip(*batch))
    signs, inputs_ids, labels = cols[0], cols[1], cols[2]
    all_images = list(signs)
    inputs_ids = torch.stack(inputs_ids, dim=0)
    labels = torch.stack(labels, dim=0)
    if len(cols) > 3:   # --debug_data: 4th element = per-sample meta ids
        return all_images, inputs_ids, labels, list(cols[3])
    return all_images, inputs_ids, labels


class WorldDataModule(L.LightningDataModule):
    def __init__(self, args, processor=None):
        super().__init__()
        self.args = args
        self.processor = processor

    def setup(self, stage=None):
        self.train_dataset = WorldDataset(self.args, self.processor)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.micro_bsz,
            shuffle=True,    # Lightning 自动替换成 DistributedSampler
            collate_fn=world_collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
