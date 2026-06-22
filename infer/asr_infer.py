# -*- coding: utf-8 -*-
"""Speech-to-text inference for the WavLM -> SpeechProjector -> RWKV7 ASR model.

This reuses the *training* stack (lg_train.ModRWKV + the fla/triton WKV kernel),
so it matches the trained checkpoint exactly and runs on ROCm — unlike the old
BlinkDL `infer.rwkv` path which needs a separate CUDA/HIP kernel build.

Decoding is plain greedy autoregression with a full forward per step (the
training RWKV7 forward has no incremental state cache). That is fine for short
ASR outputs; the audio is encoded once and cached in the prompt embeddings.
"""

import os

# WKV op + model env (must be set before importing the RWKV modules)
os.environ.setdefault("WKV", "fla")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
os.environ.setdefault("RWKV_CTXLEN", "4096")
os.environ.setdefault("RWKV_TRAIN_TYPE", "")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_JIT_ON", "0")

from types import SimpleNamespace

import numpy as np
import torch

from lg_train.model import ModRWKV
from lg_train.utils import pipeline
from lg_train.encoder.speech_encoder import speech_token_len

IMAGE_TOKEN_ID = 65532   # <|image_pad|> placeholder (audio soft-tokens)
ROLE_END = 24            # \x17  -> end-of-turn / stop


class ASRInfer:
    def __init__(self, model_path, encoder_path="microsoft/wavlm-large",
                 n_layer=24, n_embd=2048, vocab_size=65536, head_size_a=64,
                 device="cuda", dtype=torch.bfloat16):
        if not model_path.endswith(".pth"):
            model_path = model_path + ".pth"
        self.device = device
        self.dtype = dtype
        self.pipeline = pipeline

        # peek the checkpoint to auto-detect LoRA ranks (so injection matches)
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        from lg_train.llm.rwkv7.lora import detect_lora_ranks
        lora_tmix, lora_ffn = detect_lora_ranks(sd)
        if lora_tmix or lora_ffn:
            print(f"[ASRInfer] detected LoRA in checkpoint: tmix_rank={lora_tmix} ffn_rank={lora_ffn}")

        args = SimpleNamespace(
            vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
            dim_att=n_embd, dim_ffn=n_embd * 4,   # ffn uses n_embd*4 internally
            head_size_a=head_size_a, head_size_divisor=8,
            ctx_len=int(os.environ["RWKV_CTXLEN"]),
            grad_cp=0, grad_cp_layers=0, my_testing="x070", dropout=0,
            encoder_path=encoder_path, encoder_type="speech",
            train_step=[], layerwise_lr=0, weight_decay=0,
            lora_tmix=lora_tmix, lora_ffn=lora_ffn,  # scaling restored from ckpt buffer
        )
        print(f"[ASRInfer] building model (L{n_layer} D{n_embd}) + encoder {encoder_path} ...")
        self.model = ModRWKV(args)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[ASRInfer] loaded {model_path} | missing={len(missing)} unexpected={len(unexpected)}")
        self.model = self.model.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def transcribe(self, wav, max_new_tokens=200, prompt_text=""):
        """wav: 1-D float32 waveform at 16 kHz. Returns the decoded transcription."""
        if wav is None or len(wav) == 0:
            return ""
        wav = np.asarray(wav, dtype=np.float32)

        # Build the same prompt as training: \x16User:<audio pads>\x17\x16Assistant:
        n_audio = speech_token_len(len(wav))
        ids = (self.pipeline.encode("\x16User:")
               + [IMAGE_TOKEN_ID] * n_audio
               + self.pipeline.encode("\x17\x16Assistant:" + prompt_text))
        input_ids = torch.tensor([ids], device=self.device)

        emb = self.model.get_input_embeddings()
        inputs_embeds = emb(input_ids)

        # encode audio once and scatter into the placeholder positions
        audio = self.model.proj(self.model.encoder(wav))
        audio = audio.reshape(-1, inputs_embeds.shape[-1]).to(inputs_embeds.dtype)
        mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, audio)
        inputs_embeds = inputs_embeds.masked_scatter(mask, audio)

        out_ids = []
        for _ in range(max_new_tokens):
            logits = self.model.llm(inputs_embeds=inputs_embeds)
            nxt = int(logits[0, -1].float().argmax().item())
            if nxt == ROLE_END or nxt == 0:
                break
            out_ids.append(nxt)
            nxt_emb = emb(torch.tensor([[nxt]], device=self.device)).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, nxt_emb], dim=1)

        return self.pipeline.decode(out_ids)
