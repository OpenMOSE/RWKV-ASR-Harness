# -*- coding: utf-8 -*-
"""Gradio web UI for Japanese ASR inference (WavLM -> SpeechProjector -> RWKV7).

Usage:
    ASR_MODEL=out/asr-wavlm-rwkv1b5-stage2/rwkv-step-7400 python audio_web.py
The model path may be given with or without the .pth suffix.
"""

import os

import gradio as gr
import librosa
import numpy as np

from infer.asr_infer import ASRInfer

MODEL_PATH = os.environ.get("ASR_MODEL", "out/asr-wavlm-rwkv1b5-stage2/rwkv-step-8800")
ENCODER_PATH = os.environ.get("ASR_ENCODER", "microsoft/wavlm-large")
N_LAYER = int(os.environ.get("ASR_N_LAYER", 24))
N_EMBD = int(os.environ.get("ASR_N_EMBD", 2048))

print(f"Loading ASR model from {MODEL_PATH} ...")
model = ASRInfer(MODEL_PATH, encoder_path=ENCODER_PATH, n_layer=N_LAYER, n_embd=N_EMBD)
print("ASR model ready.")


def transcribe(audio):
    if audio is None:
        return "音声が入力されていません。"
    sr, data = audio

    # int PCM -> float32 [-1, 1]
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    data = data.astype(np.float32)

    # stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # resample to 16 kHz
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    return model.transcribe(data)


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="音声 (録音 or アップロード)"),
    outputs=gr.Textbox(label="認識結果"),
    title="ModRWKV ASR  (WavLM-large → RWKV7-1.5B)",
    description="日本語音声認識の推論テスト。録音またはwavをアップロードして実行してください。",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
