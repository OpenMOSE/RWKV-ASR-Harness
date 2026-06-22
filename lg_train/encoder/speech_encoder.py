import torch
import torch.nn as nn

from transformers import AutoModel
try:
    from transformers import Wav2Vec2FeatureExtractor
except Exception:  # pragma: no cover
    Wav2Vec2FeatureExtractor = None
from transformers import AutoFeatureExtractor


# ---------------------------------------------------------------------------
# Token-length helpers (must stay in sync with the conv stack below + the
# SpeechProjector in lg_train/projector/modules.py).  These let the dataset
# pre-compute the number of <|image_pad|> placeholder tokens for an audio clip
# WITHOUT instantiating the (heavy) WavLM backbone in dataloader workers.
# ---------------------------------------------------------------------------

# wav2vec2 / WavLM feature-extractor conv stack (kernel, stride per layer).
# These are the defaults for microsoft/wavlm-* and facebook/wav2vec2-* models.
WAVLM_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)
WAVLM_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)


def wavlm_feat_len(num_samples: int,
                   kernels=WAVLM_CONV_KERNEL,
                   strides=WAVLM_CONV_STRIDE) -> int:
    """Backbone output frame count for a 16 kHz waveform of `num_samples`."""
    n = int(num_samples)
    for k, s in zip(kernels, strides):
        n = (n - k) // s + 1
    return max(n, 1)


def speech_token_len(num_samples: int) -> int:
    """Final LLM token count after the SpeechProjector Conv1d(k=3, s=2, p=2)."""
    t = wavlm_feat_len(num_samples)
    # Conv1d output length: floor((L + 2*pad - kernel)/stride) + 1, pad=2,k=3,s=2
    return (t + 2 * 2 - 3) // 2 + 1


class SpeechEncoder(nn.Module):
    """Frozen self-supervised speech backbone (WavLM / wav2vec2 / HuBERT).

    Returns raw last_hidden_state; the trainable projection/down-sampling is
    handled by the SpeechProjector registered for this modality, mirroring the
    SigLIP encoder + VisualAdapter split used for vision.
    """

    def __init__(self, encoder_path, project_dim=None, device="cuda", **kwargs):
        super().__init__()
        self.device = device
        self.sampling_rate = 16000

        # WavLM ships only a feature extractor (no tokenizer) -> AutoProcessor
        # raises, so load the feature extractor directly.
        fe_cls = Wav2Vec2FeatureExtractor or AutoFeatureExtractor
        try:
            self.processor = fe_cls.from_pretrained(encoder_path)
        except Exception:
            self.processor = AutoFeatureExtractor.from_pretrained(encoder_path)

        self.model = AutoModel.from_pretrained(encoder_path)
        self.model.eval()
        self.encoder_dim = self.model.config.hidden_size  # WavLM-large -> 1024

    @torch.no_grad()
    def forward(self, x):
        """x: a single 1-D waveform (np.ndarray / list / tensor) at 16 kHz."""
        param = next(self.model.parameters())
        fe = self.processor(
            x, return_tensors="pt", sampling_rate=self.sampling_rate
        )
        input_values = fe["input_values"].to(param.device, dtype=torch.bfloat16)
        out = self.model(input_values).last_hidden_state  # (1, T, encoder_dim)
        return out
