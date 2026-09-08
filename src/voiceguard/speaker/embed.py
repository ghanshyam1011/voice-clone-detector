"""Speaker embedding via a pretrained x-vector model.

Default: ``microsoft/wavlm-base-plus-sv`` through HuggingFace transformers
(already a dependency — no new package). The model is downloaded once on
first use and cached; nothing happens at import time.

To swap the backend (e.g. SpeechBrain ECAPA, Resemblyzer) keep the
``embed(wave) -> np.ndarray`` contract and only this file changes.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "microsoft/wavlm-base-plus-sv"
SAMPLE_RATE = 16000


def _repair_pos_conv(model, model_name: str) -> None:
    """The SV checkpoints store the positional-conv weight-norm as
    ``weight_g`` / ``weight_v``; transformers >= 4.40 uses the newer
    ``parametrizations`` API and silently leaves those two tensors random
    (which noticeably hurts speaker separation). Copy them across."""
    import torch
    from huggingface_hub import hf_hub_download

    conv = model.wavlm.encoder.pos_conv_embed.conv
    param = getattr(conv, "parametrizations", None)
    if param is None:  # old API loaded it fine
        return
    try:
        sd = torch.load(hf_hub_download(model_name, "pytorch_model.bin"), map_location="cpu")
        g = sd["wavlm.encoder.pos_conv_embed.conv.weight_g"]
        v = sd["wavlm.encoder.pos_conv_embed.conv.weight_v"]
        with torch.no_grad():
            param.weight.original0.copy_(g)
            param.weight.original1.copy_(v)
    except Exception:  # noqa: BLE001 -- fall back to as-loaded weights
        pass


class SpeakerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device=None):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._fe = None

    # -- lazy load -----------------------------------------------------
    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._device = self._device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._fe = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        model = WavLMForXVector.from_pretrained(self.model_name)
        _repair_pos_conv(model, self.model_name)
        self._model = model.to(self._device).eval()

    def warm(self) -> None:
        """Trigger the one-time download / load ahead of the first request."""
        self._ensure()

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        self._ensure()
        return int(self._model.config.xvector_output_dim)

    # -- embed -------------------------------------------------------
    def embed(self, wave: np.ndarray) -> np.ndarray:
        """Mono 16 kHz float32 waveform -> L2-normalised embedding.

        Longer input is more reliable — 6 s+ for enrolment, and the
        caller should score against as long a window as it can spare.
        """
        self._ensure()
        import torch

        w = np.ascontiguousarray(np.asarray(wave, dtype=np.float32).reshape(-1))
        batch = self._fe(w, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        iv = batch.input_values.to(self._device)
        am = batch.get("attention_mask")
        am = am.to(self._device) if am is not None else None
        with torch.no_grad():
            emb = self._model(iv, attention_mask=am).embeddings[0].float().cpu().numpy()
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))
