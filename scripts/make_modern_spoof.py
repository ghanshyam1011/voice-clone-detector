"""Generate a small modern-TTS spoof set to fine-tune the detector on
current-generation synthesis (AASIST was trained on 2019-era attacks).

    python scripts/make_modern_spoof.py --n 400

Two neural TTS families:
  * Microsoft SpeechT5 + HiFi-GAN, driven by ~7900 CMU-Arctic speaker
    x-vectors -> lots of distinct voices;
  * Meta MMS-TTS (VITS), English, stochastic duration -> a 2nd architecture.
Half of each is pushed through a random telephony codec. Output: 16 kHz
wav in data/modern_spoof/ + a manifest. Real clips for the other class
come from ASVspoof bonafide at fine-tune time.
"""

# ruff: noqa: E402
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import random

import numpy as np
import pandas as pd
import soundfile as sf
import soxr

from voiceguard.config import REPO_ROOT

SR = 16000
OUT = REPO_ROOT / "data" / "modern_spoof"

_LINES = [
    "Hello, I am calling about my bank account and a recent transaction.",
    "I need to transfer seventy five thousand rupees to a new account today.",
    "Please verify my identity so we can proceed with the payment.",
    "My card was blocked and I want to reactivate it immediately.",
    "Can you confirm the balance and the last three transactions.",
    "There is an urgent issue with my loan and I need help right now.",
    "Kindly update my registered mobile number and email address.",
    "I authorise this payment to the beneficiary I just added.",
    "The amount should be credited before the end of the day please.",
    "I did not receive the one time password on my phone.",
    "Transfer the money now, it is an emergency, do not delay.",
    "This is a genuine request from the account holder, please act fast.",
    "Good morning, I would like to check the status of my recent application.",
    "There has been suspicious activity and I want to secure my account.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    import torch

    from voiceguard.audio.codec import CODECS, codec_roundtrip

    def _maybe_codec(w: np.ndarray) -> tuple[np.ndarray, str]:
        if random.random() < 0.5:
            c = random.choice(list(CODECS))
            return codec_roundtrip(w, SR, c).astype(np.float32), c
        return w, "clean"

    def _save(w: np.ndarray, fname: str, system: str, codec: str) -> None:
        w = (0.9 * w / (np.max(np.abs(w)) or 1.0)).astype(np.float32)
        sf.write(OUT / fname, w, SR, subtype="PCM_16")
        rows.append(
            {
                "path": f"data/modern_spoof/{fname}",
                "system": system,
                "codec": codec,
                "label": "spoof",
            }
        )

    rows: list[dict] = []

    # --- SpeechT5 + HiFi-GAN, random speaker vectors (no external dataset) ---
    try:
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

        proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        t5 = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").eval()
        voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
        gen = torch.Generator().manual_seed(args.seed)
        n_t5 = int(args.n * 0.5)
        for i in range(n_t5):
            spk = torch.randn(1, 512, generator=gen)
            spk = spk / spk.norm()
            txt = random.choice(_LINES)
            with torch.no_grad():
                w = t5.generate_speech(
                    proc(text=txt, return_tensors="pt")["input_ids"], spk, vocoder=voc
                ).numpy()
            if len(w) < SR:  # random vectors sometimes collapse; skip the dud
                continue
            w, codec = _maybe_codec(w.astype(np.float32))
            _save(w, f"speecht5_{i:03d}.wav", "speecht5", codec)
        print(f"speecht5: {sum(r['system'] == 'speecht5' for r in rows)} clips", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"skip speecht5: {exc}", flush=True)

    # --- MMS-TTS VITS (English), stochastic duration + varied noise/rate ---
    try:
        from transformers import AutoTokenizer, VitsModel

        m = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
        tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
        n_vits = args.n - sum(r["system"] == "speecht5" for r in rows)
        for i in range(max(n_vits, 0)):
            m.noise_scale = random.uniform(0.4, 1.0)
            m.noise_scale_duration = random.uniform(0.5, 1.0)
            m.speaking_rate = random.uniform(0.8, 1.3)
            txt = random.choice(_LINES)
            with torch.no_grad():
                w = m(**tok(txt, return_tensors="pt")).waveform[0].numpy()
            w = soxr.resample(w, m.config.sampling_rate, SR).astype(np.float32)
            w, codec = _maybe_codec(w)
            _save(w, f"vits_eng_{i:03d}.wav", "vits_eng", codec)
        nv = sum(r["system"] == "vits_eng" for r in rows)
        print(f"vits (mms-tts-eng): {nv} clips", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"skip vits: {exc}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manifest.csv", index=False)
    print(f"\n{len(df)} modern-spoof clips -> data/modern_spoof/manifest.csv")
    if len(df):
        print(df.groupby(["system", "codec"]).size().to_string())


if __name__ == "__main__":
    main()
