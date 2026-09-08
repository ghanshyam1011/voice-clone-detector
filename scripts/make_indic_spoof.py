"""Generate Indian-language synthetic-speech attacks for the IndicCall-Eval
pilot: MMS-TTS (VITS) in English + Hindi / Tamil / Bengali / Marathi,
bank-call phrasing, half pushed through a telephony codec.

    python scripts/make_indic_spoof.py --per-lang 40

Writes 16 kHz wav to data/indic_eval/spoof/<lang>/ + a manifest.

Real bona-fide speech is NOT generated or scraped — it is a small
consented pilot the team records; see docs/indic_eval.md. Drop those
clips in data/indic_eval/real/<lang>/ and scripts/eval_indic.py picks
them up.
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
OUT = REPO_ROOT / "data" / "indic_eval"

# short bank-call lines, native script (uroman not needed with the .long() cast)
_LINES = {
    "eng": [
        "I need to transfer seventy five thousand rupees to a new account right now.",
        "Please verify my identity so we can process the payment quickly.",
        "There is an urgent problem with my account and I need help immediately.",
        "Kindly update my registered mobile number and confirm the change.",
    ],
    "hin": [
        "मुझे पचहत्तर हज़ार रुपये एक नए खाते में तुरंत भेजने हैं।",
        "कृपया मेरी पहचान सत्यापित करें ताकि भुगतान जल्दी हो सके।",
        "मेरे खाते में एक ज़रूरी समस्या है और मुझे अभी मदद चाहिए।",
        "कृपया मेरा पंजीकृत मोबाइल नंबर तुरंत बदल दीजिए।",
    ],
    "tam": [
        "எழுபத்தி ஐந்தாயிரம் ரூபாயை புதிய கணக்குக்கு இப்போதே அனுப்ப வேண்டும்.",
        "பணம் விரைவாக செலுத்த என் அடையாளத்தை சரிபார்க்கவும்.",
        "என் கணக்கில் அவசர பிரச்சினை உள்ளது, உடனே உதவி தேவை.",
        "என் பதிவு செய்யப்பட்ட மொபைல் எண்ணை மாற்றவும்.",
    ],
    "ben": [
        "আমাকে পঁচাত্তর হাজার টাকা এখনই একটি নতুন অ্যাকাউন্টে পাঠাতে হবে।",
        "অনুগ্রহ করে আমার পরিচয় যাচাই করুন যাতে দ্রুত পেমেন্ট হয়।",
        "আমার অ্যাকাউন্টে জরুরি সমস্যা হয়েছে, এখনই সাহায্য দরকার।",
        "দয়া করে আমার নিবন্ধিত মোবাইল নম্বর পরিবর্তন করুন।",
    ],
    "mar": [
        "मला पंच्याहत्तर हजार रुपये नवीन खात्यात आत्ताच पाठवायचे आहेत.",
        "कृपया माझी ओळख पडताळा म्हणजे पैसे लवकर पाठवता येतील.",
        "माझ्या खात्यात तातडीची अडचण आहे, मला लगेच मदत हवी आहे.",
        "कृपया माझा नोंदणीकृत मोबाइल नंबर बदला.",
    ],
}


def _tts(lang: str):
    from transformers import VitsModel, VitsTokenizer

    name = f"facebook/mms-tts-{lang}"
    return VitsModel.from_pretrained(name).eval(), VitsTokenizer.from_pretrained(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    import torch

    from voiceguard.audio.codec import CODECS, codec_roundtrip

    rows: list[dict] = []
    for lang, lines in _LINES.items():
        try:
            model, tok = _tts(lang)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {lang}: {exc}")
            continue
        d = OUT / "spoof" / lang
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_lang):
            model.noise_scale = random.uniform(0.5, 0.95)
            model.speaking_rate = random.uniform(0.85, 1.2)
            enc = tok(random.choice(lines), return_tensors="pt")
            with torch.no_grad():
                w = model(
                    input_ids=enc["input_ids"].long(),
                    attention_mask=enc["attention_mask"].long(),
                ).waveform[0].numpy()
            w = soxr.resample(w, model.config.sampling_rate, SR).astype(np.float32)
            codec = "clean"
            if random.random() < 0.5:
                codec = random.choice(list(CODECS))
                w = codec_roundtrip(w, SR, codec).astype(np.float32)
            w = (0.9 * w / (np.max(np.abs(w)) or 1.0)).astype(np.float32)
            fname = f"{lang}_{i:03d}.wav"
            sf.write(d / fname, w, SR, subtype="PCM_16")
            rows.append(
                {
                    "path": f"data/indic_eval/spoof/{lang}/{fname}",
                    "language": lang,
                    "codec": codec,
                    "label": "spoof",
                }
            )
        print(f"{lang}: {args.per_lang} clips", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "spoof_manifest.csv", index=False)
    print(f"\n{len(rows)} Indic spoof clips -> data/indic_eval/spoof_manifest.csv")


if __name__ == "__main__":
    main()
