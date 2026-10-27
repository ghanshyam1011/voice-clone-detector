"""Generate modern synthetic speech with KNOWN targets, for held-out evaluation and training.

Every earlier number used 2019-era attacks (ASVspoof A01-A19). A fraudster today clones
a victim's voice from a few seconds of audio with an open model. This script makes
those attacks, so the system can be measured on them instead of assumed to survive them.

    python scripts/make_modern_clones.py --gen knnvc    --split eval  --speakers 24
    python scripts/make_modern_clones.py --gen speecht5x --split eval --speakers 24
    python scripts/make_modern_clones.py --gen bark     --split eval  --n 120
    python scripts/make_modern_clones.py --gen mms      --split train --n 300

Nothing is downloaded here. Every model is loaded from a local copy that was reviewed
first (kNN-VC: models/external/knn-vc-code at commit c616845 + weights with
``weights_only=True``, never ``torch.hub``, which would fetch and run code unseen) or from
the Hugging Face cache with ``local_files_only`` -- a model that is not there is an error,
not a silent multi-gigabyte download.

Generators (all open weights, no terms to accept; the ones that need no cloning are
"untargeted": they read a script in someone else's voice -- an *impostor* that happens
to be synthetic):

  knnvc      kNN voice conversion (Baas et al., 2023): WavLM-Large features of a *source*
             utterance are replaced by their nearest neighbours in ~25 s of the target's
             speech, then vocoded. A true zero-shot clone of the target's timbre.
             ``--source tts`` converts a synthetic utterance (TTS -> clone, the common
             real-world pipeline); ``--source bona`` converts a real person's speech.
  speecht5x  Microsoft SpeechT5 + HiFi-GAN conditioned on the target's x-vector
             (speechbrain). A weaker, cheaper clone: what an attacker with no GPU does.
  mms        Meta MMS-TTS (VITS), English. Untargeted.
  bark       Suno Bark-small (a codec language model: a different generator family
             from all of the above). Untargeted.

Targets and references come from the ASVspoof 2019 LA manifest of ``--split`` and are
disjoint from the clips used to enrol that speaker and from their genuine test calls
(``voiceguard.eval.trials``), so a clone never sees the audio it is tested against.
Output: 16 kHz wav under data/modern/<gen>/ (audio is never committed) and a manifest
data/modern/manifest.csv (generator, family, target_speaker, split, path, ...).
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import random
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from tqdm import tqdm

from voiceguard.audio import load_wave
from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.eval.trials import choose_speakers, eligible_speakers, plan_speaker

SR = 16000
OUT = REPO_ROOT / "data" / "modern"
MANIFEST = OUT / "manifest.csv"
EXTERNAL = REPO_ROOT / "models" / "external"
KNN_CODE = EXTERNAL / "knn-vc-code"  # github.com/bshall/knn-vc @ c616845 (MIT), reviewed
KNN_WEIGHTS = EXTERNAL / "knn-vc-weights"  # the repo's v0.1 release assets
# fairseq's WavLM builds its conv stack with eval() on a string stored in the checkpoint;
# it must look like "[(512,10,5)] + [(512,3,2)] * 4", never like code
_CONV_LAYERS = re.compile(r"[\d\s\[\](),+*]+")
FAMILY = {"knnvc": "wavlm-knn-vc", "speecht5x": "speecht5", "mms": "vits", "bark": "codec-lm"}

TEXTS = [
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
    "I would like to raise the daily limit on my savings account.",
    "Please read me the last four digits of the card on file.",
    "My name is on the account, and I am travelling, so I cannot visit the branch.",
    "Can you reset my internet banking password over the phone.",
    "I want to close the fixed deposit and move the funds to my current account.",
    "The beneficiary details are correct, please complete the transfer now.",
    "I have already confirmed this twice, why is it taking so long.",
    "Please send the statement for the last six months to my new address.",
    "I am calling from abroad, so the number will look unfamiliar.",
    "I lost my phone yesterday and need to change my registered device.",
    "Please approve the payment, the supplier is waiting for the money.",
    "Could you tell me when the next instalment on my loan is due.",
    "I would like to add my wife as a nominee on this account.",
    "The cheque I deposited last week has not been credited yet.",
    "Please increase my credit limit, I have always paid on time.",
    "I do not recognise the last charge on my card, can you check it.",
    "Let us finish this quickly, I have a meeting in five minutes.",
    "I would like to speak to a manager about this transaction.",
]


def _write(w: np.ndarray, sr: int, path) -> None:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if sr != SR:
        w = soxr.resample(w, sr, SR).astype(np.float32)
    peak = float(np.max(np.abs(w))) or 1.0
    sf.write(path, (0.9 * w / peak).astype(np.float32), SR, subtype="PCM_16")


def _append(rows: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST)
        new = pd.concat([old[~old.path.isin(new.path)], new], ignore_index=True)
    new.to_csv(MANIFEST, index=False)


def _targets(args):
    """-> [(speaker, reference clip paths disjoint from enrolment and genuine calls)]"""
    cfg = load_config()
    df = load_manifest(resolve(cfg, "manifests") / f"asvspoof19_la_{args.split}.csv")
    bona, spoof = df[df.label == "bonafide"], df[df.label == "spoof"]
    ok = eligible_speakers(bona, spoof)
    out = []
    for s in choose_speakers(ok, args.speakers, args.seed):
        plan = plan_speaker(bona, spoof, ok, s, args.seed)
        used = set(plan.enrol) | {p for k, _, ps in plan.trials if k == "genuine" for p in ps}
        refs = [p for p in bona[bona.speaker_id == s].path if p not in used]
        out.append((s, refs[: args.ref_clips]))
    return out, bona, ok


def _cat(paths):
    return np.concatenate([load_wave(REPO_ROOT / p, SR) for p in paths])


# ------------------------------------------------------------------ generators


def run_speecht5x(args) -> list[dict]:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    local = str(EXTERNAL / "spkrec-xvect-voxceleb")  # fetched once, files checked (see docs)
    xv = EncoderClassifier.from_hparams(source=local, savedir=local, run_opts={"device": dev})
    offline = {"local_files_only": True}
    proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", **offline)
    tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", **offline)
    tts = tts.to(dev).eval()
    voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", **offline).to(dev).eval()
    targets, _, _ = _targets(args)
    d = OUT / "speecht5x"
    d.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(args.seed)
    for s, refs in tqdm(targets, desc="targets"):
        with torch.no_grad():
            e = xv.encode_batch(torch.from_numpy(_cat(refs)[: 20 * SR]).float().unsqueeze(0))
            e = torch.nn.functional.normalize(e, dim=2).squeeze(1).to(dev)
        for i in range(args.per_speaker):
            path = d / f"{s}_{i:02d}.wav"
            text = rng.choice(TEXTS)
            if not path.exists():
                with torch.no_grad():
                    ids = proc(text=text, return_tensors="pt")["input_ids"].to(dev)
                    w = tts.generate_speech(ids, e, vocoder=voc).cpu().numpy()
                _write(w, SR, path)
            rows.append(_row("speecht5x", s, args.split, path, text))
    return rows


def run_mms(args) -> list[dict]:
    import torch
    from transformers import AutoTokenizer, VitsModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = VitsModel.from_pretrained("facebook/mms-tts-eng", local_files_only=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", local_files_only=True)
    d = OUT / "mms"
    d.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(args.seed + 1)
    for i in tqdm(range(args.n), desc="mms"):
        path = d / f"{args.split}_{i:04d}.wav"
        text = rng.choice(TEXTS)
        if not path.exists():
            m.noise_scale = rng.uniform(0.4, 1.0)
            m.noise_scale_duration = rng.uniform(0.5, 1.0)
            m.speaking_rate = rng.uniform(0.8, 1.3)
            with torch.no_grad():
                w = m(**tok(text, return_tensors="pt").to(dev)).waveform[0].cpu().numpy()
            _write(w, m.config.sampling_rate, path)
        rows.append(_row("mms", "none", args.split, path, text))
    return rows


def run_bark(args) -> list[dict]:
    import torch
    from transformers import AutoProcessor, BarkModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        proc = AutoProcessor.from_pretrained("suno/bark-small", local_files_only=True)
        m = BarkModel.from_pretrained("suno/bark-small", local_files_only=True).to(dev).eval()
    except OSError as e:
        raise SystemExit(f"suno/bark-small (1.7 GB) is not in the local cache: {e}") from e
    presets = [f"v2/en_speaker_{i}" for i in range(10)]
    d = OUT / "bark"
    d.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(args.seed + 2)
    for i in tqdm(range(args.n), desc="bark"):
        path = d / f"{args.split}_{i:04d}.wav"
        text = rng.choice(TEXTS)
        if not path.exists():
            inp = proc(text, voice_preset=rng.choice(presets)).to(dev)
            with torch.no_grad():
                w = m.generate(**inp, do_sample=True).cpu().numpy().squeeze()
            _write(w, m.generation_config.sample_rate, path)
        rows.append(_row("bark", "none", args.split, path, text))
    return rows


def _load_knnvc(dev: str):
    """kNN-VC from the reviewed local copy. ``torch.hub.load`` would download the repo and run
    its code unseen with ``trust_repo`` and unpickle the weights with full pickle; here the
    code is the pinned copy that was read first and the weights are plain tensors
    (``weights_only=True`` refuses anything else)."""
    import torch

    for need in (KNN_CODE / "matcher.py", KNN_WEIGHTS / "WavLM-Large.pt"):
        if not need.exists():
            raise SystemExit(f"{need} is missing -- see docs/modern_attacks.md for the fetch")
    sys.path.insert(0, str(KNN_CODE))
    try:
        from hifigan.models import Generator as HiFiGAN
        from hifigan.utils import AttrDict
        from matcher import KNeighborsVC
        from wavlm.WavLM import WavLM, WavLMConfig
    finally:
        sys.path.remove(str(KNN_CODE))
    ck = torch.load(KNN_WEIGHTS / "WavLM-Large.pt", map_location="cpu", weights_only=True)
    if not _CONV_LAYERS.fullmatch(str(ck["cfg"]["conv_feature_layers"])):
        raise SystemExit("WavLM checkpoint config looks like code, refusing to eval() it")
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    h = AttrDict(json.loads((KNN_CODE / "hifigan" / "config_v1_wavlm.json").read_text()))
    gen = HiFiGAN(h)
    g = torch.load(KNN_WEIGHTS / "prematch_g_02500000.pt", map_location="cpu", weights_only=True)
    gen.load_state_dict(g["generator"])
    gen.eval().remove_weight_norm()
    return KNeighborsVC(wavlm.to(dev).eval(), gen.to(dev), h, dev)


def _tensor(path):
    import torch

    return torch.from_numpy(load_wave(path, SR)).float()[None]  # (1, T) at 16 kHz


def run_knnvc(args) -> list[dict]:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vc = _load_knnvc(dev)
    targets, bona, ok = _targets(args)
    sources = _source_pool(args, bona)
    d = OUT / "knnvc"
    d.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(args.seed + 3)
    for s, refs in tqdm(targets, desc="targets"):
        match = vc.get_matching_set([_tensor(REPO_ROOT / p) for p in refs])
        for i in range(args.per_speaker):
            path = d / f"{args.source}_{s}_{i:02d}.wav"
            src = rng.choice(sources)
            if not path.exists():
                q = vc.get_features(_tensor(src))
                with torch.no_grad():
                    w = vc.match(q, match, topk=4).cpu().numpy()
                _write(w, SR, path)
            rows.append(_row("knnvc", s, args.split, path, f"source={args.source}"))
    return rows


def _source_pool(args, bona) -> list:
    """Utterances whose *content* gets converted into the target's voice."""
    if args.source == "bona":  # a real person's speech (a pure voice-conversion attack)
        return [str(REPO_ROOT / p) for p in bona.path.sample(400, random_state=args.seed)]
    tts = sorted((OUT / "mms").glob("*.wav")) + sorted((OUT / "bark").glob("*.wav"))
    if len(tts) < 20:
        raise SystemExit("--source tts needs synthetic utterances first: run --gen mms / bark")
    return [str(p) for p in tts]


def _row(gen: str, target: str, split: str, path, note: str) -> dict:
    return dict(
        path=str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        generator=gen,
        family=FAMILY[gen],
        target_speaker=target,
        split=split,
        label="spoof",
        note=note,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, choices=sorted(FAMILY))
    ap.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    ap.add_argument("--speakers", type=int, default=0, help="targets (0 = all eligible)")
    ap.add_argument("--per-speaker", type=int, default=12, help="clips per target speaker")
    ap.add_argument("--n", type=int, default=120, help="clips for the untargeted generators")
    ap.add_argument("--ref-clips", type=int, default=8, help="reference clips per target")
    ap.add_argument("--source", default="bona", choices=["bona", "tts"], help="knnvc content")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = {"knnvc": run_knnvc, "speecht5x": run_speecht5x, "mms": run_mms, "bark": run_bark}[
        args.gen
    ](args)
    _append(rows)
    print(f"{args.gen}: {len(rows)} clips -> {MANIFEST}")


if __name__ == "__main__":
    main()
