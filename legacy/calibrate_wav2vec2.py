"""Real calibration for the fine-tuned Wav2Vec2 detector.

This REPLACES the threshold logic that used to live at the end of
train_wav2vec2.py:

    new_threshold = min(max(best_holdout_confs.max() + 0.05, 0.5), 0.95)

That line set the production decision threshold from the single highest
spoof-confidence observed across ~6-10 recordings of ONE person's voice on
ONE mic/room. It doesn't measure a false-accept/false-reject trade-off at
all -- it just finds a number that stops flagging the presenter. It is the
shortcut-learning failure mode the project's own blueprint (Section 2.7)
warns about, applied to the threshold instead of the model.

This script instead:
  1. Scores the model on a real, class-balanced sample of the ASVspoof2019
     LA dev set (in-domain, both classes, protocol-defined split -- never
     used for training).
  2. Computes EER and the DET curve on those scores (metrics.compute_eer).
  3. Reports TWO operating points, per blueprint Section 4.7: a balanced
     "EER" threshold and a stricter "high-value transaction" threshold
     that trades higher false-rejects for a much lower false-accept rate.
  4. Reports how the user's own held-out mic samples score at each
     threshold as a DIAGNOSTIC ONLY -- it does not feed back into the
     threshold, because tuning on your own voice is exactly the bug being
     fixed here.

Run: python src/calibrate_wav2vec2.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_eer, det_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "raw" / "LA" / "LA"
PROTOCOL_DIR = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
DEV_AUDIO = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"
MIC_DIR = ROOT / "data" / "mic_samples"
MODEL_DIR = ROOT / "results" / "wav2vec2_finetuned"
THRESHOLD_OUT = ROOT / "results" / "wav2vec2_threshold.npy"
CALIBRATION_TABLE = ROOT / "results" / "tables" / "wav2vec2_calibration.csv"

SAMPLE_RATE = 16000
MAX_LENGTH = 64000  # 4s, matches training
N_PER_CLASS = 1200  # dev has only 2548 bonafide total; keep both classes well represented
BATCH_SIZE = 16
SEED = 42
HIGH_SECURITY_TARGET_FAR = 0.01  # <=1% of spoof audio accepted as genuine


def load_dev_subset():
    df = pd.read_csv(
        PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
        sep=" ", header=None,
        names=["speaker_id", "file_id", "env", "attack_id", "label"],
    )
    bon = df[df.label == "bonafide"].sample(min(N_PER_CLASS, (df.label == "bonafide").sum()), random_state=SEED)
    spf = df[df.label == "spoof"].sample(min(N_PER_CLASS, (df.label == "spoof").sum()), random_state=SEED)
    return pd.concat([bon, spf]).reset_index(drop=True)


def load_waveform(path):
    waveform, _ = torchaudio.load(path)
    waveform = waveform.squeeze().numpy()
    if len(waveform) > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    else:
        waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)))
    return waveform


def score_batch(model, processor, device, waveforms):
    inputs = processor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs["input_values"].to(device)).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 0].cpu().numpy()  # P(bonafide); model trained with label 0=bonafide, 1=spoof


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    dev_subset = load_dev_subset()
    print(f"Scoring {len(dev_subset)} dev clips "
          f"({(dev_subset.label=='bonafide').sum()} bonafide / {(dev_subset.label=='spoof').sum()} spoof)...")

    scores, labels = [], []
    for start in range(0, len(dev_subset), BATCH_SIZE):
        chunk = dev_subset.iloc[start:start + BATCH_SIZE]
        waveforms = [load_waveform(DEV_AUDIO / f"{fid}.flac") for fid in chunk.file_id]
        batch_scores = score_batch(model, processor, device, waveforms)
        scores.extend(batch_scores.tolist())
        labels.extend([1 if lab == "bonafide" else 0 for lab in chunk.label])
        if start % (BATCH_SIZE * 10) == 0:
            print(f"  {start}/{len(dev_subset)}")

    scores = np.array(scores)
    labels = np.array(labels)

    eer, eer_threshold = compute_eer(labels, scores)
    print(f"\nDev-set EER: {eer*100:.2f}%  (balanced threshold = {eer_threshold:.4f})")

    fpr, fnr, thresholds = det_curve(labels, scores)
    # fpr here = FAR (spoof accepted as bonafide) at pos_label=1(bonafide) convention.
    # Among thresholds meeting the FAR target, pick the one with lowest FRR
    # (i.e. sit right at the FAR boundary instead of over-shooting it).
    valid_idx = np.where(fpr <= HIGH_SECURITY_TARGET_FAR)[0]
    if len(valid_idx):
        best = valid_idx[np.argmin(fnr[valid_idx])]
    else:
        best = int(np.argmin(fpr))
    high_security_threshold = float(thresholds[best])
    hs_far = float(fpr[best])
    hs_frr = float(fnr[best])

    print(f"High-security threshold (FAR<={HIGH_SECURITY_TARGET_FAR*100:.0f}%) = {high_security_threshold:.4f} "
          f"-> FAR={hs_far*100:.2f}%, FRR={hs_frr*100:.2f}%")

    # ---- Diagnostic only: how do personal mic samples score? Never used to pick a threshold. ----
    mic_files = sorted(glob.glob(str(MIC_DIR / "real_sample_*.npy")))
    if mic_files:
        mic_scores = []
        for f in mic_files:
            audio = np.load(f).astype(np.float32)
            s = score_batch(model, processor, device, [
                audio[:MAX_LENGTH] if len(audio) > MAX_LENGTH
                else np.pad(audio, (0, MAX_LENGTH - len(audio)))
            ])[0]
            mic_scores.append(s)
        mic_scores = np.array(mic_scores)
        print(f"\n[Diagnostic, not used for calibration] {len(mic_scores)} personal mic samples:")
        print(f"  mean P(bonafide)={mic_scores.mean():.3f}, min={mic_scores.min():.3f}")
        for name, thr in [("balanced/EER", eer_threshold), ("high-security", high_security_threshold)]:
            flagged = (mic_scores < thr).sum()
            print(f"  at {name} threshold ({thr:.3f}): {flagged}/{len(mic_scores)} of YOUR OWN clips "
                  f"would be flagged as spoof (false rejects)")
        print("  -> If this false-reject rate is high, the fix is better mic-condition training data,")
        print("     not lowering the threshold until your own voice passes.")

    # ---- Save ----
    # NOTE ON CONVENTION: everything above is computed in P(bonafide) space,
    # matching metrics.compute_eer's docstring (1=bonafide, higher=more
    # bonafide-like). streaming.py's predict() instead thresholds on
    # P(spoof) via `spoof_conf >= SPOOF_THRESHOLD`. Since this is a 2-class
    # softmax, P(spoof) = 1 - P(bonafide), so the threshold must be flipped
    # before being written to the file streaming.py reads -- saving the
    # bonafide-space number directly here was the bug that made the old
    # threshold silently wrong by construction, independent of how it was
    # chosen.
    spoof_threshold_for_streaming = 1.0 - eer_threshold

    THRESHOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(THRESHOLD_OUT, np.array([spoof_threshold_for_streaming]))
    CALIBRATION_TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"operating_point": "balanced_eer", "bonafide_space_threshold": eer_threshold,
         "spoof_space_threshold_(streaming.py)": spoof_threshold_for_streaming, "dev_eer_pct": eer * 100},
        {"operating_point": "high_security", "bonafide_space_threshold": high_security_threshold,
         "spoof_space_threshold_(streaming.py)": 1.0 - high_security_threshold,
         "far_pct": hs_far * 100, "frr_pct": hs_frr * 100},
    ]).to_csv(CALIBRATION_TABLE, index=False)
    print(f"\nSaved default (balanced) threshold to {THRESHOLD_OUT} "
          f"(spoof-space value: {spoof_threshold_for_streaming:.4f})")
    print(f"Saved both operating points to {CALIBRATION_TABLE}")


if __name__ == "__main__":
    main()
