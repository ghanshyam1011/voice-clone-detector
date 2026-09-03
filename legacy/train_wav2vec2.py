# src/train_wav2vec2.py
#
# Proper Wav2Vec2 fine-tuning:
#   - ASVspoof2019 LA bonafide/spoof gives the model general spoof-detection ability
#   - Your own mic recordings (data/mic_samples) teach it that YOUR mic/room/voice
#     is also "bonafide" — without this, real mic audio just looks out-of-distribution
#     and gets flagged as spoof (that's the bug you hit in streaming.py)
#   - Held-out mic samples (never trained on) give an honest accuracy number instead
#     of testing on the same clips the model just memorized

import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, get_scheduler
from sklearn.metrics import classification_report
from tqdm import tqdm

# ── Config ──────────────────────────────────────────
SEED          = 42
SAMPLE_RATE   = 16000
MAX_LENGTH    = 64000  # 4s at 16kHz, matches original notebook
BATCH_SIZE    = 8
NUM_EPOCHS    = 4
LR            = 1e-4
ASV_TRAIN_N   = 2500   # per-class count sampled from ASVspoof train
ASV_DEV_N     = 300    # per-class count sampled from ASVspoof dev (eval only)
MIC_OVERSAMPLE = 8      # how many times each held-in mic sample is repeated per epoch
MIC_HOLDOUT_FRAC = 0.25 # fraction of mic samples reserved for honest eval, never trained on
MIC_NOISE_STD = 0.003   # light augmentation so oversampled copies aren't byte-identical

from pathlib import Path
ROOT         = str(Path(__file__).resolve().parent.parent)
DATA_ROOT    = f'{ROOT}\\data\\raw\\LA\\LA'
PROTOCOL_DIR = f'{DATA_ROOT}\\ASVspoof2019_LA_cm_protocols'
TRAIN_AUDIO  = f'{DATA_ROOT}\\ASVspoof2019_LA_train\\flac'
DEV_AUDIO    = f'{DATA_ROOT}\\ASVspoof2019_LA_dev\\flac'
MIC_DIR      = f'{ROOT}\\data\\mic_samples'
MODEL_SAVE   = f'{ROOT}\\results\\wav2vec2_finetuned'
MODEL_BACKUP = f'{ROOT}\\results\\wav2vec2_finetuned_backup'
THRESHOLD_OUT = f'{ROOT}\\results\\wav2vec2_threshold.npy'

rng = np.random.default_rng(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load ASVspoof protocols ──────────────────────────
train_df = pd.read_csv(f'{PROTOCOL_DIR}\\ASVspoof2019.LA.cm.train.trn.txt',
    sep=' ', header=None, names=['speaker_id', 'file_id', 'env', 'attack_id', 'label'])
dev_df = pd.read_csv(f'{PROTOCOL_DIR}\\ASVspoof2019.LA.cm.dev.trl.txt',
    sep=' ', header=None, names=['speaker_id', 'file_id', 'env', 'attack_id', 'label'])

train_bon = train_df[train_df['label'] == 'bonafide'].sample(
    min(ASV_TRAIN_N, (train_df['label'] == 'bonafide').sum()), random_state=SEED)
train_sp  = train_df[train_df['label'] == 'spoof'].sample(ASV_TRAIN_N, random_state=SEED)
asv_train_subset = pd.concat([train_bon, train_sp]).sample(frac=1, random_state=SEED).reset_index(drop=True)

dev_bon = dev_df[dev_df['label'] == 'bonafide'].sample(
    min(ASV_DEV_N, (dev_df['label'] == 'bonafide').sum()), random_state=SEED)
dev_sp  = dev_df[dev_df['label'] == 'spoof'].sample(ASV_DEV_N, random_state=SEED)
asv_dev_subset = pd.concat([dev_bon, dev_sp]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"ASVspoof train subset: {len(asv_train_subset)} (bonafide={len(train_bon)}, spoof={len(train_sp)})")
print(f"ASVspoof dev subset:   {len(asv_dev_subset)} (bonafide={len(dev_bon)}, spoof={len(dev_sp)})")

# ── Load + split your mic samples ────────────────────
mic_files = sorted(glob.glob(f'{MIC_DIR}\\real_sample_*.npy'))
if len(mic_files) < 8:
    raise SystemExit(
        f"Only {len(mic_files)} mic samples found in {MIC_DIR}. "
        "Run `python src/calibrate.py` first to record more (at least ~20, ideally 40+) "
        "real samples with varied sentences/conditions — too few and the model just memorizes them."
    )

mic_files = list(mic_files)
rng.shuffle(mic_files)
n_holdout = max(3, int(len(mic_files) * MIC_HOLDOUT_FRAC))
mic_holdout_files = mic_files[:n_holdout]
mic_train_files   = mic_files[n_holdout:]

print(f"Mic samples: {len(mic_files)} total -> {len(mic_train_files)} train / {len(mic_holdout_files)} held-out eval")

# ── Dataset ───────────────────────────────────────────
class SpoofDataset(Dataset):
    """
    items: list of dicts, each either
      {'kind': 'asv', 'path': <flac path>, 'label': 0/1}
      {'kind': 'mic', 'path': <npy path>, 'label': 0, 'augment': bool}
    """
    def __init__(self, items, processor, max_length=MAX_LENGTH):
        self.items = items
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def _load_waveform(self, item):
        if item['kind'] == 'asv':
            waveform, sr = torchaudio.load(item['path'])
            waveform = waveform.squeeze().numpy()
        else:
            waveform = np.load(item['path']).astype(np.float32)
            if item.get('augment'):
                waveform = waveform * rng.uniform(0.85, 1.15)
                waveform = waveform + rng.normal(0, MIC_NOISE_STD, size=waveform.shape).astype(np.float32)
        return waveform

    def __getitem__(self, idx):
        item = self.items[idx]
        waveform = self._load_waveform(item)

        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

        inputs = self.processor(waveform, sampling_rate=SAMPLE_RATE,
                                 return_tensors='pt', padding=False)

        return {
            'input_values': inputs['input_values'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def asv_items(df, audio_dir):
    return [{'kind': 'asv', 'path': f'{audio_dir}\\{row.file_id}.flac',
              'label': 1 if row.label == 'spoof' else 0}
            for row in df.itertuples()]

def mic_items(files, oversample, augment):
    items = []
    for f in files:
        for _ in range(oversample):
            items.append({'kind': 'mic', 'path': f, 'label': 0, 'augment': augment})
    return items

# ── Load model/processor ─────────────────────────────
print("Loading facebook/wav2vec2-base (fresh base checkpoint)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2).to(device)

train_items = asv_items(asv_train_subset, TRAIN_AUDIO) + mic_items(mic_train_files, MIC_OVERSAMPLE, augment=True)
rng.shuffle(train_items)

train_dataset = SpoofDataset(train_items, processor)
dev_dataset   = SpoofDataset(asv_items(asv_dev_subset, DEV_AUDIO), processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer,
    num_warmup_steps=50, num_training_steps=NUM_EPOCHS * len(train_loader))

print(f"Train examples: {len(train_dataset)} (incl. {len(mic_train_files) * MIC_OVERSAMPLE} mic, oversampled x{MIC_OVERSAMPLE})")
print(f"Dev batches: {len(dev_loader)}")

# ── Train / eval loops ───────────────────────────────
def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_values = batch['input_values'].to(device)
        labels       = batch['label'].to(device)

        outputs = model(input_values=input_values, labels=labels)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def eval_asv_dev():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Eval (ASVspoof dev)"):
            input_values = batch['input_values'].to(device)
            labels       = batch['label'].to(device)
            preds = model(input_values=input_values).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

def eval_mic_holdout():
    """Honest check: samples the model has never seen in training."""
    model.eval()
    spoof_confs = []
    with torch.no_grad():
        for f in mic_holdout_files:
            audio = np.load(f).astype(np.float32)
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors='pt', padding=True)
            logits = model(inputs['input_values'].to(device)).logits
            probs  = torch.softmax(logits, dim=-1)
            spoof_confs.append(probs[0][1].item())
    spoof_confs = np.array(spoof_confs)
    acc = (spoof_confs < 0.5).mean()
    return spoof_confs, acc

best_score = -1
best_state = None
best_holdout_confs = None

print("\n=== Wav2Vec2 Fine-tuning (ASVspoof + your mic samples) ===\n")
for epoch in range(NUM_EPOCHS):
    loss = train_epoch()
    preds, labels = eval_asv_dev()
    holdout_confs, holdout_acc = eval_mic_holdout()

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} — Loss: {loss:.4f}")
    print(classification_report(labels, preds, target_names=['bonafide', 'spoof'], zero_division=0))
    print(f"Held-out mic samples — accuracy (correctly kept as bonafide): {holdout_acc*100:.1f}%")
    print(f"Held-out mic spoof-confidence — mean: {holdout_confs.mean()*100:.1f}%  max: {holdout_confs.max()*100:.1f}%\n")

    asv_acc = np.mean(np.array(preds) == np.array(labels))
    score = 0.5 * asv_acc + 0.5 * holdout_acc  # balance general spoof detection vs. your-voice accuracy
    if score > best_score:
        best_score = score
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_holdout_confs = holdout_confs
        print(f"↑ New best (score={score:.3f}), checkpointing in memory\n")

# ── Save best model ───────────────────────────────────
model.load_state_dict(best_state)

if os.path.exists(MODEL_SAVE):
    if os.path.exists(MODEL_BACKUP):
        import shutil
        shutil.rmtree(MODEL_BACKUP)
    os.rename(MODEL_SAVE, MODEL_BACKUP)
    print(f"Old model backed up to {MODEL_BACKUP}")

os.makedirs(MODEL_SAVE, exist_ok=True)
model.save_pretrained(MODEL_SAVE)
processor.save_pretrained(MODEL_SAVE)
print(f"New model saved to {MODEL_SAVE} ✅")

# ── Threshold ──────────────────────────────────────────
# Do NOT set the production threshold from best_holdout_confs here -- that
# was "max confidence on a handful of personal mic clips + 5%", which
# measures nothing about false-accept/false-reject trade-offs and just
# stops flagging whoever recorded the mic samples. Run
# src/calibrate_wav2vec2.py after this script: it computes a real
# threshold from EER on the ASVspoof dev set and writes THRESHOLD_OUT.
print(f"\nHeld-out mic samples (diagnostic only, not used for threshold): "
      f"mean spoof-confidence {best_holdout_confs.mean()*100:.1f}%, max {best_holdout_confs.max()*100:.1f}%")
print("\nDone training. Next: run `python src/calibrate_wav2vec2.py` to set a real threshold.")
