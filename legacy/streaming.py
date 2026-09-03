import os
from pathlib import Path
import sounddevice as sd
import numpy as np
import torch
import queue
import time
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ── Config ──────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 3
CHUNK_SIZE    = SAMPLE_RATE * CHUNK_SECONDS

ROOT          = Path(__file__).resolve().parent.parent
RESULTS_DIR   = str(ROOT / "results")
MODEL_SAVE    = f'{RESULTS_DIR}\\wav2vec2_finetuned'
THRESHOLD_FILE = f'{RESULTS_DIR}\\wav2vec2_threshold.npy'
DEFAULT_THRESHOLD = 0.7

# ── Load model ───────────────────────────────────────
print("Loading Wav2Vec2 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = Wav2Vec2Processor.from_pretrained(MODEL_SAVE)
model     = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_SAVE).to(device)
model.eval()

if os.path.exists(THRESHOLD_FILE):
    SPOOF_THRESHOLD = float(np.load(THRESHOLD_FILE)[0])
else:
    SPOOF_THRESHOLD = DEFAULT_THRESHOLD

print(f"Model loaded ✅ | Device: {device} | Spoof threshold: {SPOOF_THRESHOLD*100:.1f}%")

# ── Predict ──────────────────────────────────────────
def predict(audio_chunk):
    # Silence check
    if np.abs(audio_chunk).max() < 0.01:
        return "🔇 SILENCE"

    inputs = processor(
        audio_chunk.astype(np.float32),
        sampling_rate=16000,
        return_tensors='pt',
        padding=True
    )

    with torch.no_grad():
        logits = model(
            inputs['input_values'].to(device)
        ).logits

    probs      = torch.softmax(logits, dim=-1)
    spoof_conf = probs[0][1].item()
    real_conf  = probs[0][0].item()

    if spoof_conf >= SPOOF_THRESHOLD:
        return f"🚨 SPOOF DETECTED  ({spoof_conf*100:.1f}% confidence)"
    else:
        return f"✅ REAL VOICE      ({real_conf*100:.1f}% confidence)"

# ── Real-time stream ─────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy().flatten())

def run_realtime():
    print("\n🎙️  Real-time Voice Clone Detector (Wav2Vec2)")
    print(f"   Chunk: {CHUNK_SECONDS}s | SR: {SAMPLE_RATE}Hz")
    print("   Press Ctrl+C to stop\n")

    buffer = np.array([], dtype=np.float32)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='float32', callback=audio_callback,
                        blocksize=1024):
        while True:
            try:
                chunk  = audio_queue.get(timeout=1)
                buffer = np.concatenate([buffer, chunk])

                if len(buffer) >= CHUNK_SIZE:
                    audio_chunk = buffer[:CHUNK_SIZE]
                    buffer      = buffer[CHUNK_SIZE:]

                    result = predict(audio_chunk)
                    print(f"[{time.strftime('%H:%M:%S')}] {result}")

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("\n⏹️  Stopped.")
                break

if __name__ == '__main__':
    run_realtime()