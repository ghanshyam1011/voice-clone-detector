# src/calibrate.py
import sounddevice as sd
import numpy as np
import os
import glob

SAMPLE_RATE = 16000
DURATION    = 3  # seconds
SAVE_DIR    = r'C:\Users\GHANSHYAM\Desktop\voice-clone-detector\data\mic_samples'
NUM_NEW     = 40  # kitne naye samples record karne hain

PROMPTS = [
    "Apna naam aur ek chhoti si sentence bolo.",
    "Aaj ka din kaisa raha, thoda bata do.",
    "1 se 10 tak ginti bolo.",
    "Koi bhi paragraph zor se padho.",
    "Dheere aur shaant awaaz mein bolo.",
    "Thoda tez aur excited hoke bolo.",
    "Mic se thoda door hoke bolo.",
    "Mic ke paas se bolo.",
    "Kal kya kiya tha, wo bata do.",
    "Apna favourite khana describe karo.",
]

os.makedirs(SAVE_DIR, exist_ok=True)

existing = glob.glob(f'{SAVE_DIR}\\real_sample_*.npy')
start_idx = len(existing)

print("🎙️  Mic Calibration — Real Voice Collection")
print(f"{start_idx} samples already saved. {NUM_NEW} naye samples record karenge.")
print("Har baar alag sentence/tone/distance try karo — jitni variety utna behtar model.\n")

for i in range(NUM_NEW):
    idx    = start_idx + i
    prompt = PROMPTS[i % len(PROMPTS)]
    input(f"Sample {i+1}/{NUM_NEW} — {prompt}\nEnter dabao aur bolna shuru karo...")
    print("Recording...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()

    audio = audio.flatten()
    np.save(f'{SAVE_DIR}\\real_sample_{idx}.npy', audio)
    print(f"✅ Sample {i+1} saved\n")

print("Sab samples save ho gaye ✅")
print(f"Total samples ab: {start_idx + NUM_NEW}")
print(f"Location: {SAVE_DIR}")
