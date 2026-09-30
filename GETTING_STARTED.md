# Getting started — running VoiceGuard on your own computer

This is a step-by-step guide for someone who has **never touched this project
before** and wants to get it running from scratch, starting from the GitHub
page. No prior knowledge of the codebase is assumed. If you already know
your way around Python projects, the short version is in
[`README.md`](README.md)'s "Reproduce" section instead — this file is the
long, hand-holding version.

**What you'll end up with:** a working voice-clone/deepfake-speech detector
running in your own browser, and (optionally) the ability to train your own
copy of the detection model.

---

## Before you start

You'll need, on your computer:

- **Windows, Mac, or Linux** — all commands below are given for Windows
  (PowerShell) and Mac/Linux (bash/zsh) side by side.
- **Git** — to download ("clone") the code. Check you have it:
  `git --version`. If that fails, install it from
  [git-scm.com](https://git-scm.com/downloads).
- **Python 3.11 or newer** — check with `python --version` (Windows) or
  `python3 --version` (Mac/Linux). If you don't have it, install from
  [python.org](https://www.python.org/downloads/) (tick "Add python.exe to
  PATH" during the Windows installer).
- **~20 GB of free disk space** — the dataset alone is about 7 GB
  compressed, more once extracted, plus the Python libraries.
- **A decent internet connection** — you'll be downloading a multi-gigabyte
  dataset.
- **A GPU is not required.** An NVIDIA GPU makes everything (especially
  training) much faster, but the whole project also works on a plain CPU —
  it's just slower. Nothing below assumes you have one.

Rough time budget if you're doing this for the first time: 10-15 minutes to
install everything, 30 minutes to a few hours to download the dataset
(depends entirely on your internet speed), and then the live demo works
immediately. Training your own model (optional, last section) takes
hours-to-days regardless of hardware — that part is not a "do this in one
sitting" step.

---

## Step 1 — Get the code

Open a terminal (Windows: **PowerShell**; Mac: **Terminal**; Linux: your
usual shell) and run:

```bash
git clone https://github.com/ghanshyam1011/voice-clone-detector.git
cd voice-clone-detector
```

This downloads the project's code into a new `voice-clone-detector` folder
and moves you into it. Every command from here on assumes you're inside
this folder.

---

## Step 2 — Create an isolated Python environment

This keeps this project's Python packages separate from anything else on
your computer, so it can't conflict with other software you have.

**Windows (PowerShell):**

```powershell
python -m venv myenv
myenv\Scripts\activate
```

**Mac / Linux:**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

You'll know it worked when your terminal prompt starts showing `(myenv)` in
front of it. **Every command below assumes this is still active** — if you
close and reopen your terminal, re-run just the activate line (not the
`python -m venv myenv` line — that only needs to happen once, ever).

> **If Windows says "running scripts is disabled on this system":**
> PowerShell blocks script execution by default. Run this once in the same
> window, then try activating again:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
> This only relaxes the restriction for this one terminal window, not your
> whole computer.

---

## Step 3 — Install the required libraries

The project is split into optional pieces ("extras") so you only install
what you need. For the full experience described in this guide (demo +
training), install:

**Windows / Mac / Linux (same command):**

```bash
pip install -e ".[dl,serve,prosody,dev]"
```

This installs, among others: PyTorch (the deep-learning library the
detector runs on), FastAPI + Uvicorn (the local web server for the demo),
the audio-analysis libraries, and pytest (to run the test suite). It takes
a few minutes — PyTorch in particular is a large download.

You'll also want **ffmpeg** on your system (not a Python package — a
separate program the audio pipeline calls out to, for simulating
phone-call audio quality):

- **Windows:** `winget install ffmpeg` (or download from
  [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH)
- **Mac:** `brew install ffmpeg` (needs [Homebrew](https://brew.sh))
- **Linux:** `sudo apt install ffmpeg` (Debian/Ubuntu) or your
  distribution's equivalent

Check it worked: `ffmpeg -version` should print a version number, not
"command not found."

> **Optional extras you can add later, if you want them:** `baseline`
> (the older handcrafted-features comparison model), `grpc` (a second way
> to talk to the detector besides the web API — see
> [`docs/grpc.md`](docs/grpc.md)), `research` (tools for generating
> synthetic multilingual test data). None of these are needed for the demo
> or for training.

---

## Step 4 — Get the dataset

The detector needs real examples of genuine and AI-generated speech to be
evaluated against (and, later, to train on). This is the biggest,
most fiddly step — take it slowly.

### 4a. Download ASVspoof 2019 LA (required)

This is the main dataset the project is built around.

- Go to the official page:
  **[datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)**
  (hosted by the University of Edinburgh — this is the dataset's original,
  citable source).
- Find the file list and download **`LA.zip`** (about **7.12 GB**). No
  account should be needed — it's an open research dataset.
- **Do not download `PA.zip`** (16+ GB) — that's a different partition of
  the same challenge that this project doesn't use.
- If that server is slow for you, a community mirror exists on Kaggle:
  search Kaggle for **"ASVspoof 2019 dataset"** (e.g. the dataset uploaded
  by user `awsaf49`) and use its **Download** button instead — same data,
  just a different host.

**Once downloaded, extract it so you end up with this exact folder
layout** (the code looks for these exact paths):

```
voice-clone-detector/
  data/
    raw/
      LA/
        LA/
          ASVspoof2019_LA_cm_protocols/
          ASVspoof2019_LA_train/
          ASVspoof2019_LA_dev/
          ASVspoof2019_LA_eval/
          ... (a few other files)
```

In practice: `LA.zip` extracts to a single top-level `LA` folder already
containing exactly those subfolders — so extract it, then move/rename so
that folder ends up at `data/raw/LA/LA/` (note **`LA` appears twice** in
that path — that's intentional, not a typo).

**Windows (PowerShell), assuming `LA.zip` landed in your Downloads folder:**

```powershell
mkdir data\raw\LA -Force
Expand-Archive -Path "$env:USERPROFILE\Downloads\LA.zip" -DestinationPath data\raw\LA
```

> PowerShell's built-in unzip can be very slow (sometimes appears to hang)
> on a file this large. If it's taking more than 15-20 minutes with no
> progress, cancel it and instead install [7-Zip](https://www.7-zip.org/)
> and right-click `LA.zip` → **7-Zip → Extract to "LA\"** — much faster for
> archives this size.

**Mac / Linux:**

```bash
mkdir -p data/raw/LA
unzip ~/Downloads/LA.zip -d data/raw/LA
```

Either way, double-check afterwards that `data/raw/LA/LA/ASVspoof2019_LA_cm_protocols`
actually exists before moving on — if the folder ended up one level too
deep or too shallow, move it until the layout above matches exactly.

### 4b. In-the-Wild (optional — a tougher "real world" test set)

Skip this on your first pass; the demo and basic training don't need it.
It's used later for a harder, out-of-distribution check (does the detector
still work on audio nothing like its training data?).

- Hosted on Hugging Face:
  **[huggingface.co/datasets/mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild)**
- Download `release_in_the_wild.zip`, then extract it to:
  `data/external/in_the_wild/extracted/release_in_the_wild/` (so that
  `.../release_in_the_wild/meta.csv` exists).

### 4c. Turn the raw data into the project's own index files

Whatever you downloaded above, run this once to let the project index it:

```bash
python scripts/build_manifests.py
```

You should see it report writing `asvspoof19_la_train.csv`,
`asvspoof19_la_dev.csv`, `asvspoof19_la_eval.csv` (and `in_the_wild_eval.csv`
if you did step 4b) — check `data/manifests/` now has these files. If it
instead prints `SKIP asvspoof_la (not found at ...)`, the folder layout in
4a isn't quite right yet — that message tells you the exact path it looked
for.

---

## Step 5 — Run the live demo (the payoff!)

This is the fastest way to actually see the detector working.

```bash
python scripts/serve_demo.py
```

The server itself starts in a few seconds. The **first time you actually use
the Enrol step**, it downloads one more pretrained model automatically
(WavLM, for the speaker-matching signal — a few hundred MB, one-time only,
then cached). On startup you should see something like:

```
VoiceGuard demo  ->  http://127.0.0.1:8000
Ctrl+C to stop
```

and opens that address in your default browser automatically (if it
doesn't, open it yourself). You'll see a console with:

- An **Enrol caller's voice** step (record yourself briefly, or upload a
  short clip of a voice).
- A place to **drop or upload a call recording** to analyze.
- Live signal meters and an **ALLOW / VERIFY / ESCALATE** decision as you
  feed it audio.

Play with your own voice, or try a clip generated by any free text-to-speech
site — the detector should flag the synthetic one. For a fully scripted
walkthrough (what to click, in what order, to tell a complete story), see
[`docs/demo_runbook.md`](docs/demo_runbook.md).

Stop the server with **Ctrl+C** in the terminal when you're done.

---

## Step 6 — Run the automated tests (confirms everything installed correctly)

Optional, but a good sanity check that nothing above went wrong:

```bash
pytest -q
```

A healthy result ends with a line like `103 passed`. A handful of tests
automatically **skip** (not fail) if you didn't download every optional
piece (e.g. the min-tDCF tests skip without a specific ASV score file) —
that's expected and fine. Actual **failures** (not skips) usually mean a
step above needs revisiting — the error message will name the missing file
or package.

---

## Step 7 — Train your own model (optional, and it's a real time commitment)

Everything above uses the detector's **published, pretrained weights** —
you don't need to train anything to see it work. Training your own is for
people who want to try to improve on that.

Fair warning: even on a good GPU this is a 1-2+ day, multi-session
undertaking, not a quick extra step — read
**[`docs/training_runbook.md`](docs/training_runbook.md)** before starting,
it explains the reasoning (which model to train first) and both ways to do
it:

- **On your own computer**, however powerful or modest — fully stoppable
  and resumable, so you can train it a little at a time:
  ```bash
  python scripts/train_cm.py --model aasist --epochs 40 --batch-size 16
  ```
  Stop it however you need to (close the terminal, shut down, Ctrl+C); run
  that exact same command again later and it picks up where it left off
  automatically.
- **On Kaggle's free cloud GPUs**, if your own computer doesn't have a
  capable graphics card: upload
  **[`notebooks/kaggle_train_cm.ipynb`](notebooks/kaggle_train_cm.ipynb)**
  to a new Kaggle notebook and run it top to bottom — it handles getting
  the code and data in place for you.

Once training finishes (or even partway through — it always keeps the best
checkpoint so far), see how your own weights compare to the published ones:

```bash
python scripts/eval_cm.py --model aasist
```

---

## Troubleshooting

- **`python: command not found` / `'python' is not recognized`** — Python
  isn't installed, or isn't on your PATH. Reinstall from python.org and
  make sure to tick "Add to PATH" on Windows. On Mac/Linux, try `python3`
  instead of `python` everywhere in this guide.
- **`ModuleNotFoundError: No module named 'X'`** — either your virtual
  environment isn't active (your prompt should show `(myenv)` — re-run the
  activate command from Step 2), or Step 3's install command didn't
  finish successfully — scroll up in its output for the actual error.
- **GPU / CUDA not detected** — not an error. Everything automatically
  falls back to your CPU; it'll just run slower, especially training.
- **`Could not decode audio` in the demo, or ffmpeg-related errors** —
  ffmpeg isn't installed or isn't on your PATH (Step 3). Run
  `ffmpeg -version` to check.
- **Port 8000 already in use** — something else on your computer is using
  that port. Run `python scripts/serve_demo.py --port 8080` instead (or
  any other free port number), then visit that port in your browser.
- **`build_manifests.py` says `SKIP asvspoof_la (not found at ...)`** —
  the path it printed is exactly where it looked; your extracted dataset
  isn't there yet. Re-check Step 4a's folder layout.

---

## Where to go next

- **[`README.md`](README.md)** — the full technical write-up: what each
  detection signal does, measured accuracy numbers, architecture, and
  known limitations, written for a more technical reader.
- **[`docs/project_status.md`](docs/project_status.md)** — an honest,
  up-to-date account of what's built, what's partial, and what isn't
  started yet.
- **[`docs/demo_runbook.md`](docs/demo_runbook.md)** — a scripted,
  timed walkthrough for presenting the demo to an audience.
- **[`docs/grpc.md`](docs/grpc.md)**, **[`docs/calibration.md`](docs/calibration.md)**,
  **[`docs/tdcf.md`](docs/tdcf.md)** — deeper dives into specific pieces,
  for anyone who wants to go further than this guide.
