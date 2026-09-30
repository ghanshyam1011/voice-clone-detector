# Training runbook: item c, a detector that's actually ours

_Added 2026-09-25. Everything below is either already-verified project
convention (the train/eval scripts, the resumability contract) or an
explicit, flagged recommendation -- nothing here claims a number this repo
hasn't actually produced._

**Prefer not to copy-paste these steps into Kaggle cells by hand?**
[`notebooks/kaggle_train_cm.ipynb`](../notebooks/kaggle_train_cm.ipynb) is
this exact runbook as a ready-to-upload notebook -- clone, install, locate
and link the attached dataset, build manifests, smoke-test, train, evaluate.
Upload it to a new Kaggle notebook and run top to bottom.

## The decision this makes: train AASIST first, not SSL-AASIST

`docs/p1_notes.md` names SSL-AASIST (frozen wav2vec2 + AASIST backend) as
the architecture most likely to reach the <15% In-the-Wild EER target --
that is a claim from the wider ASVspoof/deepfake-detection literature about
SSL front-ends in general, **not a number this repo has measured**, since
no model here has completed a real training run yet (`docs/project_status.md`
item c). Two things currently make it the wrong first run:

1. **No feature cache.** `SSLFrontend` (`src/voiceguard/models/ssl_aasist.py`)
   runs the full 94M-parameter frozen wav2vec2 forward pass on every step,
   every epoch. `docs/p1_notes.md` already flags this as a ~10x throughput
   loss, worth fixing before a real SSL run, and it hasn't been fixed.
2. **Untested at scale.** SSL-AASIST's own notes say "forward + frozen-backbone
   verified" -- correctness of one forward pass, not a training loop run
   for real epochs.

Training it today would burn a disproportionate share of Kaggle's weekly
GPU quota (see below) on an architecture that isn't tuned to use that quota
well yet. **Recommendation: run AASIST end-to-end first.** It is simpler,
faster per epoch, and already smoke-tested locally (a 2-epoch run took the
dev EER from 34.1% to 26.6%, proving the loop is correct). If it beats
35.8% ITW EER by a meaningful margin, that alone is a real result worth
shipping. SSL-AASIST becomes the natural next step once the cache exists
and there's a baseline to beat.

## Kaggle setup

Kaggle's free GPU quota is **30 hours/week** (T4x2 or P100, whichever your
account is assigned), and **each individual session has a runtime cap**
(currently below 12h -- check the current limit in the Kaggle UI, it has
changed before). `scripts/train_cm.py` is built for exactly this
constraint: it is **resumable**. `last.pt` carries model + optimizer +
scheduler + scaler state plus the epoch and running log, and is overwritten
every epoch; a restart picks it up automatically unless `--fresh` is
passed. A run spanning multiple Kaggle sessions is the expected, supported
path, not a workaround.

1. **Get the data onto Kaggle without spending session time downloading
   it every run.** Search Kaggle's own dataset search for "ASVspoof 2019
   LA" first -- community-uploaded copies of this corpus are common, and
   attaching an existing public dataset costs nothing. Only upload your
   own copy (Kaggle: Datasets -> New Dataset) if you can't find one you
   trust; the LA partition (train+dev+eval) is several GB, so this is a
   one-time cost either way, not a per-session one.
2. **New Notebook -> Add Data -> your attached dataset -> GPU accelerator
   on** (T4x2 or P100, Settings panel).
3. **Get this repo onto the notebook.** Simplest: `!git clone` your repo
   (push the current uncommitted work first -- see `docs/project_status.md`'s
   "Loose end" section) as a notebook cell, or upload it as a second
   Kaggle dataset if you'd rather not make the repo public.
4. **Point the repo-relative config at Kaggle's data mount** without
   editing the one file `config/default.yaml` warns not to scatter paths
   outside of (`README.md` §6): symlink instead.
   ```bash
   mkdir -p data/raw/LA
   ln -s /kaggle/input/<your-dataset-slug>/LA data/raw/LA/LA
   ```
   Adjust the source path to match whatever the attached dataset's own
   internal folder structure actually is (check with `!ls /kaggle/input/<slug>/`
   first -- community datasets don't all nest the same way).
5. **Install and run:**
   ```bash
   pip install -e ".[dl,dev]"
   python scripts/train_cm.py --model aasist --epochs 40 --batch-size 16
   ```
   Watch the printed `~Nmin left this epoch` ETA on the first few batches
   to sanity-check the session's remaining time budget before committing
   to a long unattended run.
6. **Persist across sessions.** Everything Kaggle needs to keep is under
   `models/cm/aasist/` (`last.pt`, `best.pt`) and
   `results/tables/cm_aasist_trainlog.csv`. At the end of a session, these
   files are already in `/kaggle/working/` if you ran the repo from there,
   so they become the notebook's **Output** automatically -- save a
   version before the session ends. Next session: attach that output as a
   new input dataset, copy `last.pt`/`best.pt` back into
   `models/cm/aasist/`, and re-run the exact same command (no `--fresh`)
   -- it resumes from the saved epoch.

## Local training instead: stop and resume on your own schedule

If Kaggle's session boundaries (commit, wait, re-attach the previous
output, repeat) are more friction than they're worth, training locally on
your own GPU with the exact same resumability is a completely reasonable
alternative -- **nothing new needs building for this; `scripts/train_cm.py`
already does it.** It is not faster (the project's own numbers: ~50-57
min/epoch on a 6 GB laptop GPU, so 40 epochs is still 1.5-2+ days of GPU
time either way) -- what changes is *who* controls the stop/resume
boundary. Locally, it's whenever you want, not whenever Kaggle's clock says.

```bash
python scripts/train_cm.py --model aasist --epochs 40 --batch-size 16
```

**Stop it however you need to** -- Ctrl+C, close the terminal, shut the
machine down. `last.pt` (model + optimizer + scheduler + scaler state) is
overwritten after **every** completed epoch, not just improvements, so at
most one in-progress epoch's work is ever at risk.

**Resume by running the exact same command again** -- no new flag, nothing
to point at a checkpoint by hand:

```bash
python scripts/train_cm.py --model aasist --epochs 40 --batch-size 16
```

It finds `models/cm/aasist/last.pt`, loads everything, and prints
`resumed from ...: continuing at epoch N/40` before picking up at epoch
`N`. Passing `--fresh` is the only way to discard this and start over --
never pass it by habit between runs of the same model.

**What "resume" actually guarantees:** per-epoch, not per-batch. An
interruption mid-epoch loses that one epoch's progress (re-trained from
its start next run), not anything before it -- there is no mid-epoch
checkpoint. At ~50-57 min/epoch locally, that's the most you ever lose to
an ungraceful stop.

**Two different ways to "stop," worth telling apart:**
- **Sleep** (closing a laptop lid, by default) freezes the process in
  memory rather than ending it -- on wake, training just continues from
  the exact instruction it was on, no resume logic even invoked. Simplest
  option if you can leave the machine as-is between sessions.
- **Actually ending the process** (shutdown, closing the terminal window
  it's running in, a crash) needs the resume-on-restart behavior above.
  On Windows, closing the terminal window itself typically kills the
  process inside it -- leave that window open (minimizing is fine) if you
  want a sleep/wake cycle to just work without relying on `last.pt` at all.

**Still true regardless of stop/resume:** don't run this alongside another
extraction/training/profiling job on the same machine -- one at a time,
whether stopping for an hour or picking back up next week.

## What "done" looks like

```bash
python scripts/eval_cm.py --model aasist          # not --pretrained: uses models/cm/aasist/best.pt
```

`results/tables/cm_aasist_cross_dataset.csv` gets a new row for our own
weights, directly comparable to the existing published-weights row (35.8%
ITW EER) in the same file. Ship the new weights (update
`scripts/serve_demo.py`'s `--ours` path is already wired for this) **only
if** the number is actually better under the identical protocol -- this
project's own stated rule (`docs/project_status.md` item c), not a new one
invented for this runbook.

Once real numbers exist, `docs/project_status.md` item c and every EER
table in `README.md`/the paper that currently says "published weights"
needs the same kind of reconciliation pass this session did for
calibration, VAD, min-tDCF, and gRPC -- flagging it now so it isn't
forgotten later.
