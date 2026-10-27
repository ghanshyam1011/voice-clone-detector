# Modern attacks: where the data and models come from

Every number before this work used 2019-era attacks (ASVspoof 2019 LA, A01-A19).
A fraudster today clones one victim's voice from a few seconds of audio with an
open model. This page records what was fetched to measure that, where it came
from, what was checked before it was run, and under which licence it may be used.

Nothing is downloaded by the project's scripts at run time. Every model is loaded
from a local copy or from the Hugging Face cache with `local_files_only`; a model
that is missing is an error, not a silent multi-gigabyte download.

## What was fetched (2026-10-22, with the owner's approval)

| Item | Source | Size | Licence | Integrity | Lives in |
|---|---|--:|---|---|---|
| LibriSpeech `dev-clean`, `test-clean` | openslr.org/12 | 338 MB, 347 MB | CC BY 4.0 | MD5 equals the server's `md5sum.txt` | `data/external/librispeech/` |
| kNN-VC code | github.com/bshall/knn-vc @ `c616845c` (2026-01-16) | 0.2 MB (code only) | MIT | read in full, see below | `models/external/knn-vc-code/` |
| kNN-VC `WavLM-Large.pt` | the repo's `v0.1` release | 1,261,965,425 B | MIT (WavLM, Microsoft) | size equals the GitHub API; SHA-256 `6fb4b3c3...f100f` | `models/external/knn-vc-weights/` |
| kNN-VC `prematch_g_02500000.pt` | same release | 66,214,643 B | MIT | size equals the GitHub API; SHA-256 `f924c763...2dd5` | same |
| `speechbrain==1.0.3` | PyPI | 0.9 MB (+3 small deps) | Apache-2.0 | pinned: a newer one would upgrade torch 2.3.1 | `myenv` |
| `speechbrain/spkrec-ecapa-voxceleb` | Hugging Face @ `0f99f2d0eb` | 89 MB | Apache-2.0 | every `.ckpt` loads with `weights_only=True` | `models/external/` |
| `speechbrain/spkrec-xvect-voxceleb` | Hugging Face @ `56895a2df4` | 33 MB | Apache-2.0 | same | `models/external/` |

GitHub publishes no checksums for the kNN-VC release, so the SHA-256 values are
recorded here for provenance only: they prove a later copy is the same file, not
that the file is what the authors built.

```
6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f  WavLM-Large.pt
f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5  prematch_g_02500000.pt
76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3  dev-clean.tar.gz
39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23  test-clean.tar.gz
```

Already on the machine and used offline: MMS-TTS (English, Hindi, Bengali, Marathi,
Tamil), SpeechT5 + its HiFi-GAN vocoder. Not fetched: Bark-small (declined).

## What was checked before anything ran

**kNN-VC code.** The inference path (`hubconf.py`, `matcher.py`, `knnvc_utils.py`,
`hifigan/models.py`, `hifigan/utils.py`, `wavlm/WavLM.py`, `wavlm/modules.py`) was read.
A search of every file for process, network, socket, pickle, `open(`, `os.` and
`sys.` use found none in the model files. The one hazard is upstream fairseq
behaviour: `WavLM.py` calls `eval()` on a configuration string stored in the
checkpoint. `scripts/make_modern_clones.py` refuses the checkpoint unless that string
matches `[\d\s\[\](),+*]+`, i.e. looks like `[(512,10,5)] + [(512,3,2)] * 4` and not
like code. `torch.hub.load("bshall/knn-vc", trust_repo=True)` is deliberately not
used: it would fetch whatever the repository holds on the day and execute it.

**Weights.** Loaded with `torch.load(..., weights_only=True)`, which refuses
anything but tensors and plain containers (a pickle that carries code fails to load).
The same was done for the speechbrain checkpoints; their `hyperparams.yaml` files
instantiate only `speechbrain.*` classes and `torch.nn.LeakyReLU`.

## Licences, and what they allow

- **MMS-TTS is CC BY-NC 4.0 (non-commercial).** Audio it generates is fine as an
  *evaluation* attack. Training a shipped detector on it needs the licence cleared
  or the generator replaced; treat MMS output as evaluation-only until then.
- SpeechT5, its vocoder, kNN-VC, WavLM-Large, speechbrain and both speaker models
  are MIT or Apache-2.0.
- **The speaker model that ships today, `microsoft/wavlm-base-plus-sv`, declares no
  licence on its Hugging Face page.** Its code lineage is MIT, but the weights were
  fine-tuned on VoxCeleb. Resolve this before a commercial deployment (an audit item,
  not a blocker for research use).

## Attack families

| Generator | Family | Targets a person? | Notes |
|---|---|---|---|
| `knnvc --source bona` | WavLM kNN voice conversion | yes | converts a real person's speech into the target's voice |
| `knnvc --source tts` | TTS then voice conversion | yes | the common real-world pipeline: synthesise the words, then clone the voice |
| `speecht5x` | SpeechT5 + x-vector | yes (weakly) | what an attacker without a GPU does |
| `mms` | VITS | no (impostor that happens to be synthetic) | evaluation only, see licence |

Each target's reference audio (about 25 s for kNN-VC) is taken from clips that are
neither the speaker's enrolment clips nor their genuine test calls
(`voiceguard.eval.trials`), so a clone never hears the audio it is tested against.

## Results: the shipped system against these attacks

Same trial design and the same genuine / impostor calls as `system_eval.md`; the attack
calls are replaced by modern clones (3 clips, about 10 s, of one generator per call, made
from reference audio that is neither the enrolment nor a genuine test call). Three
conditions, identical for every call: clean, background voices at 10 dB, G.711.
`dev` = 10 speakers never used to fit anything. `eval` = 24 speakers whose *genuine and
impostor* audio the fusion was fitted on; the clone audio is new to it, so the clone rows
are a fair held-out test and the genuine rows are not. Reproduce:
`scripts/eval_stress.py --tag _modern --split dev --by-attack` (and `--split eval`).

**Update 2026-10-23 (evening): the clone head changed, and these numbers moved.** Every table
below up to the reference-length sweep was measured with the *linear* clone head, which could alarm
on recording style alone. A real WhatsApp voice note showed that is a shortcut
([`real_world_audio.md`](real_world_audio.md)), and the head was replaced by a trust-weighted one in
which the channel only sets how far the detector is believed. The same trials scored with the
shipped head (`results/tables/modern_by_attack_gain_{dev,eval}.csv`; the linear head's are the
`..._linear_...` files), share of calls flagged, linear head then shipped head:

| source | clean | babble 10 dB | G.711 | reaching ESCALATE (shipped head) |
|---|--:|--:|--:|--:|
| kNN-VC clone, dev (20 calls) | 100 % then **75 %** | 100 % then 100 % | 100 % then **95 %** | 5-25 % (was 75-100 %) |
| kNN-VC clone, eval (48 calls) | 100 % then **71 %** | 100 % then 98 % | 100 % then **83 %** | 6-19 % (was 73-98 %) |
| SpeechT5 + x-vector, dev (20 calls) | 50 % then 50 % | 55 % then 55 % | 60 % then 60 % | 25-40 % (was 35-50 %) |
| MMS-TTS, dev / eval | 100 % then 100 % | 100 % then 100 % | 100 % then 100 % | 100 % (unchanged) |

So the clean-line kNN-VC catch fell from 100 % to about 72 %, and kNN-VC no longer reaches
"Stop" on style alone. The old figure was partly a property of the recordings, not of cloning.

Share of calls flagged (VERIFY or ESCALATE) / share reaching ESCALATE, **linear clone head**:

| source | calls per condition | clean | babble 10 dB | G.711 |
|---|--:|--:|--:|--:|
| **genuine caller accepted** (dev / eval) | 20 / 48 | 100 % / 100 % | 100 % / 90 % | 100 % / 100 % |
| **kNN-VC clone**, dev | 20 | 100 / 75 % | 100 / 100 % | 100 / 95 % |
| **kNN-VC clone**, eval | 48 | 100 / 73 % | 100 / 98 % | 100 / 96 % |
| SpeechT5 + x-vector clone, dev only | 20 | **50** / 45 % | **55** / 35 % | **60** / 50 % |
| MMS-TTS (synthetic, nobody's voice), dev / eval | 20 / 48 | 100 / 100 % | 100 / 100 % | 100 / 100 % |
| a different real human (dev / eval) | 20 / 48 | 85 / 75 %, 83 / 69 % | 85 / 85 %, 88 / 63 % | 85 / 75 %, 81 / 69 % |

An attacker who masks the clone does no better. kNN-VC clones with pink noise at 20 dB,
background voices at 20 dB, room reverberation, or a speakerphone (dev, 20 calls each):
100 % flagged in all four, 55-100 % reaching ESCALATE, genuine callers accepted 100 %.

### Which signal does the catching

| source | detector score (mean) | reads with detector ≥ 0.5 | speaker similarity | reads with speaker head ≥ 0.6 |
|---|--:|--:|--:|--:|
| genuine caller | 0.04-0.05 | 0 % | 0.94-0.96 | 3-7 % |
| a different human | 0.05 | 0 % | 0.69-0.71 | 87-93 % |
| kNN-VC clone | 0.16-0.17 | **2-3 %** | 0.85-0.87 | 58-65 % |
| SpeechT5 + x-vector | 0.16 | **2 %** | 0.78 | 80 % |
| MMS-TTS | 0.44-0.46 | 40-48 % | 0.40-0.44 | 100 % |

**The acoustic detector is nearly blind to modern clones.** It was trained on 2019 attack
systems; 2-3 % of kNN-VC reads and 2 % of SpeechT5 reads reach even 0.5. Only the older
VITS-style MMS-TTS draws it in (40-48 %). What caught the kNN-VC calls, every one of them,
was not the detector's verdict but two other things:

- the **speaker head** (similarity 0.87 against 0.94-0.96 for the real person), and
- the **clone head of the fusion** (the linear one, until 2026-10-23), which scored kNN-VC calls
  0.88 on average although the raw detector score was only 0.16. It was reading the vocoder's
  output: an unusually clean floor (-34 dB below the median frame against -22 dB for real VCTK
  speech), little high-frequency energy, and a low loudness-envelope correlation (0.41 against
  0.67). That is a property of this vocoder on studio-clean reference audio, not of cloning, and
  it is exactly what noise suppression plus a low-bitrate codec does to a real voice note. In the
  80-call sweep below, 39 of the 74 flagged calls were flagged by this head alone. The shipped
  head cannot use it any more, so kNN-VC is now caught by the speaker head and, on noisy lines,
  by the detector.

**SpeechT5 shows what happens without that signature.** Its output looks like natural speech
to the channel descriptors (clone head 0.05), so only the speaker head works, and it works
only when the clone is a poor match: the calls it flagged had similarity 0.70, the ones that
escaped 0.88 (a real different person is 0.71, the real speaker 0.955).

### Does more of the customer's voice help the attacker?

The benchmark above gave the attacker about 25 s of the customer's speech. A better-prepared
attacker has more, so `scripts/eval_clone_effort.py` rebuilds the clones of the 10 `dev`
speakers from nested reference sets of 4, 8, 16 and 32 clips (12, 26, 51 and 107 s on
average; each set contains the one before and none contains an enrolment or genuine-test
clip), two clone calls per speaker and size, each through the real session stack against
that speaker's enrolment. The speaker's own genuine calls are the baseline. Reproduce:
`python scripts/eval_clone_effort.py --split dev --sizes 4,8,16,32 --calls 2` (about 75
minutes on this machine; per-call rows in `results/tables/clone_effort_dev.csv`).

| calls | reference audio | similarity to the enrolled voice | clone head (mean) | flagged | reached ESCALATE |
|---|--:|--:|--:|--:|--:|
| genuine caller (baseline) | none | 0.970 | 0.00 | 0 % | 0 % |
| clone, 4 clips | 12 s | 0.859 | 0.67 | 85 % | 60 % |
| clone, 8 clips | 26 s | 0.877 | 0.75 | 90 % | 75 % |
| clone, 16 clips | 51 s | 0.862 | 0.79 | 100 % | 85 % |
| clone, 32 clips | 107 s | 0.863 | 0.80 | 95 % | 80 % |
| **all 80 clone calls** | | 0.865 | 0.75 | **92.5 %** (74 of 80) | 75 % |

(20 calls per row. "Similarity" is the speaker model's cosine to the enrolment: a genuine
caller scores 0.94-0.99.) **This sweep was run with the linear clone head.** On the single-read
benchmark the shipped head flags 71-75 % of clean-line kNN-VC calls instead of 100 %, so expect
the sweep's 92.5 % to fall into that range; it is to be re-run once the detector is retrained
(re-running it now would measure a system that is about to change). Across the 80 calls the
speaker head alone would have flagged 35 (44 %); the clone head alone, 62.

- **More reference audio did not make the clone sound more like the customer** (0.86 at 12 s,
  0.86 at 107 s) and did not get more calls through; the differences between rows are within
  what 20 calls can show. Whose voice is cloned matters far more: the ten speakers' mean
  similarity ranges from 0.77 to 0.94.
- **The closer the clone sounds, the more often it gets through.** Flagged: similarity below
  0.85, 28 of 28; 0.85-0.92, 34 of 37; 0.92 and above, 12 of 15. This is the "protection rests
  on imperfection" point below, now measured on one generator.
- **The six calls that got through** had similarity 0.88-0.95 and a low clone-head score
  (median 0.13): their audio did not carry the vocoder's signature either, so neither head
  had anything to find.
- **The console's sample clone is one of these.** `demo_audio/cloned_knnvc.wav` (an `eval`
  speaker, 10 reference clips, made by `scripts/make_demo_clone.py` under a rule fixed before
  looking at the result) has similarity 0.928, and the system lets it through (ALLOW on a
  routine call, with a mild "a few signs of a computer-generated voice" note, after an early
  warning that eases off in real time). It is shown as the hard case on purpose, not hidden.
- **Sample size.** 80 calls from 10 speakers; calls of one speaker share an enrolment, so they
  are not independent. For 6 of 80 getting through, the 95 % interval is 3.5-15 %.

### What this means

1. **Against today's open-source clones the system flags most, not all.** With the shipped
   clone head, 71-75 % of clean-line kNN-VC calls (single reads, 68 distinct calls over dev and
   eval) and 83-100 % under background voices or a phone codec. With the previous head it was
   100 % and 92.5 % in the sweep, but that head was partly reading recording style. Plan for a
   quarter of clean-line clones of this kind getting through, more for a clone that sounds close
   to the customer.
2. **That protection rests on imperfection, not detection.** A clone that matches the
   enrolled voice about as well as the speaker's own recordings do, and whose audio looks
   natural, would give both heads nothing to find. SpeechT5 with a better-matched voice
   already gets half of its calls through. A fine-tuned or commercial voice-cloning model
   was not tested (none was available under the licence and approval constraints here) and
   is the main remaining risk. Treat the speaker head as the real control against cloning,
   and the acoustic detector as a supporting signal.
3. **The way forward is data, not fusion.** Train the detector on these generators with one
   family held out (`finetune_robust.py --modern data/modern/manifest.csv --modern-train
   knnvc,speecht5x --extra-bona <LibriSpeech manifest>`, already supported; MMS-TTS is the
   natural held-out family), then refit the fusion. The generators are in place and
   LibriSpeech is downloaded; this is roughly 3-4 hours of compute and was not done here.
4. Sample sizes: 20 (dev) and 48 (eval) calls per source per condition, and the three
   conditions replay the same calls, so they are not independent evidence. SpeechT5 was run
   on the 10 dev speakers only (about 10 s per clip). A 100 % over 68 distinct calls does not
   mean 100 % in general (the 95 % lower bound is about 95 %), and the next set of 80 did not
   reproduce it: 92.5 %, with the head that has since been replaced. Read the numbers together,
   not the best one.

## Notes from running it

- **speechbrain on Windows wants to create a symlink**, which needs a privilege an ordinary
  account lacks (`WinError 1314`). It does so for `label_encoder.txt` because it looks for
  the name `label_encoder.ckpt`; copying the file under that name into
  `models/external/spkrec-*/` makes it use the local copy, and no network at all.
- **Speed.** kNN-VC makes a clone clip in about 0.35 s on the GPU. SpeechT5 takes about 10 s
  per clip (its autoregressive loop is CPU-bound), so it was generated for the 10 dev speakers
  only (6 clips each, enough for two calls); MMS-TTS is fast.
- **MMS-TTS prints a long "newly initialized weights" warning.** It is cosmetic: all 64
  weight-normalised layers were compared with the checkpoint and match.
## Reproduce

```bash
python scripts/fetch_librispeech.py                  # download (resumes) + extract + manifests
python scripts/make_modern_clones.py --gen mms --split eval --n 120
python scripts/make_modern_clones.py --gen knnvc --source tts --split eval --speakers 24
python scripts/make_modern_clones.py --gen knnvc --source bona --split eval --speakers 24
python scripts/make_modern_clones.py --gen speecht5x --split eval --speakers 24
python scripts/build_modern_trials.py --help
```

Generated audio goes under `data/modern/` (git-ignored; audio is never committed).
