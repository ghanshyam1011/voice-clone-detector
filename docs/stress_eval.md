# Stress evaluation: situations the system was never fitted on

`docs/system_eval.md` measures the fused decision under five conditions
(clean, two babble levels, pink noise, G.711). Those five are also what the
fusion was fitted on, so they say how well it does where it was trained. A
call centre, a car, a speakerphone and a bad VoIP link are none of those five.
This document measures the system in **fourteen situations it had never
seen**, says plainly where it failed, records what was done about it, and
separates three different questions that a single accuracy number blurs:

- **situation never seen** (`before`, and `unseen` after the fix): how it behaves in a
  place nobody planned for;
- **situation covered, person new** (`seen`): the normal operating regime, since the
  shipped model is fitted on all nineteen situations;
- **untouched dev** (`dev`): ten further speakers, never used to fit or choose anything.

Reproduce the "before" measurement:

```bash
python scripts/build_system_trials.py --split eval --speakers 20 --tag _stress --conds stress
python scripts/eval_stress.py --tag _stress           # the model that existed at the time
python scripts/fit_fusion.py --tag _v2,_stress        # leave-situations-out check
```

In the "before" tables nothing is fitted to the stress reads: `eval_stress.py` scores the
cached reads with a fusion model fitted on the five standard conditions only, at its
thresholds. Every situation is applied identically to genuine and attack calls
(`docs/evaluation_protocol.md`), so "is it degraded?" is never a shortcut.

## The situations

Defined once in `voiceguard/audio/degrade.py`, used both here and in training
augmentation (so what we test on and what we train on cannot drift apart).

| Situation | What it models |
|---|---|
| `reverb` | small-to-large room, direct sound + decaying tail |
| `speakerphone` | reverb, then a narrow loudspeaker/mic band, plus room noise |
| `narrow` | 300–3400 Hz band-limiting, no codec |
| `clip` | overdriven microphone or loud talker (hard clipping) |
| `quiet` | speaker far from the microphone (-45 dBFS) |
| `loss` | VoIP packet loss, 20–60 ms dropouts |
| `hum` | 50/60 Hz mains hum with harmonics, 12 dB SNR |
| `music` | harmonic music-like background, 8 dB SNR |
| `burst` | door slams / keyboard clacks, 8 dB SNR |
| `amr`, `opus8k`, `g722` | mobile / VoIP codecs (ffmpeg) |
| `tel_babble` | background voices at 10 dB, then AMR-NB |
| `tel_reverb` | room reverb, then AMR-NB |

**These are synthetic.** They exercise the failure modes; they are not
recordings of real rooms, streets or phone networks. A real deployment will
meet situations that are none of these.

Design: 20 `eval` speakers x 8 calls (2 genuine, 4 clone, 2 impostor; 40
genuine, 80 clone, 40 impostor reads per situation) x 14 situations = **2,240
reads**. Call context is held neutral, as in `system_eval.md`.

## Reading the tables

`acc` counts a call correct when the decision is right (flag = anything but
ALLOW). **Six of every eight calls are attacks, so a system that flags
everything scores 75 % `acc`.** Read `bal` (the mean of genuine-accepted and
attack-flagged) and `genOK` (genuine callers let through) first.

`EER` is threshold-free: it says whether the signals still *separate* genuine
from attack in that situation. Comparing it with `bal` separates two different
failures:

- **EER good, `bal` poor**: the signals still tell the classes apart, but the
  operating point moved (e.g. genuine speech scores higher than usual). A
  better-fitted fusion can recover this.
- **EER poor**: the detector lost the evidence. Nothing downstream can bring
  it back; the detector has to be trained for the situation.

## Before: the system as it stood (gain-invariant fusion fitted on the five standard conditions)

| situation | acc | bal | genOK | clone | imp | ESCgen | ESCatk | EER |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| reverb | 90.0 % | 84.2 % | 72.5 % | 95.0 % | 97.5 % | 17.5 % | 91.7 % | 10.0 % |
| speakerphone | 81.8 % | 62.8 % | 25.6 % | 100 % | 100 % | 51.3 % | 95.8 % | 23.1 % |
| narrow | 85.6 % | 90.4 % | 100 % | 72.5 % | 97.5 % | 0.0 % | 70.0 % | 5.0 % |
| clip | 77.4 % | 53.8 % | 7.7 % | 100 % | 100 % | 89.7 % | 99.2 % | 20.0 % |
| quiet | 87.5 % | 91.7 % | 100 % | 78.8 % | 92.5 % | 0.0 % | 71.7 % | 1.7 % |
| loss | 90.6 % | 81.2 % | 62.5 % | 100 % | 100 % | 25.0 % | 97.5 % | 12.5 % |
| hum | 80.6 % | 66.2 % | 37.5 % | 100 % | 85.0 % | 17.5 % | 90.8 % | 10.0 % |
| music | 80.6 % | 61.3 % | 22.5 % | 100 % | 100 % | 40.0 % | 94.2 % | 11.7 % |
| burst | 86.2 % | 90.0 % | 97.5 % | 73.8 % | 100 % | 0.0 % | 79.2 % | 6.7 % |
| amr | 96.9 % | 96.2 % | 95.0 % | 98.8 % | 95.0 % | 0.0 % | 88.3 % | 3.3 % |
| opus8k | 93.8 % | 93.3 % | 92.5 % | 92.5 % | 97.5 % | 0.0 % | 81.7 % | 5.8 % |
| g722 | 90.0 % | 93.3 % | 100 % | 83.8 % | 92.5 % | 0.0 % | 74.2 % | 2.5 % |
| tel_babble | 84.4 % | 70.4 % | 42.5 % | 100 % | 95.0 % | 25.0 % | 95.0 % | 8.3 % |
| tel_reverb | 83.0 % | 65.4 % | 30.8 % | 100 % | 100 % | 51.3 % | 100 % | 17.9 % |
| **mean of the 14** | **86.3 %** | **78.6 %** | **63.3 %** | **92.5 %** | **96.6 %** | **22.7 %** | **87.8 %** | **9.9 %** |

For reference, the same model on the untouched standard-condition `dev` split:
clean 93.8 %, babble 10 dB 95.0 %, babble 5 dB 86.2 %, pink 93.8 %, G.711 93.8 %.

What this shows:

1. **Phone codecs alone are fine** (`amr` 96.9 %, `opus8k` 93.8 %, `g722` 90.0 %;
   EER 2.5–5.8 %). Telephony is the case the product is built for, and it holds.
2. **The signals mostly still separate the classes** (EER 1.7–12.5 % in eleven of
   fourteen situations), so most of the damage is the operating point moving.
   The problem is on the genuine side: genuine callers are flagged in `clip`,
   `speakerphone`, `music`, `tel_reverb`, `hum`, `tel_babble`.
3. **Three situations have a real detector problem**: `speakerphone` (EER 23.1 %),
   `clip` (20.0 %) and `tel_reverb` (17.9 %). With a clipped microphone a
   genuine caller reaches ESCALATE 89.7 % of the time. That is not something a
   better threshold can fix.
4. **`narrow`, `quiet`, `burst`, `g722` go the other way**: genuine callers are
   accepted but 16–28 % of clones slip through, although EER is 1.7–6.7 %. The
   operating point is too lenient there.

## Can a better fusion fix it?

`fit_fusion.py --tag _v2,_stress` refits the fusion on the five standard
conditions plus the fourteen stress situations and checks it two ways:

- **Situations in the fit, unseen speakers** (4-fold, grouped by speaker):
  88.0 % accuracy, 87.7 % balanced, EER 12.3 % over all 19 conditions. Most
  situations land near their EER, so the fusion is not what limits them.
- **Situations held out as well as speakers** (leave-situations-out; related
  situations, e.g. `reverb` + `speakerphone` + `tel_reverb`, are held out
  together): 86.3 % accuracy but **79.9 % balanced**, genuine accepted 67.0 %,
  ESCALATE on genuine 11.0 %. That is roughly what the shipped model already
  gets (78.6 % balanced, 63.3 % genuine accepted, 22.7 % genuine ESCALATE).
  `clip` (5 % genuine accepted) and `music` (7.5 %) stay broken when they are
  not in the fit.

So fitting the fusion on more situations helps the situations it is fitted on
and little else: a new situation is not rescued by a fusion that has not met
it. It also appears to cost clean-audio clone recall (95 % to 85 % on the `dev`
split, which has only 40 clone reads per condition, so read it as "about ten
points, give or take five"), because one decision rule is stretched across 19
situations. Not shipped on its own.

The lever that remains is the detector itself: it has to stop reading ordinary
distortions as synthesis.

## After: what was changed, and what each change bought

Two changes, measured one at a time so each one's effect is known.

### 1. A detector trained for these situations (`AASIST_robust2`)

`scripts/finetune_robust.py --aug v2` continues from the previous robust model on
ASVspoof 2019 LA **train** only. Every clip, bona fide and spoof alike, is degraded in the
order a real call reaches the model (the line damages the audio, then our front-end runs):
background voices and noise as before, plus room reverb 15 %, speakerphone 6 %, narrow band
6 %, **clipping 10 %**, **packet loss 8 %**, mains hum 5 %. `music` and `burst` noise are
never trained on, so they stay a true unseen-situation check. Six epochs, resumable; the
epoch is chosen on dev EER (not on the stress reads) with a clean-EER budget of 5 %.

| dev EER | previous | `AASIST_robust2` |
|---|--:|--:|
| clean | 3.4 % | 3.8 % |
| 5 dB background voices | 17.8 % | 13.2 % |
| the room / line situations it was trained on | 21.0 % | 11.6 % |
| music + burst (never trained on) | 5.2 % | 4.4 % |

The standing gates still pass (`scripts/eval_cm.py --weights AASIST_robust2.pth`, 6,000
clips per class): ASVspoof dev EER 4.03 %, eval EER 20.15 %, In-the-Wild 36.18 % (upstream
weights 9.6 / 23.3 / 35.8 %), and the silence-only ablation sits at 44.75 % (no shortcut).
In-the-Wild is unchanged, within noise: the new training teaches robustness to
distortion, not new kinds of synthesis.

To isolate the detector, the fusion was refitted with the **same recipe as before**
(five standard conditions only) and scored on the same 2,240 stress reads. Nothing else
differs:

| mean of the 14 situations | accuracy | balanced | genuine accepted | clones caught | EER | genuine ESCALATE |
|---|--:|--:|--:|--:|--:|--:|
| previous detector | 86.3 % | 78.6 % | 63.3 % | 92.5 % | 9.9 % | 22.7 % |
| **`AASIST_robust2`** | **89.9 %** | **82.6 %** | **68.2 %** | **96.2 %** | **7.2 %** | **11.1 %** |

On the five standard conditions the same swap gives out-of-fold accuracy 89.5 % to
92.1 % and EER 11.2 % to 9.7 %, and the untouched dev split stays level (92.5 % to 92.8 %).
Per situation (balanced accuracy, previous to new): packet loss 81 to 97, mains hum 66 to
80, clipping 54 to 64, music 61 to 71 (never trained on), speakerphone 63 to 71, narrow
band 90 to 97. **Three got worse:** `tel_babble` 70 to 59, `tel_reverb` 65 to 62 and
`reverb` 84 to 82. The compound cases (a noisy room or an echoey one, then a mobile
codec) are still the weakest part of the system.

### 2. Descriptors that name the cause

The four channel descriptors the fusion had (dynamic range, noise floor, high-frequency
share, flatness) say a call is "degraded" but not *how*. Four more were added, each
measuring one cause of a genuine voice reading as synthetic:

| descriptor | measures | real speech, dry vs affected |
|---|---|---|
| `env_acf` | loudness-envelope autocorrelation at 100 ms | 0.67 dry, 0.86 reverberant, 0.40 in steady noise |
| `crest_db` | peak over RMS of the louder frames | 15.9 dB, 2.1 dB clipped |
| `flat_top_db` | share of samples sitting on the peak | -43 dB, -6 dB clipped |
| `hum_db` | mains-harmonic (50/60 Hz) prominence | 0.5 dB, 18.3 dB with hum |

A fifth, a low-frequency share, was tried and dropped because it made results slightly
worse. A first reverberation measure (a modulation-energy ratio) did not respond to
reverberation on real speech at all and was replaced by `env_acf` before it was used.

Balanced accuracy, mean of the situations, linear logistic heads (the form that shipped until
2026-10-23; the clone head has since been replaced, see "The shipped system" below):

| descriptors | `seen` (situation in the fit, new speakers) | `unseen` (situation AND speaker held out) | genuine ESCALATE, seen |
|---|--:|--:|--:|
| the original four | 88.7 % | 83.1 % | 4.1 % |
| + `env_acf` | 92.9 % | 88.7 % | 3.6 % |
| **+ `env_acf`, `crest_db`, `flat_top_db`, `hum_db` (shipped until 2026-10-23)** | **93.4 %** | **89.1 %** | **3.2 %** |

`env_acf` alone does most of the work (speakerphone 77 to 89, music 74 to 90, reverb plus
codec 67 to 82 when held out). The clipping and hum descriptors pay off only where those
situations are in the fit, which is the shipped regime: clipping 76.5 to **91.5**, mains
hum 92.1 to **98.3**. Held out, they cannot help, because a model cannot learn what a
clipped signal looks like from no clipped examples.

Boosted-tree heads (monotone in the detector score; `fit_fusion.py --head gbm`, exported to
plain lists so serving needs no scikit-learn) were also built and tested. They beat the
logistic heads on the original four descriptors (85.5 % vs 83.1 % unseen) but not once the
new descriptors exist (88.3 % vs 89.1 %), so the simpler logistic model ships. The option
is kept, tested, and not used.

### The shipped system

Fitted on 48 eval speakers x 19 situations (4,157 reads) with a 3 % cap on genuine
ESCALATE. Balanced accuracy, mean over situations.

**Update 2026-10-23 (evening): the clone head changed.** The linear heads described above could
alarm on recording style alone, and a real WhatsApp voice note was called synthetic because of it
(`real_world_audio.md`). The clone head is now trust-weighted: the channel only sets how far the
detector is believed. The numbers below are for that shipped model; the linear head's are in the
last column for comparison (`results/tables/clone_head_variants.csv`, same reads, same 3 % cap).

| regime | balanced | genuine accepted | clones / impostors caught | genuine ESCALATE | linear head, balanced |
|---|--:|--:|--:|--:|--:|
| before: 14 situations never seen | 78.6 % | 63.3 % | 92.5 % / 96.6 % | 22.7 % | 78.6 % |
| after, situation held out as well as speaker | 86.6 % | 79.2 % | 93.7 % / 94.6 % | 8.6 % | 89.2 % |
| after, situation in the fit, new speakers (all 19) | 90.2 % | 89.0 % | 92.1 % / 89.8 % | 3.1 % | 92.9 % |
| **after, untouched dev: 10 new speakers x 14 situations** | **91.3 %** | **89.6 %** | **97.5 % / 83.9 %** | **1.8 %** | 92.6 % |

The two to three points are the shortcut being taken out, not a regression in what the system
can do: on ASVspoof's recordings the style of a clip separates genuine from attack, and on a
real voice note it does not.

Untouched dev, per situation (80 reads each: 20 genuine, 40 clone, 20 impostor):

| situation | balanced | genuine accepted | | situation | balanced | genuine accepted |
|---|--:|--:|---|---|--:|--:|
| reverb | 89.2 % | 85 % | | burst | 96.7 % | 100 % |
| speakerphone | 85.0 % | 75 % | | amr | 96.7 % | 100 % |
| narrow | 95.0 % | 100 % | | opus8k | 95.8 % | 100 % |
| clip | 80.8 % | 70 % | | g722 | 96.7 % | 100 % |
| quiet | 96.7 % | 100 % | | **tel_babble** | **78.3 %** | **60 %** |
| loss | 96.7 % | 100 % | | tel_reverb | 82.5 % | 70 % |
| hum | 93.3 % | 95 % | | music | 95.0 % | 100 % |

Reproduce:

```bash
python scripts/finetune_robust.py --aug v2 --init AASIST_robust.pth --out AASIST_robust2.pth --epochs 6 --per-class 4000 --max-clean-eer 0.05 --workers 4
python scripts/eval_cm.py --model aasist --weights AASIST_robust2.pth --limit-per-class 6000 --workers 4
python scripts/build_system_trials.py --split eval --tag _r2 --weights AASIST_robust2.pth
python scripts/build_system_trials.py --split dev --tag _r2 --weights AASIST_robust2.pth
python scripts/build_system_trials.py --split eval --speakers 20 --conds stress --tag _r2s --weights AASIST_robust2.pth
python scripts/build_system_trials.py --split dev --speakers 10 --conds stress --tag _r2s --weights AASIST_robust2.pth
python scripts/fit_fusion.py --tag _r2,_r2s --esc-fpr 0.03 --ship --channel dyn_db,floor_rel_db,hf_ratio,flatness,env_acf,crest_db,flat_top_db,hum_db
python scripts/eval_stress.py --tag _r2s --split dev
```

## Honest limits

- **Every situation here is synthetic.** The fusion is fitted on these same synthetic
  situations, so the `seen` and `dev` rows say it handles *these*. A situation nobody
  planned for is the `unseen` row, 87 %, and that is the number to plan with, not 90 %.
- **Noisy room or echo, then a mobile codec, is still weak.** On the untouched dev split
  `tel_babble` accepts 60 % of genuine callers (`tel_reverb` 70 %), and held out it is 47.5 %.
  This is the one case worth real recordings.
- **A clipped microphone the system has never met is still misread.** Held out, genuine
  callers are accepted 8 % of the time (the audio evidence is damaged); with clipping in
  the fit it is 70-72 %.
- **Everything above is about distortion, not about newer cloning tools.** The detector is
  nearly blind to modern voice-conversion and TTS clones (2-3 % of reads); the system still
  flags 71-75 % of clean-line kNN-VC calls (the linear head flagged all of them, partly by reading
  recording style) and half of the SpeechT5 ones, through the speaker head. See
  [`modern_attacks.md`](modern_attacks.md) and [`real_world_audio.md`](real_world_audio.md).
- **Impostors (a different human, not a clone) are caught 85 %** on dev. That is the speaker
  model's limit, not the detector's; see `docs/system_eval.md`.
- **Small samples.** A dev cell is 20 genuine and 60 attack reads: a single call moves it
  by 5 points. The pooled rows are the ones to quote, and `eval` rows are in-sample for the
  final model (it is fitted on those speakers), so they are not quoted above.
- ASVspoof clone calls mix three clips that may come from different attack systems, so the
  per-attack breakdown in `eval_stress.py --by-attack` is indicative.
