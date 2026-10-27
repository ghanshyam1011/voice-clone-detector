"""Cheap channel-quality descriptors: how noisy / how narrow-band is this call?

The spoof and speaker scores both drift with the acoustic channel -- a genuine
caller with other people talking nearby scores more "synthetic" than the same
caller in a quiet room. The decision layer conditions on these descriptors so
it can tell "noisy genuine call" from "synthetic call" (see
``voiceguard.risk.learned``).

Model-free, so training (``scripts/build_system_trials.py``) and serving
(``SessionManager.assess``) share this one definition. Callers pass the audio
they want described; the serving path slices the most recent
``RECENT_SECONDS`` itself so long streams stay cheap.

The first five descriptors are the ones the fusion uses (``risk.learned.CHANNEL``;
``level_db`` is informational only). The rest are *candidates*: each targets one
measured cause of the detector reading genuine speech as synthetic (a clipped
microphone, mains hum, a missing low band, reverberation). They are recorded in the
trial tables so a refit can be validated -- including on situations held out of the
fit -- before the fusion is allowed to depend on any of them.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import welch

SR = 16000
RECENT_SECONDS = 10.0
FEATURE_NAMES = (
    "dyn_db",
    "floor_rel_db",
    "level_db",
    "hf_ratio",
    "flatness",
    "crest_db",
    "flat_top_db",
    "lf_ratio_db",
    "hum_db",
    "env_acf",
)

_FRAME = 400  # 25 ms @ 16 kHz
_ENV_FRAME, _ENV_HOP, _ENV_LAG = 400, 160, 10  # 25 ms frames every 10 ms; lag = 100 ms
_HUM_WINDOW = 8192  # 0.5 s: ~2 Hz resolution, enough to separate 50/60 Hz harmonics


def _crest_db(y: np.ndarray, fr: np.ndarray, db: np.ndarray, p50: float) -> float:
    """Peak over RMS of the louder half of the frames. Speech is ~15-25 dB; hard clipping
    flattens the peaks and pulls it towards 6-10 dB."""
    loud = fr[db > p50]
    body = loud if loud.size else y
    rms = float(np.sqrt(np.mean(np.square(body, dtype=np.float64))))
    return float(20 * np.log10((float(np.max(np.abs(y))) + 1e-10) / (rms + 1e-10)))


def _flat_top_db(y: np.ndarray) -> float:
    """Share of samples within 2% of the peak, in dB. A clipped waveform sits on its ceiling
    for many samples (-10 to -25 dB); an unclipped one touches it a handful of times."""
    peak = float(np.max(np.abs(y)))
    frac = float(np.mean(np.abs(y) >= 0.98 * peak)) if peak > 1e-6 else 0.0
    return float(10 * np.log10(frac + 1e-5))


def _hum_db(y: np.ndarray, sr: int) -> float:
    """How far the mains harmonics (50 or 60 Hz, x1..x4) stand above the spectrum around
    them, in dB, for whichever mains frequency fits better. ~0-3 dB for speech or noise."""
    if len(y) < _HUM_WINDOW:
        return 0.0
    f, p = welch(y, fs=sr, nperseg=_HUM_WINDOW, noverlap=_HUM_WINDOW // 2)
    best = 0.0
    for mains in (50.0, 60.0):
        prom = []
        for k in (1, 2, 3, 4):
            c = mains * k
            peak = p[(f >= c - 1.5) & (f <= c + 1.5)].max()
            ring = p[((f >= c - 12) & (f < c - 4)) | ((f > c + 4) & (f <= c + 12))]
            prom.append(10 * np.log10((peak + 1e-12) / (np.median(ring) + 1e-12)))
        best = max(best, float(np.mean(prom)))
    return best


def _env_acf(y: np.ndarray, sr: int) -> float:
    """Autocorrelation of the loudness envelope 100 ms apart. A reverberant room carries
    each syllable's energy into the next 100 ms, so it rises (real speech: 0.67 dry ->
    0.86 reverberant); steady noise fills the dips and lowers it (0.40 at 10 dB SNR);
    babble, band-limiting, hum, music and clipping leave it within ~0.05 of dry."""
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=_ENV_FRAME, hop_length=_ENV_HOP, n_mels=40, power=2.0
    )
    e = (10 * np.log10(m + 1e-10)).mean(axis=0)
    e = e - e.mean()
    lag = _ENV_LAG
    if len(e) <= 2 * lag:
        return 0.0
    return float(np.sum(e[:-lag] * e[lag:]) / (np.sum(e * e) + 1e-9))


def channel_features(y: np.ndarray, sr: int = SR) -> dict[str, float]:
    """dyn_db: spread between loud and quiet frames (low = constant background);
    floor_rel_db: level of the quietest frames RELATIVE to the median frame (high = noisy);
    level_db: the median frame level itself -- informational, deliberately not a model input;
    hf_ratio: energy share above 3.4 kHz (low = telephone band-limiting);
    flatness: median spectral flatness (high = noise-like).

    Every model input is invariant to recording gain: an absolute noise floor would make
    the same call look "noisy" or "clean" depending on microphone gain (measured: a
    recording 20 dB quieter let 72% of clones through), so the floor is relative.

    Candidates (see module docstring), all gain-invariant as well:
    crest_db / flat_top_db: clipping; lf_ratio_db: energy share below 150 Hz (hum, rumble,
    or a speakerphone that has none); hum_db: mains-harmonic prominence;
    env_acf: loudness-envelope autocorrelation at 100 ms (reverberation raises it)."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) < 2 * _FRAME:
        y = np.pad(y, (0, 2 * _FRAME - len(y)))
    n = len(y) // _FRAME
    fr = y[: n * _FRAME].reshape(n, _FRAME)
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
    p10, p50, p90 = np.percentile(db, [10, 50, 90])
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256)) ** 2
    f = np.fft.rfftfreq(512, 1 / sr)
    tot = S.sum() + 1e-10
    flat = float(np.median(librosa.feature.spectral_flatness(S=np.sqrt(S))))
    return {
        "dyn_db": float(p90 - p10),
        "floor_rel_db": float(p10 - p50),
        "level_db": float(p50),
        "hf_ratio": float(S[f > 3400].sum() / tot),
        "flatness": flat,
        "crest_db": _crest_db(y, fr, db, float(p50)),
        "flat_top_db": _flat_top_db(y),
        "lf_ratio_db": float(10 * np.log10(S[f < 150].sum() / tot + 1e-6)),
        "hum_db": _hum_db(y, sr),
        "env_acf": _env_acf(y, sr),
    }


def describe_channel(f: dict) -> list[str]:
    """Plain-English notes on what is wrong with the line, for people rather than models.

    Each threshold was set from the 19-situation trials (genuine calls) so that it almost
    never fires on clean audio and reliably fires on the matching degradation. False alarms on
    clean-ish calls / share of the affected situation caught: clipping ``flat_top_db`` > -25 dB
    0 % / 100 %, mains hum ``hum_db`` > 8 dB 0 % / 100 %, echo ``env_acf`` > 0.82 0 % / 92 %,
    steady noise ``floor_rel_db`` > -6 dB 2.3 % / 89 % (at -8 dB it would call 9.6 % of clean
    calls noisy, so it is not used). A quiet, ordinary call returns no notes."""
    notes = []
    if f.get("flat_top_db", -99.0) > -25.0:
        notes.append("The audio is distorted, as from an overloaded microphone.")
    if f.get("hum_db", 0.0) > 8.0:
        notes.append("There is an electrical hum on the line.")
    if f.get("env_acf", 0.0) > 0.82:
        notes.append("The speaker is in an echoey room.")
    if f.get("floor_rel_db", -99.0) > -6.0:
        notes.append("There is steady background noise.")
    return notes
