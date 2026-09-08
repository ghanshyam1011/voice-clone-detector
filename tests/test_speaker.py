"""Speaker embedding + per-session verification.

The embedding model is downloaded on first use (~few hundred MB). These
tests skip cleanly when it is not available (offline CI, first run before
the demo warm-up)."""

import numpy as np
import pytest

from voiceguard.speaker import SpeakerEmbedder, SpeakerVerifier
from voiceguard.speaker.embed import cosine


def _voiceish(seed: int, secs: float = 3.0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * sr)) / sr
    f0 = 100 + 40 * rng.random()
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t) for k in (1, 2, 3, 4))
    sig += 0.05 * rng.standard_normal(t.shape)
    return (0.2 * sig / np.max(np.abs(sig))).astype(np.float32)


@pytest.fixture(scope="module")
def embedder():
    emb = SpeakerEmbedder()
    try:
        emb.warm()
    except Exception as exc:  # noqa: BLE001 -- model download unavailable
        pytest.skip(f"speaker model unavailable: {exc}")
    return emb


def test_cosine_bounds():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert cosine(a, np.zeros(3)) == 0.0


def test_embed_shape_and_norm(embedder):
    e = embedder.embed(_voiceish(1))
    assert e.ndim == 1 and e.shape[0] == embedder.dim
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)


def test_same_signal_more_similar_than_different(embedder):
    v = SpeakerVerifier(embedder)
    v.enroll("s1", _voiceish(1))
    same = v.score("s1", _voiceish(1))
    diff = v.score("s1", _voiceish(999))
    assert same.enrolled and diff.enrolled
    assert same.similarity > diff.similarity
    assert 0.0 <= same.risk <= 1.0 and 0.0 <= diff.risk <= 1.0


def test_score_before_enrol_is_unavailable(embedder):
    v = SpeakerVerifier(embedder)
    r = v.score("nope", _voiceish(1))
    assert r.enrolled is False and r.risk is None
