"""The console's sample audio is served from a fixed catalogue, never from a path the caller
chooses, and only when the file is really there."""

import numpy as np
import pytest
import soundfile as sf

from voiceguard.serve import demo_clips


@pytest.fixture
def demo_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_clips, "DEMO_DIR", tmp_path)
    return tmp_path


def _write(path, seconds=2.0):
    sf.write(path, (0.1 * np.sin(np.arange(int(16000 * seconds)) / 20)).astype(np.float32), 16000)


def test_only_catalogued_clips_that_exist_are_offered(demo_dir):
    assert demo_clips.available() == []  # nothing on disk: nothing offered
    _write(demo_dir / "enrol.wav", 3.0)
    _write(demo_dir / "benign.wav", 2.0)
    _write(demo_dir / "not_in_the_catalogue.wav")
    got = {c["id"]: c for c in demo_clips.available()}
    assert set(got) == {"enrol", "benign"}
    assert got["enrol"]["role"] == "register" and got["benign"]["role"] == "call"
    assert got["enrol"]["truth"] is None and got["benign"]["truth"] == "real"
    assert got["enrol"]["seconds"] == 3.0
    assert all(c["title"] and c["text"] for c in got.values())


def test_the_id_is_looked_up_never_joined_onto_a_path(demo_dir):
    _write(demo_dir / "enrol.wav")
    _write(demo_dir.parent / "secret.wav")
    assert demo_clips.clip_path("enrol") == demo_dir / "enrol.wav"
    for evil in ("../secret", "..\\secret", "secret", "enrol/../../secret", "", "ENROL"):
        assert demo_clips.clip_path(evil) is None


def test_every_catalogue_entry_is_described_in_plain_words():
    assert {c.role for c in demo_clips.CLIPS.values()} == {"register", "call"}
    for cid, clip in demo_clips.CLIPS.items():
        assert 3 <= len(clip.title.split()) <= 8 and clip.text.endswith("."), cid
        assert not any(
            w in (clip.title + clip.text).lower() for w in ("aasist", "spoof", "embedding", "ema")
        ), cid


def test_every_call_says_what_it_really_is_so_the_console_can_show_how_it_did():
    for cid, clip in demo_clips.CLIPS.items():
        assert clip.truth in (("real", "fake") if clip.role == "call" else (None,)), cid
    # a description says what the audio is; it never promises what VoiceGuard will answer
    for clip in demo_clips.CLIPS.values():
        assert not any(w in clip.text.lower() for w in ("caught", "flagged", "will be", "detect"))
