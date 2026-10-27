"""LibriSpeech manifests: a second bona fide domain must index cleanly and never collide
with ASVspoof's speaker ids."""

import pandas as pd
import pytest

from voiceguard.data import build_librispeech, load_manifest
from voiceguard.data.manifests import MANIFEST_COLUMNS


def _tree(root, speakers=(("84", ["121123", "121550"]), ("1272", ["128104"]))):
    for spk, chapters in speakers:
        for ch in chapters:
            d = root / "dev-clean" / spk / ch
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"{spk}-{ch}-{i:04d}.flac").write_bytes(b"x")


def test_librispeech_manifest_schema_labels_and_speaker_prefix(tmp_path):
    _tree(tmp_path)
    out = build_librispeech(tmp_path, "dev-clean", tmp_path / "manifests", root=tmp_path)
    d = pd.read_csv(out)
    assert list(d.columns) == MANIFEST_COLUMNS
    assert len(d) == 9 and (d.label == "bonafide").all() and (d.attack_id == "bonafide").all()
    assert set(d.speaker_id) == {"LS_84", "LS_1272"}  # can never collide with ASVspoof's LA_xxxx
    assert (d.dataset == "librispeech").all() and (d.split == "dev-clean").all()
    assert all(p.endswith(".flac") and not p.startswith("/") for p in d.path)  # portable


def test_manifest_is_deterministic_and_loads_through_the_standard_loader(tmp_path):
    _tree(tmp_path)
    a = pd.read_csv(build_librispeech(tmp_path, "dev-clean", tmp_path / "m1", root=tmp_path))
    b = pd.read_csv(build_librispeech(tmp_path, "dev-clean", tmp_path / "m2", root=tmp_path))
    assert a.equals(b)
    loaded = load_manifest(tmp_path / "m1" / "librispeech_dev-clean.csv", root=tmp_path)
    assert all((tmp_path / p).exists() or str(p).startswith(str(tmp_path)) for p in loaded.path)


def test_a_missing_subset_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract"):
        build_librispeech(tmp_path, "dev-clean", tmp_path / "m")
