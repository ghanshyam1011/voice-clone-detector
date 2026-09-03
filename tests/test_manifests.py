"""Manifest build checks. Skipped automatically where the raw corpora
aren't present (e.g. CI)."""

import pandas as pd
import pytest

from voiceguard.config import load_config, resolve
from voiceguard.data import load_manifest
from voiceguard.data.manifests import MANIFEST_COLUMNS

CFG = load_config()

_LA = resolve(CFG, "asvspoof_la")
_MANIFESTS = resolve(CFG, "manifests")

needs_la = pytest.mark.skipif(not _LA.exists(), reason="ASVspoof LA not present")
needs_manifests = pytest.mark.skipif(
    not (_MANIFESTS / "asvspoof19_la_dev.csv").exists(),
    reason="manifests not built (run scripts/build_manifests.py)",
)

EXPECTED = {"train": 25380, "dev": 24844, "eval": 71237}


@needs_la
def test_build_asvspoof_la_row_counts(tmp_path):
    from voiceguard.data import build_asvspoof_la

    written = build_asvspoof_la(_LA, tmp_path)
    for split, path in written.items():
        df = pd.read_csv(path)
        assert len(df) == EXPECTED[split]
        assert list(df.columns) == MANIFEST_COLUMNS
        assert set(df["label"]) <= {"bonafide", "spoof"}
    # bonafide rows must carry attack_id == "bonafide", not "-"
    dev = pd.read_csv(written["dev"])
    assert (dev.loc[dev.label == "bonafide", "attack_id"] == "bonafide").all()


@needs_manifests
def test_built_manifests_point_at_real_files():
    df = load_manifest(_MANIFESTS / "asvspoof19_la_dev.csv")
    sample = df.sample(min(20, len(df)), random_state=0)
    from pathlib import Path

    assert sample["path"].map(lambda p: Path(p).exists()).all()
