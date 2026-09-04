"""Build dataset manifests from official protocol files.

Manifest schema (CSV, one row per clip):
    dataset      e.g. "asvspoof19_la", "in_the_wild"
    split        "train" | "dev" | "eval"
    path         path to the audio file, RELATIVE to the repo root when the
                 file lives under it (so the manifest is portable and can be
                 committed); resolved to absolute by ``load_manifest``
    label        "bonafide" | "spoof"
    attack_id    "bonafide" for genuine; "A01".."A19" or "unknown" for spoof
    speaker_id   as given by the corpus
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from voiceguard.config import REPO_ROOT

MANIFEST_COLUMNS = ["dataset", "split", "path", "label", "attack_id", "speaker_id"]

_CM_PROTOCOL = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
_CM_AUDIO_SUBDIR = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
}

# expected row counts, as a sanity check against a truncated download
_EXPECTED_ROWS = {"train": 25380, "dev": 24844, "eval": 71237}


def _portable(p: Path, root: Path = REPO_ROOT) -> str:
    """Repo-root-relative POSIX string when possible, else absolute.

    Pure string math -- no filesystem calls (this runs per clip, ~120k times).
    """
    try:
        return Path(os.path.relpath(p, root)).as_posix()
    except ValueError:  # different drive on Windows
        return str(p)


def build_asvspoof_la(
    la_root: str | Path, out_dir: str | Path, *, root: Path = REPO_ROOT
) -> dict[str, Path]:
    """Write ``asvspoof19_la_{train,dev,eval}.csv`` into ``out_dir``."""
    la_root, out_dir = Path(la_root), Path(out_dir)
    proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split, proto_name in _CM_PROTOCOL.items():
        raw = pd.read_csv(
            proto_dir / proto_name,
            sep=r"\s+",
            header=None,
            names=["speaker_id", "file_id", "_env", "attack_id", "label"],
        )
        if len(raw) != _EXPECTED_ROWS[split]:
            raise ValueError(
                f"{proto_name}: got {len(raw)} rows, expected {_EXPECTED_ROWS[split]} "
                "-- protocol file looks wrong or truncated"
            )
        audio_dir = la_root / _CM_AUDIO_SUBDIR[split]
        out = pd.DataFrame(
            {
                "dataset": "asvspoof19_la",
                "split": split,
                "path": [_portable(audio_dir / f"{fid}.flac", root) for fid in raw["file_id"]],
                "label": raw["label"],
                "attack_id": raw["attack_id"].replace("-", "bonafide"),
                "speaker_id": raw["speaker_id"],
            },
            columns=MANIFEST_COLUMNS,
        )
        dest = out_dir / f"asvspoof19_la_{split}.csv"
        out.to_csv(dest, index=False)
        written[split] = dest
    return written


def build_in_the_wild(itw_root: str | Path, out_dir: str | Path, *, root: Path = REPO_ROOT) -> Path:
    """Write ``in_the_wild_eval.csv``. In-the-Wild is evaluation-only; it
    never carries a train/dev split and must never be trained on."""
    itw_root, out_dir = Path(itw_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(itw_root / "meta.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    label_map = {
        "bona-fide": "bonafide",
        "bonafide": "bonafide",
        "real": "bonafide",
        "spoof": "spoof",
        "fake": "spoof",
    }
    out = pd.DataFrame(
        {
            "dataset": "in_the_wild",
            "split": "eval",
            "path": [_portable(itw_root / f, root) for f in meta["file"]],
            "label": meta["label"].str.strip().str.lower().map(label_map),
            "attack_id": "unknown",
            "speaker_id": meta.get("speaker", "unknown"),
        },
        columns=MANIFEST_COLUMNS,
    )
    if out["label"].isna().any():
        raise ValueError("in-the-wild meta.csv has labels outside the known set")
    dest = out_dir / "in_the_wild_eval.csv"
    out.to_csv(dest, index=False)
    return dest


def load_manifest(path: str | Path, *, root: Path = REPO_ROOT) -> pd.DataFrame:
    """Load a manifest and resolve every ``path`` to an absolute path."""
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: manifest missing columns {sorted(missing)}")
    df["path"] = df["path"].map(
        lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    )
    return df
