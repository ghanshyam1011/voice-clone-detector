"""Load ``config/default.yaml``, resolve its paths, and build a
``PreprocessConfig`` from it. No module hard-codes a path -- everything
goes through here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from voiceguard.audio.preprocess import PreprocessConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = REPO_ROOT
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """``paths.<key>`` as an absolute Path (repo-root relative if not absolute)."""
    raw = Path(cfg["paths"][key])
    return raw if raw.is_absolute() else (cfg["_root"] / raw)


def preprocess_config(cfg: dict) -> PreprocessConfig:
    a = cfg["audio"]
    return PreprocessConfig(
        sample_rate=a["sample_rate"],
        trim_enabled=a["trim"]["enabled"],
        trim_top_db=a["trim"]["top_db"],
        trim_frame_length=a["trim"]["frame_length"],
        trim_hop_length=a["trim"]["hop_length"],
        trim_pad_ms=a["trim"]["pad_ms"],
        loudness_enabled=a["loudness"]["enabled"],
        loudness_method=a["loudness"]["method"],
        target_lufs=a["loudness"]["target_lufs"],
        peak_ceiling_dbfs=a["loudness"]["peak_ceiling_dbfs"],
    )
