"""Build data/manifests/*.csv from the official protocol files.

python scripts/build_manifests.py
"""

from __future__ import annotations

from voiceguard.config import load_config, resolve
from voiceguard.data import build_asvspoof_la, build_in_the_wild


def main() -> None:
    cfg = load_config()
    manifests_dir = resolve(cfg, "manifests")

    la_root = resolve(cfg, "asvspoof_la")
    if la_root.exists():
        for split, path in build_asvspoof_la(la_root, manifests_dir).items():
            print(f"  asvspoof19_la {split:5s} -> {path}")
    else:
        print(f"  SKIP asvspoof_la (not found at {la_root})")

    itw_root = resolve(cfg, "in_the_wild")
    if (itw_root / "meta.csv").exists():
        print(f"  in_the_wild eval  -> {build_in_the_wild(itw_root, manifests_dir)}")
    else:
        print(f"  SKIP in_the_wild (no meta.csv at {itw_root})")


if __name__ == "__main__":
    main()
