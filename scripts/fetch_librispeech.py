"""Fetch LibriSpeech subsets (a second, non-VCTK bona fide domain) and index them.

    python scripts/fetch_librispeech.py                          # dev-clean + test-clean
    python scripts/fetch_librispeech.py --subsets dev-clean

Why: every bona fide clip in ASVspoof 2019 LA is VCTK (one studio corpus), so a detector
can score well by learning "VCTK studio audio = real" -- one reason the In-the-Wild number
is so much worse than the in-domain one. LibriSpeech (CC BY 4.0, OpenSLR #12: volunteers
reading audiobooks on their own microphones) gives real speech from a different world to
train against and to check false alarms on.

Source: https://www.openslr.org/12  (dev-clean 338 MB, test-clean 347 MB). The size is checked
against the server's Content-Length; the archive is extracted with a path-traversal guard
into data/external/librispeech/ (gitignored: audio is never committed) and a manifest
data/manifests/librispeech_<subset>.csv is written (speaker ids prefixed ``LS_``).
Interrupted downloads resume.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

from voiceguard.config import REPO_ROOT, load_config, resolve
from voiceguard.data import build_librispeech

URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
DEST = REPO_ROOT / "data" / "external" / "librispeech"


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, method="HEAD")
    total = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    have = path.stat().st_size if path.exists() else 0
    if have == total:
        print(f"{path.name}: already downloaded ({total / 1e6:.0f} MB)")
        return
    headers = {"Range": f"bytes={have}-"} if 0 < have < total else {}
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    mode = "ab" if headers else "wb"
    got = have if headers else 0
    with path.open(mode) as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            got += len(chunk)
            print(f"\r{path.name}: {got / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if path.stat().st_size != total:
        raise SystemExit(f"{path.name}: got {path.stat().st_size} bytes, expected {total}")


def _extract(tar_path: Path, dest: Path) -> None:
    root = dest.resolve()
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(root)) or m.issym() or m.islnk():
                raise SystemExit(f"refusing unsafe archive member: {m.name}")
        tf.extractall(dest)  # noqa: S202 -- every member was checked above


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["dev-clean", "test-clean"])
    args = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    man = resolve(load_config(), "manifests")
    for subset in args.subsets:
        tar = DEST / f"{subset}.tar.gz"
        _download(URL.format(subset=subset), tar)
        if not (DEST / "LibriSpeech" / subset).exists():
            print(f"extracting {tar.name} ...", flush=True)
            _extract(tar, DEST)
        out = build_librispeech(DEST / "LibriSpeech", subset, man)
        print(f"{subset}: manifest -> {out}")


if __name__ == "__main__":
    main()
