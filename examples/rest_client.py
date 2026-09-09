"""Drive a VoiceGuard session over the REST API — the whole flow, headless.

    python scripts/serve_demo.py                 # in one terminal
    python examples/rest_client.py enrol.wav benign.wav cloned.wav

Prints the decision + reason codes after each clip and the feature-only
audit trail at the end. This is the integration contract a caller (an IVR,
a contact-centre platform) would use.
"""

from __future__ import annotations

import sys

import requests

BASE = "http://127.0.0.1:8000"


def _post(path: str, **kw):
    r = requests.post(f"{BASE}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def main() -> None:
    # Windows consoles default to cp1252; reason codes carry non-Latin-1 glyphs (e.g. the rupee sign).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    if len(sys.argv) != 4:
        sys.exit("usage: rest_client.py <enrol.wav> <benign.wav> <cloned.wav>")
    enrol, benign, cloned = sys.argv[1:4]

    sid = _post("/api/session", json={"scenario": "transaction"})["session_id"]
    print(f"session {sid}")

    with open(enrol, "rb") as fh:
        print("enrol:", _post(f"/api/session/{sid}/enroll", files={"file": fh}))

    def analyse(path: str, label: str) -> None:
        with open(path, "rb") as fh:
            d = _post(f"/api/session/{sid}/analyze", files={"file": fh})
        print(f"\n[{label}]  {d['action']}  risk={round(d['fused'] * 100)}")
        for name in ("spoof", "speaker", "prosody", "context"):
            v = d.get(name)
            print(f"    {name:8s} {'--' if v is None else round(v * 100)}")
        for reason in d["reasons"]:
            print(f"    - {reason}")
        print(f"    -> {d['recommendation']}")

    analyse(benign, "benign call")

    _post(f"/api/session/{sid}/context", json={"caller_known": False, "amount": 75000})
    analyse(cloned, "cloned voice + unknown caller + large transfer")

    _post(f"/api/session/{sid}/acknowledge")
    events = requests.get(f"{BASE}/api/session/{sid}/audit", timeout=30).json()["events"]
    print(f"\naudit trail ({len(events)} events, feature-only):")
    for e in events:
        print(f"    {e['utc']}  {e['action']:12s}  fused={e['fused']}  {e['signals']}")


if __name__ == "__main__":
    main()
