"""Screenshots of the operator console's answer panel, for the poster.

    python scripts/serve_demo.py                      # in one terminal
    python scripts/capture_console_shots.py           # writes docs/poster/shots/*.png

Drives a throw-away headless Chrome over the DevTools protocol: it opens the running console,
plays a sample call exactly as a person would (click "Play", wait for "Call finished") and saves
the answer panel at 2x resolution, whole (``<name>.png``) and cut after the "What to do now" box
(``<name>_top.png``, the part that fits a poster panel). The scorecard text is checked against
what the clip should produce, so a poster never shows a screenshot of a failed run. Only the
built-in demo clips are played, never anyone's own recording. Needs Chrome or Edge installed;
nothing is downloaded.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "poster" / "shots"
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
# (clip id in the console, output name, scorecard headline it should end on):
# a real customer, an obvious fake, and the hard case that is only partly caught
SHOTS = (
    ("benign", "console_real", "Correct"),
    ("cloned_2", "console_fake", "Caught"),
    ("cloned_knnvc", "console_hard", "Partly caught"),
)
PORT = 9333


class Cdp:
    """Just enough of the Chrome DevTools protocol to click, wait and take a screenshot."""

    def __init__(self, ws_url: str) -> None:
        self._conn = connect(ws_url, max_size=128 * 1024 * 1024)
        self.n = 0

    def __enter__(self) -> Cdp:
        self.ws = self._conn.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._conn.__exit__(*exc)

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=90))
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, js: str):
        r = self.call("Runtime.evaluate", expression=js, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("text", "script error"))
        return r.get("result", {}).get("value")

    def wait_for(self, js: str, what: str, timeout: float = 60.0) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.eval(js):
                return
            time.sleep(0.4)
        raise TimeoutError(f"gave up waiting for {what}")

    def shot(self, x: float, y: float, w: float, h: float) -> bytes:
        r = self.call(
            "Page.captureScreenshot",
            format="png",
            captureBeyondViewport=True,
            clip={"x": x, "y": y, "width": w, "height": h, "scale": 1},
        )
        return base64.b64decode(r["data"])


def _find_browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    raise SystemExit("Chrome or Edge was not found; install one or edit BROWSERS.")


def _get(path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
        return json.load(r)


# where the panel is on the page, and where its "What to do now" box ends
_RECTS = """(() => {
  const a = document.getElementById('answer').getBoundingClientRect();
  const t = document.getElementById('todo');
  const tb = t && !t.hidden ? t.getBoundingClientRect().bottom : a.top + 520;
  return [a.x + scrollX, a.y + scrollY, a.width, a.height, tb + scrollY];
})()"""


def _capture(cdp: Cdp, clip: str, name: str, expect: str) -> None:
    cdp.eval(f"document.querySelector('.clip[data-id=\"{clip}\"]').click()")
    cdp.wait_for("!!document.querySelector('.clip.playing')", f"'{clip}' to start", timeout=30)
    cdp.wait_for(
        "!document.querySelector('.clip.playing') && "
        "document.getElementById('call-status').textContent.includes('Call finished')",
        f"'{clip}' to finish",
        timeout=90,
    )
    time.sleep(1.0)  # let the last animation settle
    score = cdp.eval("document.getElementById('score').innerText") or ""
    head = score.strip().splitlines()[0] if score.strip() else "(no scorecard)"
    if head != expect:
        raise SystemExit(f"'{clip}' ended on '{head}', expected '{expect}'; not saving a bad shot")
    x, y, w, h, todo_bottom = cdp.eval(_RECTS)
    (OUT / f"{name}.png").write_bytes(cdp.shot(x, y, w, h))
    (OUT / f"{name}_top.png").write_bytes(cdp.shot(x, y, w, todo_bottom - y + 20))
    words = cdp.eval("document.getElementById('answer').innerText")
    (OUT / f"{name}.txt").write_text(words, encoding="utf-8")
    print(f"wrote {name}.png and {name}_top.png  ({int(w)} css px wide, scorecard: {head})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="vg_shots_")
    proc = subprocess.Popen(
        [
            _find_browser(),
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--autoplay-policy=no-user-gesture-required",
            "--hide-scrollbars",
            f"--window-size={args.width},1100",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            try:
                target = next(t for t in _get("/json/list") if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("the browser did not start")
        with Cdp(target["webSocketDebuggerUrl"]) as cdp:
            cdp.call("Page.enable")
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                width=args.width,
                height=1100,
                deviceScaleFactor=args.scale,
                mobile=False,
            )
            cdp.call("Page.navigate", url=args.url)
            cdp.wait_for(
                "!!document.getElementById('reg-status') && "
                "document.getElementById('reg-status').className.includes('ok')",
                "the sample customer's voice to be put on file",
                timeout=60,
            )
            for clip, name, expect in SHOTS:
                _capture(cdp, clip, name, expect)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
