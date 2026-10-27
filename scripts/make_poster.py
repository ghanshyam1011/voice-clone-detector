"""The VoiceGuard poster: A1 portrait, print-ready PDF + PNG preview, built from the repository's own data.

    python scripts/make_poster.py                                   # docs/poster/voiceguard_poster_A1.{html,pdf,png}
    python scripts/make_poster.py --subject "Machine Learning" --faculty "Prof. X" --college "Y" \
        --members "Asha Rao, 21CS001; Ravi Kumar, 21CS002; Meena Shah, 21CS003"
    python scripts/make_poster.py --check                           # also report any box whose text overflows

Sections follow the college checklist: title, group members with roll numbers, subject and faculty,
problem statement, objectives, methodology / workflow diagram, technologies and tools used, results,
conclusion, and two signature boxes (faculty, external examiner) at the bottom.

Every number is read from ``results/tables/`` (or quoted, with its source, in ``docs/``), the same
tables ``scripts/make_poster_figures.py`` draws from, so the poster can be regenerated and checked
rather than redrawn. Charts are drawn here as vector SVG at poster size; the console screens come from
``scripts/capture_console_shots.py`` (real runs of the built-in demo clips). Rendering uses the
Chrome or Edge already installed; nothing is downloaded. The PDF is vector and scales to A0-A3.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qrcode_svg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
SHOTS = ROOT / "docs" / "poster" / "shots"
OUT = ROOT / "docs" / "poster"
REPO_URL = "https://github.com/ghanshyam1011/voice-clone-detector"
BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)
FONTS = Path(r"C:\Windows\Fonts")

INK, MUTED, LINE = "#17202B", "#566678", "#D3DBE5"
BLUE, GREEN, AMBER, RED, GREY = "#2457D6", "#17794A", "#B86F00", "#B3261E", "#8A97A5"
BLUE_L = "#7A9BE8"
W, H = 2245, 3178  # A1 portrait at 96 dpi (594 x 841 mm)


def r0(v: float) -> int:
    """Round half up (Python's round() goes to the even number, which reads oddly on a poster)."""
    return int(math.floor(v + 0.5))


def parse_members(s: str | None) -> list[tuple[str, str]]:
    """'Asha Rao, 21CS001; Ravi Kumar, 21CS002' -> [('Asha Rao', '21CS001'), ('Ravi Kumar', '21CS002')]."""
    out = []
    for item in (s or "").split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, roll = item.rpartition(",") if "," in item else (item, "", "")
        out.append((name.strip() or item, roll.strip()))
    return out


# ----------------------------------------------------------------------------- data --
def _eer(csv: str, split: str) -> float:
    d = pd.read_csv(TAB / csv)
    return float(d[d.split == split].value_pct.iloc[0])


def load_numbers() -> dict:
    """Every number on the poster. Keep in step with scripts/make_poster_figures.py."""
    v = pd.read_csv(TAB / "clone_head_variants.csv")
    gain = v[v.clone_head.str.contains("gain")]
    n: dict = {}
    # 1: honest gap, equal error rate (%). Sources: README table; cm_aasist-*.csv
    n["gap_base"] = [7.0, 33.4, 58.2]
    n["gap_pre"] = [
        _eer("cm_aasist-pretrained_in_domain.csv", "dev_known_attacks"),
        _eer("cm_aasist-pretrained_in_domain.csv", "eval_unknown_attacks"),
        _eer("cm_aasist-pretrained_cross_dataset.csv", "cross_dataset"),
    ]
    n["gap_ours"] = [
        _eer("cm_aasist-AASIST_robust2_in_domain.csv", "dev_known_attacks"),
        _eer("cm_aasist-AASIST_robust2_in_domain.csv", "eval_unknown_attacks"),
        _eer("cm_aasist-AASIST_robust2_cross_dataset.csv", "cross_dataset"),
    ]
    # 2: silence-only error. README 2.1 / docs/evaluation_protocol.md; cm_*_silence_ablation.csv
    n["silence"] = [
        14.0,
        59.0,
        _eer("cm_aasist-pretrained_silence_ablation.csv", "silence_only"),
        _eer("cm_aasist-AASIST_robust2_silence_ablation.csv", "silence_only"),
    ]
    # 4: balanced accuracy on unseen speakers. docs/system_eval.md; clone_head_variants.csv
    n["fusion"] = [
        73.6,
        83.9,
        float(gain[gain.evaluation == "speaker-held-out (OOF)"].balanced.iloc[0]),
    ]
    n["fusion_dev"] = float(gain[gain.evaluation == "untouched dev"].balanced.iloc[0])
    # 5: fourteen awkward situations. docs/stress_eval.md; stress_eval_final_dev.csv
    n["stress"] = [78.6, 86.6, 91.3]
    s = pd.read_csv(TAB / "stress_eval_final_dev.csv").sort_values("bal")
    n["stress_weakest"] = [(r.cond, float(r.bal) * 100) for r in s.head(4).itertuples()]
    # 6: modern clones, share flagged by the shipped system. modern_by_attack_gain_{dev,eval}.csv
    t = pd.concat([pd.read_csv(TAB / f"modern_by_attack_gain_{sp}.csv") for sp in ("dev", "eval")])

    def pooled(attack: str, cond: str) -> float:
        x = t[(t.attack == attack) & (t.cond == cond)]
        return float((x.flagged * x.n).sum() / x.n.sum() * 100)

    n["modern"] = {
        a: [pooled(a, c) for c in ("clean", "babble10", "ulaw")]
        for a in ("knnvc", "speecht5x", "mms")
    }
    return n


# ---------------------------------------------------------------------- text metrics --
_font_cache: dict = {}


class _Estimate:
    """Width estimate used when the Windows fonts are missing (the page still renders, in a fallback font)."""

    def __init__(self, size: int, bold: bool) -> None:
        self.k = size * (0.56 if bold else 0.5)

    def getlength(self, s: str) -> float:
        return self.k * len(s)


def _font(kind: str, size: int):
    key = (kind, size)
    if key not in _font_cache:
        try:
            if kind == "d":  # Bahnschrift Bold
                f = ImageFont.truetype(str(FONTS / "bahnschrift.ttf"), size)
                try:
                    f.set_variation_by_axes([700, 100])
                except Exception:
                    pass
            else:
                f = ImageFont.truetype(
                    str(FONTS / ("segoeuib.ttf" if kind == "b" else "segoeui.ttf")), size
                )
        except OSError:
            f = _Estimate(size, kind in ("b", "d"))
        _font_cache[key] = f
    return _font_cache[key]


def text_w(s: str, size: int, kind: str = "r") -> float:
    return _font(kind, size).getlength(s)


def wrap(s: str, size: int, max_w: float, kind: str = "r") -> list[str]:
    """Greedy word wrap using the real font metrics (with a 3 % safety margin)."""
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if text_w(trial, size, kind) * 1.03 <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# --------------------------------------------------------------------------- svg bits --
class Svg:
    def __init__(self, w: float, h: float, cls: str = "") -> None:
        self.w, self.h, self.cls, self.parts = w, h, cls, []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, rx=0, stroke=None, sw=2, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st} {extra}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def text(self, x, y, s, size=26, fill=INK, kind="r", anchor="start", extra=""):
        cls = {"d": "d", "b": "b", "r": ""}[kind]
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" class="{cls}" {extra}>{esc(s)}</text>'
        )

    def lines(self, x, y, ls, size=26, fill=INK, kind="r", anchor="start", lh=1.25):
        for i, s in enumerate(ls):
            self.text(x, y + i * size * lh, s, size, fill, kind, anchor)

    def svg(self) -> str:
        return f'<svg class="{self.cls}" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg">{"".join(self.parts)}</svg>'


ICONS = {
    "phone": '<rect x="14" y="5" width="20" height="38" rx="5"/><path d="M21 10h6"/><circle cx="24" cy="37" r="1.5"/>',
    "gate": '<path d="M24 4l16 6v11c0 11-7 19-16 23C15 40 8 32 8 21V10z"/><path d="M16 24l6 6 11-12"/>',
    "sliders": '<path d="M14 8v32M24 8v32M34 8v32"/><circle cx="14" cy="30" r="3.6"/><circle cx="24" cy="16" r="3.6"/><circle cx="34" cy="26" r="3.6"/>',
    "wave": '<path d="M9 21v6M16 14v20M23 7v34M30 13v22M37 19v10"/>',
    "match": '<circle cx="18" cy="24" r="10"/><circle cx="30" cy="24" r="10"/>',
    "mic": '<rect x="18" y="5" width="12" height="23" rx="6"/><path d="M11 23a13 13 0 0 0 26 0M24 36v7M17 43h14"/>',
    "clip": '<rect x="10" y="8" width="28" height="34" rx="4"/><path d="M18 5h12v7H18zM16 22h16M16 29h16M16 36h10"/>',
    "merge": '<path d="M5 12h9c9 0 9 12 18 12M5 36h9c9 0 9-12 18-12M5 24h27M32 24h9M36 18l6 6-6 6"/>',
    "dial": '<path d="M8 35a16 16 0 1 1 32 0"/><path d="M24 35l8-12"/><circle cx="24" cy="35" r="2.6"/>',
}

ARROW = '<svg width="26" height="26" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 24h34M27 11l13 13-13 13"/></svg>'


def inline_icon(name: str, size: int, color: str, x: float, y: float, sw: float = 3.2) -> str:
    return f'<g transform="translate({x:.1f} {y:.1f}) scale({size / 48:.4f})" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>'


# ---------------------------------------------------------------------------- charts --
CHART_H = 218  # height of the four charts in the results row


def chart_gap(n: dict, w: int = 522, h: int = 306) -> str:
    """Fig 1: error rate of three detectors on three kinds of audio (vertical grouped bars)."""
    s = Svg(w, h)
    top, bottom = 64, 96
    ph = h - top - bottom
    ymax = 60.0
    series = [
        ("Handcrafted", n["gap_base"], GREY),
        ("Published AASIST", n["gap_pre"], BLUE_L),
        ("VoiceGuard", n["gap_ours"], BLUE),
    ]
    groups = [
        ("Known attacks", "same as training"),
        ("Unseen attacks", "new methods"),
        ("Real-world", "In-the-Wild audio"),
    ]
    gw = w / 3
    bw, gap = 48, 4
    base_y = top + ph
    for frac in (0.5, 1.0):
        s.line(0, base_y - frac * ph, w, base_y - frac * ph, "#E6ECF3", 1.5)
    s.line(0, base_y, w, base_y, "#9AA8B6", 2)
    for gi, (g1, g2) in enumerate(groups):
        x0 = gi * gw + (gw - (3 * bw + 2 * gap)) / 2
        for si, (_, vals, col) in enumerate(series):
            v = vals[gi]
            bh = max(v / ymax * ph, 3)
            x = x0 + si * (bw + gap)
            s.rect(x, base_y - bh, bw, bh, col, rx=7)
            s.text(x + bw / 2, base_y - bh - 8, f"{r0(v)}%", 25, INK, "d", "middle")
        cx = gi * gw + gw / 2
        s.text(cx, base_y + 28, g1, 21, INK, "b", "middle")
        s.text(cx, base_y + 52, g2, 19, MUTED, "r", "middle")
    # the authors' own claim, as a dashed marker just above the axis
    y1 = base_y - 1.0 / ymax * ph
    s.line(0, y1, w, y1, GREEN, 2.5, "7 6")
    # legend: one row of three swatches on top, the authors' claim under the axis labels
    lx = 2
    for name, _, col in series:
        s.rect(lx, 4, 19, 19, col, rx=5)
        s.text(lx + 27, 20, name, 21, INK)
        lx += 27 + text_w(name, 21) + 20
    ky = base_y + 84
    s.line(2, ky - 7, 30, ky - 7, GREEN, 3, "6 4")
    s.text(38, ky, "≈1%: the authors' own claim", 21, GREEN, "b")
    return s.svg()


def chart_silence(n: dict, w: int = 446, h: int = CHART_H) -> str:
    """Fig 2: error from the silent parts alone (horizontal bars, red zone below 40 %)."""
    s = Svg(w, h)
    rows = [
        ("First pipeline", "before the fix", n["silence"][0], RED),
        ("Handcrafted", "after the fix", n["silence"][1], GREEN),
        ("Published AASIST", "after the fix", n["silence"][2], GREEN),
        ("VoiceGuard", "after the fix", n["silence"][3], GREEN),
    ]
    lab_w = 200
    px0, px1 = lab_w, w - 58
    scale = (px1 - px0) / 70.0
    top, rh = 24, 41
    zone_x = px0 + 40 * scale
    halo = 'style="paint-order:stroke;stroke:#fff;stroke-width:7px;stroke-linejoin:round"'
    s.rect(px0, top, zone_x - px0, rh * 4, "#FBE3E0")
    s.text((px0 + zone_x) / 2, top - 7, "shortcut", 21, RED, "b", "middle")
    s.line(px0 + 50 * scale, top, px0 + 50 * scale, top + rh * 4 + 6, INK, 2, "6 5")
    s.text(px0 + 50 * scale, top + rh * 4 + 26, "50% = guessing", 21, INK, "b", "middle")
    for i, (a, b, v, col) in enumerate(rows):
        y = top + i * rh
        s.text(0, y + 18, a, 22, INK, "b")
        s.text(0, y + 36, b, 19, MUTED)
        s.rect(px0, y + (rh - 24) / 2, v * scale, 24, col, rx=6)
        s.text(px0 + v * scale + 8, y + rh / 2 + 9, f"{r0(v)}%", 26, INK, "d", extra=halo)
    return s.svg()


def chart_fusion(n: dict, w: int = 446, h: int = CHART_H) -> str:
    """Fig 4: balanced accuracy on speakers never seen: detector alone, hand-set mix, learned fusion."""
    s = Svg(w, h)
    rows = [
        ("The detector alone", n["fusion"][0], BLUE_L),
        ("Hand-set mix of the signals", n["fusion"][1], BLUE_L),
        ("VoiceGuard's learned fusion", n["fusion"][2], BLUE),
        ("…on 10 untouched speakers", n["fusion_dev"], GREEN),
    ]
    px1 = w - 74
    lo, hi = 50.0, 100.0
    scale = px1 / (hi - lo)
    rh = 47
    for i, (name, v, col) in enumerate(rows):
        y = 2 + i * rh
        s.text(0, y + 19, name, 23, INK, "b")
        s.rect(0, y + 26, (v - lo) * scale, 18, col, rx=5)
        s.text((v - lo) * scale + 8, y + 43, f"{v:.1f}%", 25, INK, "d")
    ay = 2 + 4 * rh + 2
    s.line(0, ay, px1, ay, "#9AA8B6", 2)
    s.text(0, ay + 22, "50% = guessing", 20, MUTED)
    s.text(px1, ay + 22, "100%", 20, MUTED, "r", "end")
    return s.svg()


def chart_stress(n: dict, w: int = 446, h: int = CHART_H) -> str:
    """Fig 5: the 14 awkward situations: before, after, and on untouched speakers."""
    s = Svg(w, h)
    rows = [
        ("Before: situations never seen", n["stress"][0], RED),
        ("After: the same situations", n["stress"][1], AMBER),
        ("After: untouched speakers", n["stress"][2], GREEN),
    ]
    px1 = w - 80
    lo, hi = 60.0, 100.0
    scale = px1 / (hi - lo)
    rh = 62
    for i, (name, v, col) in enumerate(rows):
        y = 2 + i * rh
        s.text(0, y + 22, name, 24, INK, "b")
        s.rect(0, y + 30, (v - lo) * scale, 24, col, rx=6)
        s.text((v - lo) * scale + 8, y + 51, f"{v:.1f}%", 26, INK, "d")
    ay = 2 + 3 * rh + 2
    s.line(0, ay, px1, ay, "#9AA8B6", 2)
    s.text(0, ay + 22, "60%", 20, MUTED)
    s.text(px1, ay + 22, "100%", 20, MUTED, "r", "end")
    return s.svg()


_SITUATION_NAMES = {
    "tel_babble": "phone + crowd noise",
    "clip": "clipped microphone",
    "tel_reverb": "phone + echoey room",
    "speakerphone": "speakerphone",
    "reverb": "echoey room",
    "hum": "mains hum",
    "narrow": "narrow-band line",
    "music": "background music",
    "opus8k": "low-rate Opus",
    "quiet": "very quiet",
    "loss": "packet loss",
    "burst": "burst loss",
    "amr": "AMR phone codec",
    "g722": "G.722 codec",
}


def chart_modern(n: dict, w: int = 446, h: int = CHART_H) -> str:
    """Fig 6: share of cloned calls flagged, by cloning tool and line type (heat matrix)."""
    s = Svg(w, h)
    cols = [("Clean", "line"), ("Noisy", "room"), ("Phone", "line")]
    rows = [
        ("kNN-VC", "open-source cloning", n["modern"]["knnvc"]),
        ("SpeechT5", "laptop-class cloning", n["modern"]["speecht5x"]),
        ("MMS-TTS", "synthetic speech", n["modern"]["mms"]),
    ]
    lab_w = 176
    cw = (w - lab_w) / 3
    hy, rh = 50, 50
    for j, (a, b) in enumerate(cols):
        s.text(lab_w + j * cw + cw / 2, 20, a, 22, MUTED, "b", "middle")
        s.text(lab_w + j * cw + cw / 2, 42, b, 22, MUTED, "b", "middle")
    for i, (name, sub, vals) in enumerate(rows):
        y = hy + i * (rh + 6)
        s.text(0, y + 22, name, 25, INK, "b")
        s.text(0, y + 43, sub, 19, MUTED)
        for j, v in enumerate(vals):
            if v >= 90:
                fill, tc = GREEN, "#FFFFFF"
            elif v >= 70:
                fill, tc = "#BFE3CE", "#0E5A36"
            elif v >= 50:
                fill, tc = "#FFE0A3", "#7A4A00"
            else:
                fill, tc = "#F6C5C0", "#8A1C15"
            s.rect(lab_w + j * cw + 4, y, cw - 8, rh, fill, rx=10)
            s.text(lab_w + j * cw + cw / 2, y + 35, f"{r0(v)}%", 30, tc, "d", "middle")
    return s.svg()


# --------------------------------------------------------------------------- diagram --
def diagram() -> str:
    """Fig 3: how VoiceGuard decides: the whole pipeline in one picture."""
    w, h = 2056, 340
    s = Svg(w, h, "diagram")

    def card(x, y, cw, ch, ec, title, body, ico, tsize=30, bsize=22, ic=40):
        s.rect(x, y, cw, ch, "#FFFFFF", rx=18, stroke=ec, sw=3)
        s.rect(x, y, 9, ch, ec, rx=4)
        s.add(inline_icon(ico, ic, ec, x + 26, y + 14))
        s.text(x + 26 + ic + 14, y + 14 + ic / 2 + 10, title, tsize, INK, "d")
        s.lines(x + 26, y + 14 + ic + 26, wrap(body, bsize, cw - 44), bsize, MUTED, lh=1.22)

    def arrow(x1, y1, x2, y2, col="#7B8A9A", sw=3.5):
        s.line(x1, y1, x2 - 6, y2, col, sw)
        ang = math.atan2(y2 - y1, x2 - x1)
        for da in (2.6, -2.6):
            s.line(x2, y2, x2 + 14 * math.cos(ang + da), y2 + 14 * math.sin(ang + da), col, sw)

    # 1 live call
    s.rect(0, 80, 170, 180, "#F4F6F9", rx=18, stroke=GREY, sw=3)
    s.add(inline_icon("phone", 56, INK, 57, 96))
    s.text(85, 184, "Live call", 30, INK, "d", "middle")
    s.lines(85, 214, ["phone, VoIP", "or microphone"], 21, MUTED, anchor="middle", lh=1.2)
    # 2 listen and clean
    cx0, cw = 206, 320
    card(
        cx0,
        0,
        cw,
        164,
        GREEN,
        "Input gate",
        "Repairs bad audio; says WAITING when there is nothing to judge.",
        "gate",
    )
    card(
        cx0,
        176,
        cw,
        164,
        GREEN,
        "Fair front-end",
        "Identical for real and fake, so the recording can't give it away.",
        "sliders",
    )
    arrow(170, 170, cx0 - 4, 170)
    # 3 four kinds of evidence
    ex0, ew, eh, eg = 564, 366, 164, 14
    ev = [
        (
            ex0,
            0,
            BLUE,
            "Voice detector",
            "AASIST neural network, tuned for noise, rooms and phone lines.",
            "wave",
        ),
        (
            ex0 + ew + eg,
            0,
            BLUE,
            "Voice match",
            "WavLM voiceprint compared with the customer's voice on file.",
            "match",
        ),
        (
            ex0,
            eh + 12,
            AMBER,
            "Recording style",
            "Noise, echo, clipping: sets how far to trust the detector.",
            "mic",
        ),
        (
            ex0 + ew + eg,
            eh + 12,
            GREY,
            "Call details",
            "Known number? Amount? Earlier warnings? Not from the audio.",
            "clip",
        ),
    ]
    for ex, ey, ec, t, b, ic in ev:
        card(ex, ey, ew, eh, ec, t, b, ic, tsize=30)
    for ey in (82, 258):
        arrow(cx0 + cw + 2, 170, ex0 - 4, ey)
    # 4 fusion
    fx, fw = ex0 + 2 * ew + eg + 40, 256
    s.rect(fx, 0, fw, h, "#F1F5FF", rx=18, stroke=BLUE, sw=3)
    s.rect(fx, 0, 9, h, BLUE, rx=4)
    s.add(inline_icon("merge", 52, BLUE, fx + fw / 2 - 26, 18))
    s.lines(fx + fw / 2 + 4, 104, ["Trust-weighted", "fusion"], 29, INK, "d", "middle", 1.1)
    s.lines(
        fx + 26,
        178,
        wrap(
            "Clone, voice and call alarms, kept separate. Recording style sets how far to believe the detector.",
            22,
            fw - 44,
        ),
        22,
        MUTED,
        lh=1.22,
    )
    for ey in (82, 258):
        arrow(ex0 + 2 * ew + eg + 4, ey, fx - 4, 170 + (ey - 170) * 0.3)
    # 5 policy and answer
    ax = fx + fw + 40
    aw = w - ax
    arrow(fx + fw + 2, 112, ax - 4, 60)
    s.rect(ax, 0, aw, 118, "#FFFFFF", rx=18, stroke=BLUE, sw=3)
    s.rect(ax, 0, 9, 118, BLUE, rx=4)
    s.add(inline_icon("dial", 40, BLUE, ax + 26, 14))
    s.text(ax + 26 + 40 + 14, 14 + 20 + 10, "Policy", 30, INK, "d")
    s.lines(
        ax + 26,
        80,
        wrap("Routine, payment or sensitive call: how much evidence is enough.", 22, aw - 44),
        22,
        MUTED,
        lh=1.2,
    )
    answers = [
        ("ALLOW", "carry on", GREEN),
        ("VERIFY", "double-check", AMBER),
        ("ESCALATE", "stop and verify", RED),
    ]
    ay = 140
    arrow(ax + aw / 2, 120, ax + aw / 2, ay - 3)
    for i, (a, b, col) in enumerate(answers):
        y = ay + i * 66
        s.rect(ax, y, aw, 58, col, rx=16)
        s.text(ax + 22, y + 39, a, 27, "#FFFFFF", "d")
        s.text(ax + aw - 20, y + 38, b, 22, "#FFFFFF", "b", "end")
    return s.svg()


# ---------------------------------------------------------------------------- header --
def wave_art(w: int = 2245, h: int = 300) -> str:
    """The header texture: a natural waveform that turns into the stepped one of synthetic speech."""

    def y_at(x: float, ph: float) -> float:
        t = x / w
        env = 0.35 + 0.65 * math.sin(math.pi * min(max(t * 1.05, 0), 1)) ** 0.8
        return h / 2 + env * (
            46 * math.sin(x / 26 + ph) * (0.55 + 0.45 * math.sin(x / 160 + ph))
            + 22 * math.sin(x / 10 + ph * 2.1)
            + 10 * math.sin(x / 5.7)
        )

    switch = int(w * 0.62)
    out = []
    for ph, op, sw in ((0.0, 0.9, 3.5), (1.4, 0.5, 2.5), (2.9, 0.3, 2)):
        pts = [f"{x},{y_at(x, ph):.1f}" for x in range(0, switch + 1, 4)]
        d_smooth = "M" + " L".join(pts)
        d_step = f"M{switch},{round(y_at(switch, ph) / 18) * 18:.0f}"
        for x in range(switch + 14, w + 1, 14):
            d_step += f" H{x} V{round(y_at(x, ph) / 18) * 18:.0f}"
        out.append(
            f'<path d="{d_smooth}" fill="none" stroke="url(#wg)" stroke-width="{sw}" opacity="{op}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        out.append(
            f'<path d="{d_step}" fill="none" stroke="#FF8D84" stroke-width="{sw}" opacity="{op}" stroke-linejoin="round"/>'
        )
    return (
        f'<svg class="wave" width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="wg" x1="0" x2="1"><stop offset="0" stop-color="#4C7BFF" stop-opacity=".15"/><stop offset=".4" stop-color="#7FA2FF"/><stop offset="1" stop-color="#B8CCFF"/></linearGradient></defs>'
        f'{"".join(out)}<line x1="{switch}" y1="6" x2="{switch}" y2="{h - 6}" stroke="#FFFFFF" stroke-opacity=".3" stroke-width="2" stroke-dasharray="3 9"/></svg>'
    )


def logo_mark(size: int = 100) -> str:
    bars = [(34, 20), (50, 44), (66, 70), (82, 52), (98, 28)]
    b = "".join(
        f'<rect x="{x - 6}" y="{85 - hh / 2:.0f}" width="12" height="{hh}" rx="6" fill="#FFFFFF"/>'
        for x, hh in bars
    )
    return (
        f'<svg width="{size}" height="{size * 1.15:.0f}" viewBox="0 0 132 152" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="sg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6E93FF"/><stop offset="1" stop-color="#2457D6"/></linearGradient></defs>'
        '<path d="M66 6 L122 26 V74 C122 110 98 134 66 146 C34 134 10 110 10 74 V26 Z" fill="url(#sg)"/>'
        '<path d="M66 6 L122 26 V74 C122 110 98 134 66 146 C34 134 10 110 10 74 V26 Z" fill="none" stroke="#FFFFFF" stroke-opacity=".35" stroke-width="3"/>'
        f"{b}</svg>"
    )


# ----------------------------------------------------------------------------- shots --
def shot_uri(name: str, crop_h: int = 836) -> str:
    """The console answer panel, cut to its top part (verdict, scorecard, risk meter), as a data URI."""
    im = Image.open(SHOTS / f"{name}.png").convert("RGB")
    im = im.crop((3, 3, im.width - 3, min(crop_h, im.height)))
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------------------ page --
CSS = r"""
@page { size: 594mm 841mm; margin: 0 }
*{box-sizing:border-box;margin:0;padding:0}
:root{--ink:#17202B;--muted:#566678;--line:#D3DBE5;--bg:#E9EEF5;--blue:#2457D6;--blue2:#1B3FA6;--green:#17794A;--amber:#B86F00;--red:#B3261E;--navy:#0E1622}
html,body{width:2245px;height:3178px;background:var(--bg)}
body{font-family:"Segoe UI",system-ui,sans-serif;color:var(--ink);font-size:29px;line-height:1.36;-webkit-print-color-adjust:exact;print-color-adjust:exact}
svg text{font-family:"Segoe UI",system-ui,sans-serif}
svg text.b{font-weight:700}
svg text.d{font-family:"Bahnschrift","Segoe UI",sans-serif;font-weight:700}
.poster{position:relative;width:2245px;height:3178px;overflow:hidden;display:flex;flex-direction:column}

/* header: title on the left, who made it on the right */
.hero{position:relative;height:340px;flex:none;color:#fff;overflow:hidden;
  background:radial-gradient(1100px 520px at 88% 6%,rgba(60,110,255,.55),transparent 62%),linear-gradient(115deg,#0B121C 0%,#122036 55%,#1B3A7A 130%)}
.hero .wave{position:absolute;left:0;top:20px;opacity:.2}
.hero-main{position:absolute;left:62px;top:18px}
.pill{display:inline-flex;align-items:center;gap:16px;font-family:"Bahnschrift";font-weight:600;font-size:24px;letter-spacing:.15em;color:#B9CBFF;background:rgba(120,160,255,.14);border:1.5px solid rgba(160,190,255,.35);border-radius:40px;padding:3px 22px}
.pill i{width:8px;height:8px;border-radius:50%;background:#7FA2FF;display:inline-block}
.title-row{display:flex;align-items:center;gap:26px;margin-top:4px}
h1{font-family:"Bahnschrift";font-weight:700;font-size:164px;line-height:1;letter-spacing:-.025em;color:#fff}
h1 span{color:#8FB0FF}
.ptitle{font-family:"Bahnschrift";font-weight:600;font-size:40px;line-height:1.1;margin-top:6px;color:#FFFFFF}
.sub{font-size:27px;line-height:1.3;color:#B8C6DA;margin-top:6px;white-space:nowrap}
.idcard{position:absolute;right:62px;top:16px;width:1010px;display:grid;grid-template-columns:480px 1fr;gap:24px;background:rgba(255,255,255,.09);border:1.5px solid rgba(160,190,255,.32);border-radius:22px;padding:18px 28px}
.idrow{margin-bottom:12px}
.idrow:last-child{margin-bottom:0}
.idrow>span,.idhead>span{display:block;font-family:"Bahnschrift";font-weight:600;font-size:19px;letter-spacing:.15em;text-transform:uppercase;color:#9DB6FF}
.idrow b{display:block;font-size:28px;line-height:1.2;font-weight:700;color:#fff}
.idrow em{display:block;font-style:normal;font-size:22px;line-height:1.25;color:#C9D8FF;margin-top:1px}
.idhead{display:flex;justify-content:space-between;border-bottom:1.5px solid rgba(255,255,255,.22);padding-bottom:6px;margin-bottom:6px}
.mem{display:flex;justify-content:space-between;gap:20px;font-size:26px;line-height:1.45}
.mem b{font-weight:600;color:#fff}
.mem i{font-style:normal;color:#C9D8FF}
.ph{background:#FFE08A;color:#5A4200;border-radius:8px;padding:0 10px;font-weight:600}

.main{display:flex;flex-direction:column;gap:14px;padding:14px 0;flex:1}
.chart svg,.chartbox svg{display:block}
.band{display:grid;gap:30px;padding:0 56px}
.card{position:relative;background:#fff;border:1.5px solid #D9E1EB;border-radius:26px;box-shadow:0 8px 28px rgba(15,24,36,.07);padding:24px 34px;overflow:hidden}
.lab{display:flex;align-items:center;gap:14px;font-family:"Bahnschrift";font-weight:700;font-size:26px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);white-space:nowrap}
.lab b{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:50%;background:var(--blue);color:#fff;font-size:26px;letter-spacing:0;flex:none}
.lab.light{color:#9DB6FF}.lab.light b{background:#fff;color:var(--navy)}
.lab.amber{color:var(--amber)}.lab.amber b{background:var(--amber)}
.lab.red{color:var(--red)}.lab.red b{background:var(--red)}
h2{font-family:"Bahnschrift";font-weight:700;font-size:52px;line-height:1.07;letter-spacing:-.01em;margin:12px 0 10px}
p{margin:0}
p,li{text-wrap:pretty}
h2,.claim{text-wrap:balance}
.giant{font-family:"Bahnschrift";font-weight:700;font-size:100px;line-height:.95;letter-spacing:-.03em}
.red{color:var(--red)}.green{color:var(--green)}.amber{color:var(--amber)}.blue{color:var(--blue)}
.claim{font-size:27px;line-height:1.28;font-weight:600}
.cap{font-size:23px;line-height:1.3;color:var(--muted)}
.cap b{color:var(--ink)}
.cap-tag{font-family:"Bahnschrift";font-weight:600;font-size:21px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}

/* problem statement | why one detector | objectives */
.b2{height:462px;grid-template-columns:540px 960px 1fr}
.c-problem h2{font-size:44px}
.c-problem p{font-size:27px;line-height:1.34}
.c-why{display:grid;grid-template-columns:340px 1fr;grid-template-rows:auto 1fr;column-gap:30px}
.c-why .lab{grid-column:1/-1}
.c-why .chartbox{display:flex;flex-direction:column;justify-content:flex-end;gap:4px}
.obj{background:linear-gradient(150deg,#2D63E4 0%,#1B3FA6 100%);border:0;color:#fff}
.obj .lab{color:#C9D8FF}.obj .lab b{background:#fff;color:#1B3FA6}
.obj ol{list-style:none;counter-reset:o;margin-top:14px}
.obj li{counter-increment:o;position:relative;padding-left:46px;margin-bottom:10px;font-size:26px;line-height:1.28;color:#EEF3FF}
.obj li:before{content:counter(o);position:absolute;left:0;top:2px;width:34px;height:34px;border-radius:50%;background:rgba(255,255,255,.2);border:2px solid rgba(255,255,255,.55);font-family:"Bahnschrift";font-weight:700;font-size:21px;line-height:30px;text-align:center;color:#fff}

/* methodology / workflow diagram */
.b3{height:468px;grid-template-columns:1fr}
.how-head{display:flex;align-items:center;gap:26px;margin-bottom:14px}
.how-head h2{margin:0;font-size:48px;flex:none}
.diagram{display:block}

/* technologies and tools */
.b3b{height:178px;grid-template-columns:1fr}
.tools{display:grid;grid-template-columns:250px repeat(5,1fr);gap:22px;align-items:start;padding-top:20px}
.tools .lab{white-space:normal;line-height:1.15;align-items:flex-start;flex-direction:column;gap:10px;font-size:25px}
.tg{font-family:"Bahnschrift";font-weight:700;font-size:20px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);margin-bottom:6px;border-bottom:2px solid #DCE6F7;padding-bottom:4px}
.tools ul{list-style:none}
.tools li{position:relative;padding-left:20px;font-size:22px;line-height:1.28;color:var(--ink);white-space:nowrap}
.tools li:before{content:"";position:absolute;left:2px;top:10px;width:8px;height:8px;border-radius:50%;background:var(--blue)}

/* results: the console in action (dark band, full width) */
.b4{height:492px;background:linear-gradient(180deg,#0E1622,#14233A);grid-template-columns:1fr 440px 440px 440px;padding-top:24px;padding-bottom:20px;align-items:start}
.b4 h2{color:#fff;font-size:46px;max-width:700px}
.b4 .ops-text{font-size:26px;line-height:1.34;color:#B8C6DA;max-width:700px}
.answers{margin-top:16px;display:flex;flex-direction:column;gap:8px;font-size:24px;line-height:1.25;color:#C7D3E5;white-space:nowrap}
.answers b{color:#fff}
.answers i{display:inline-block;width:20px;height:20px;border-radius:50%;margin-right:12px;vertical-align:-2px}
.shot{display:flex;flex-direction:column;gap:8px;min-width:0}
.chip{align-self:flex-start;font-family:"Bahnschrift";font-weight:700;font-size:24px;letter-spacing:.1em;text-transform:uppercase;color:#fff;border-radius:30px;padding:2px 20px}
.chip.g{background:var(--green)}.chip.r{background:var(--red)}.chip.a{background:#C27800}
.frame{border-radius:20px;overflow:hidden;background:#fff;line-height:0;box-shadow:0 12px 36px rgba(0,0,0,.4)}
.frame img{width:100%;display:block}
.shot .cap{color:#C7D3E5;font-size:23px;line-height:1.28}
.shot .cap b{color:#fff}

/* results: the numbers */
.b5{height:566px;grid-template-columns:repeat(4,1fr)}
.ev{display:flex;flex-direction:column;padding:18px 26px 16px}
.ev .tag{font-family:"Bahnschrift";font-weight:700;font-size:24px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue)}
.ev .giant{font-size:76px;margin-top:2px}
.ev .claim{font-size:26px;line-height:1.25;margin:6px 0 8px;height:98px}
.ev .chart{height:218px}
.ev .cap{font-size:22px;margin-top:6px;line-height:1.26}

/* conclusion | honest limits | what is next */
.b6{height:366px;grid-template-columns:880px 760px 1fr}
.c-concl p{font-size:27px;line-height:1.32;margin-top:12px}
.limits{background:#FFF6E2;border:2px solid #E7C778}
.limits ul,.next ul{list-style:none;margin-top:12px}
.limits li,.next li{position:relative;padding-left:42px;margin-bottom:9px;font-size:25px;line-height:1.28}
.limits li:before{content:"";position:absolute;left:4px;top:10px;width:19px;height:19px;border-radius:5px;background:var(--amber)}
.next li:before{content:"";position:absolute;left:4px;top:10px;width:19px;height:19px;border-radius:50%;border:5px solid var(--blue)}

.footer{height:64px;flex:none;background:var(--navy);color:#B8C6DA;padding:8px 62px;font-size:19px;line-height:1.3;display:flex;align-items:center}
.footer b{color:#fff}

/* signature boxes, the last thing on the poster */
.sig{height:144px;flex:none;display:grid;grid-template-columns:1fr 410px 1fr;gap:28px;padding:14px 56px 14px;align-items:stretch}
.sigbox{position:relative;background:#fff;border:2.5px solid var(--ink);border-radius:16px}
.sigbox span{position:absolute;left:24px;right:24px;bottom:10px;border-top:2px solid #9AA8B6;padding-top:6px;font-family:"Bahnschrift";font-weight:700;font-size:25px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink)}
.qrbox{display:flex;align-items:center;gap:16px;color:var(--ink)}
.qrbox svg{width:96px;height:96px;flex:none;background:#fff;border-radius:10px}
.qrbox div{font-size:20px;line-height:1.25}
.qrbox b{font-family:"Bahnschrift";font-size:22px;display:block;line-height:1.15;margin-bottom:3px}
.qrbox small{font-size:18px;color:var(--muted);display:block;line-height:1.2}
"""


def build_html(team: dict, check: bool = False) -> str:
    n = load_numbers()
    qr = qrcode_svg.to_svg(REPO_URL, quiet=2, dark="#0E1622", light="#FFFFFF")

    def ph(text: str | None, fallback: str) -> str:
        return esc(text) if text else f'<span class="ph">{esc(fallback)}</span>'

    members = team.get("members") or []
    if isinstance(members, str):
        members = parse_members(members)
    if not members:
        members = [("", "")] * 3
    mem_rows = "".join(
        f'<div class="mem"><b>{ph(nm, "[Member name]")}</b><i>{ph(rl, "[Roll no.]")}</i></div>'
        for nm, rl in members
    )
    team_line = (
        f'<div class="idrow"><span>Team</span><b>{esc(team["team"])}</b></div>'
        if team.get("team")
        else ""
    )
    term_line = f"<em>{esc(team['term'])}</em>" if team.get("term") else ""
    idcard = (
        '<div class="idcard"><div>'
        f'<div class="idrow"><span>Subject</span><b>{ph(team.get("subject"), "[Subject name]")}</b></div>'
        f'<div class="idrow"><span>Faculty</span><b>{ph(team.get("faculty"), "[Faculty name]")}</b></div>'
        f'<div class="idrow"><span>College</span><b>{ph(team.get("college"), "[College name]")}</b>{term_line}</div>'
        f"{team_line}</div>"
        f'<div><div class="idhead"><span>{"Group member" if len(members) == 1 else "Group members"}</span><span>Roll no.</span></div>{mem_rows}</div></div>'
    )
    mod = n["modern"]
    weak_txt = ", ".join(f"{_SITUATION_NAMES.get(c, c)} {r0(v)}%" for c, v in n["stress_weakest"])
    gap_ours, gap_pre = n["gap_ours"], n["gap_pre"]
    fus = n["fusion"]

    body = f"""
<div class="poster">
<header class="hero">
  {wave_art()}
  <div class="hero-main">
    <div class="pill">SMART INDIA HACKATHON 2026 <i></i> PROBLEM STATEMENT SIH26104</div>
    <div class="title-row">{logo_mark(88)}<h1>Voice<span>Guard</span></h1></div>
    <p class="ptitle">Real-time detection of AI voice-cloning fraud on phone calls</p>
    <p class="sub">Is the person on this call really your customer? A privacy-preserving decision layer, with numbers you can check.</p>
  </div>
  {idcard}
</header>

<main class="main">
<section class="band b2">
  <article class="card c-problem">
    <div class="lab"><b>1</b>Problem statement</div>
    <h2>A voice no longer proves who is speaking.</h2>
    <p>A few seconds of speech are enough for free AI tools to copy someone's voice. Fraudsters phone banks as the customer and push through payments and account changes. The agent has seconds to decide.</p>
  </article>
  <article class="card c-why">
    <div class="lab red"><b>!</b>Why one detector is not enough</div>
    <div>
      <div class="giant red" style="margin-top:10px">{r0(gap_pre[2])}%</div>
      <p class="claim" style="margin-top:8px">error on real-world audio from the best published detector we tested. On its own benchmark it reports about 1%.</p>
      <p class="cap" style="margin-top:8px">A current text-to-speech voice was flagged only <b>8%</b> of the time.</p>
    </div>
    <div class="chartbox"><div class="cap-tag">Error rate (EER) · lower is better</div>{chart_gap(n)}</div>
  </article>
  <article class="card obj">
    <div class="lab"><b>2</b>Objectives</div>
    <ol>
      <li>Detect cloned and computer-generated voices on live calls, in real time.</li>
      <li>Combine four independent kinds of evidence instead of trusting one detector.</li>
      <li>Say plainly how sure it is, and keep a person in charge.</li>
      <li>Measure honestly and keep callers' audio private.</li>
    </ol>
  </article>
</section>

<section class="band b3">
  <article class="card">
    <div class="how-head">
      <div class="lab"><b>3</b>Methodology / workflow diagram</div>
      <h2>Four kinds of evidence, combined every half second</h2>
    </div>
    {diagram()}
  </article>
</section>

<section class="band b3b">
  <article class="card tools">
    <div class="lab"><b>4</b>Technologies and tools used</div>
    <div><div class="tg">Models</div><ul><li>AASIST anti-spoofing network</li><li>WavLM voice embeddings</li><li>Silero VAD, RawBoost</li></ul></div>
    <div><div class="tg">Code</div><ul><li>Python 3.11, PyTorch 2.3</li><li>scikit-learn, SciPy</li><li>NumPy, pandas, librosa</li></ul></div>
    <div><div class="tg">Serving</div><ul><li>FastAPI, WebSocket</li><li>gRPC, Protocol Buffers</li><li>Browser console (Web Audio)</li></ul></div>
    <div><div class="tg">Data</div><ul><li>ASVspoof 2019, In-the-Wild</li><li>LibriSpeech</li><li>Clones: kNN-VC, SpeechT5, MMS</li></ul></div>
    <div><div class="tg">Quality</div><ul><li>pytest: 268 tests</li><li>ruff, Git and GitHub</li><li>CUDA GPU, local and Kaggle</li></ul></div>
  </article>
</section>

<section class="band b4">
  <div class="ops-head">
    <div class="lab light"><b>5</b>Results: the console in action</div>
    <h2>One plain answer, the reasons, and an honest scorecard.</h2>
    <p class="ops-text">Screens from the live console, playing three built-in demo calls. The scorecard tells you whether VoiceGuard was right, including when it missed.</p>
    <div class="answers">
      <div><i style="background:var(--green)"></i><b>Looks like the real customer:</b> carry on</div>
      <div><i style="background:#C27800"></i><b>Double-check:</b> ask a security question</div>
      <div><i style="background:var(--red)"></i><b>Stop:</b> pause payments, phone back</div>
    </div>
  </div>
  <figure class="shot"><span class="chip g">A real customer</span><div class="frame"><img src="{shot_uri("console_real")}"></div><p class="cap"><b>Green answer, low risk.</b> Scorecard: Correct.</p></figure>
  <figure class="shot"><span class="chip r">A cloned voice</span><div class="frame"><img src="{shot_uri("console_fake")}"></div><p class="cap"><b>Red Stop,</b> with what to do: pause the payment, call the customer back.</p></figure>
  <figure class="shot"><span class="chip a">The honest hard case</span><div class="frame"><img src="{shot_uri("console_hard")}"></div><p class="cap"><b>A free open-source clone:</b> warned early, then eased off. We show misses too.</p></figure>
</section>

<section class="band b5">
  <article class="card ev" style="border-top:7px solid var(--amber)">
    <div class="tag">5a · A fair test</div>
    <div class="giant amber">13–15%</div>
    <p class="claim">error from silence alone in our first pipeline: a benchmark shortcut, now guarded by a permanent test.</p>
    <div class="chart">{chart_silence(n)}</div>
    <p class="cap">Silent parts only: a model that listens to the voice scores about 50%. Real-world audio is never used for training or thresholds.</p>
  </article>
  <article class="card ev" style="border-top:7px solid var(--blue)">
    <div class="tag">5b · Combine the evidence</div>
    <div class="giant blue">{r0(fus[2])}%</div>
    <p class="claim">balanced accuracy on speakers the system never saw, against {r0(fus[0])}% for the detector alone.</p>
    <div class="chart">{chart_fusion(n)}</div>
    <p class="cap">Genuine callers accepted and attacks flagged, averaged: 19 situations, 4,157 reads.</p>
  </article>
  <article class="card ev" style="border-top:7px solid var(--green)">
    <div class="tag">5c · Awkward situations</div>
    <div class="giant green">{n["stress"][1]:.1f}%</div>
    <p class="claim">in 14 situations it had never met: rooms, echo, clipping, hum, music, phone codecs, packet loss.</p>
    <div class="chart">{chart_stress(n)}</div>
    <p class="cap"><b>Weakest:</b> {esc(weak_txt)}.</p>
  </article>
  <article class="card ev" style="border-top:7px solid var(--blue)">
    <div class="tag">5d · Newer cloning tools</div>
    <div class="giant blue">{r0(mod["knnvc"][0])}%</div>
    <p class="claim">of clean-line clones from a free open-source tool were flagged; {r0(mod["knnvc"][2])}–{r0(mod["knnvc"][1])}% on phone or noisy lines.</p>
    <div class="chart">{chart_modern(n)}</div>
    <p class="cap">Share of clones flagged. The detector alone sees 2–3% of kNN-VC and SpeechT5; the voice match does the rest.</p>
  </article>
</section>

<section class="band b6">
  <article class="card c-concl">
    <div class="lab"><b>6</b>Conclusion</div>
    <p>Stopping voice-clone fraud takes several partly independent signals, measured fairly and explained in plain words, not one clever detector. VoiceGuard reaches about {r0(fus[2])}% balanced accuracy on unseen speakers and {n["stress"][1]:.1f}% in situations it had never met, flags {r0(mod["knnvc"][0])}–100% of modern clones, and stores no audio. It is decision support, not a gatekeeper. When a real voice note was called fake, we fixed our own model and reported the cost.</p>
  </article>
  <article class="card limits">
    <div class="lab amber"><b>!</b>Honest limits</div>
    <ul>
      <li>On real-world audio the detector alone is weak ({r0(gap_ours[2])}% error): it supports, it does not decide.</li>
      <li>A clone that sounds exactly like the customer can still pass. Tested on simulated lines and codecs, not recorded real calls.</li>
      <li>Indian-language results are a small synthetic pilot; alerts are simulated; authentication and TLS are not built.</li>
    </ul>
  </article>
  <article class="card next">
    <div class="lab"><b>{ARROW}</b>What is next</div>
    <ul>
      <li>Retrain on modern clones and real codec audio.</li>
      <li>Consented real calls and Indian-language speech.</li>
      <li>Real alerts, security hardening, edge inference.</li>
    </ul>
  </article>
</section>
</main>

<footer class="footer">
  <div><b>References.</b> AASIST (Jung et al., ICASSP 2022) · ASVspoof 2019 (Wang et al.) · In-the-Wild (Müller et al., Interspeech 2022) · WavLM (Chen et al., 2022) · RawBoost (Tak et al., ICASSP 2022) · kNN-VC (Baas et al., 2023) · MMS-TTS (Pratap et al., 2023) · Silero VAD · LibriSpeech. Datasets and models belong to their authors and are used under their own licences. Every number here is reproducible from the repository.</div>
</footer>

<section class="sig">
  <div class="sigbox"><span>Faculty Signature</span></div>
  <div class="qrbox">{qr}<div><b>Scan for the code, results and demo</b><small>github.com/ghanshyam1011/voice-clone-detector</small></div></div>
  <div class="sigbox"><span>External Examiner's Signature</span></div>
</section>
</div>
"""
    script = ""
    if check:
        script = """<script>window.addEventListener('load',()=>{const bad=[];document.querySelectorAll('.card,.chip,.ph,.hero-main,.idcard,.shot,.frame,.chart,.ev .claim,.sigbox,.qrbox,.footer,.sig').forEach(e=>{if(e.scrollHeight>e.clientHeight+2||e.scrollWidth>e.clientWidth+2)bad.push([e.className.toString().slice(0,30),e.scrollHeight,e.clientHeight,e.scrollWidth,e.clientWidth])});const p=document.querySelector('.poster');const sig=document.querySelector('.sig').getBoundingClientRect();document.body.setAttribute('data-check',JSON.stringify({bad,w:p.scrollWidth,h:p.scrollHeight,sigBottom:Math.round(sig.bottom)}))})</script>"""
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>VoiceGuard poster (SIH26104)</title><style>{CSS}</style></head><body>{body}{script}</body></html>'


# --------------------------------------------------------------------------- render --
def _browser() -> str:
    for p in BROWSERS:
        if Path(p).exists():
            return p
    raise SystemExit("Chrome or Edge was not found; install one or edit BROWSERS.")


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="vg_poster_") as prof:
        subprocess.run(
            [
                _browser(),
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={prof}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                "--virtual-time-budget=8000",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=240,
        )


def overflow_report(html_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="vg_poster_") as prof:
        out = subprocess.run(
            [
                _browser(),
                "--headless=new",
                "--disable-gpu",
                f"--user-data-dir={prof}",
                "--virtual-time-budget=8000",
                "--dump-dom",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=240,
        ).stdout.decode("utf-8", "replace")
    m = re.search(r"data-check='(.*?)'", out) or re.search(r'data-check="(.*?)"', out)
    return json.loads(html.unescape(m.group(1))) if m else {}


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    doc[0].get_pixmap(dpi=dpi).save(png_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", help="subject name")
    ap.add_argument("--faculty", "--guide", dest="faculty", help="faculty name")
    ap.add_argument("--college")
    ap.add_argument("--term", help='class and year, e.g. "BE CMPN, Sem 7 · AY 2025-26"')
    ap.add_argument(
        "--members", help='"Name, Roll; Name, Roll; ..." (roll numbers after the comma)'
    )
    ap.add_argument("--team", help="optional team name")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dpi", type=int, default=150, help="resolution of the PNG preview")
    ap.add_argument("--check", action="store_true", help="report boxes whose content overflows")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # the details live in team.json next to the poster, so a plain re-run keeps them; flags update it
    team_file = out / "team.json"
    saved = json.loads(team_file.read_text(encoding="utf-8")) if team_file.exists() else {}
    for key in ("subject", "faculty", "college", "term", "members", "team"):
        if getattr(a, key) is not None:
            saved[key] = getattr(a, key)
    if saved:
        team_file.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    team = {**saved, "members": parse_members(saved.get("members"))}
    html_path = out / "voiceguard_poster_A1.html"
    html_path.write_text(build_html(team, check=a.check), encoding="utf-8")
    print("wrote", html_path)
    if a.no_render:
        return
    if a.check:
        rep = overflow_report(html_path)
        print(
            "page",
            rep.get("w"),
            "x",
            rep.get("h"),
            "(want",
            W,
            "x",
            H,
            "); signature band ends at",
            rep.get("sigBottom"),
        )
        for b in rep.get("bad", []):
            print("  OVERFLOW", b)
        if not rep.get("bad"):
            print("  no box overflows")
    pdf = out / "voiceguard_poster_A1.pdf"
    render_pdf(html_path, pdf)
    png = out / "voiceguard_poster_A1.png"
    pdf_to_png(pdf, png, a.dpi)
    print("wrote", pdf.name, f"({pdf.stat().st_size / 1e6:.1f} MB) and", png.name)


if __name__ == "__main__":
    main()
