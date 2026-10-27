"""Poster / viva figures, drawn from the measured tables in ``results/tables/``.

    python scripts/make_poster_figures.py            # writes docs/poster_figures/*.png (300 dpi)

Every number comes from a CSV in ``results/tables`` or from a figure quoted, with its source, in
``docs/`` (the comment above each chart names it), so a figure can be regenerated and checked, not
redrawn by hand. Colours match the operator console: green = fine, amber = double-check, red = stop.
"""

# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "poster_figures"

INK, MUTED, LINE = "#17202b", "#5b6b7b", "#d5dce4"
BLUE, GREEN, AMBER, RED, GREY = "#2457d6", "#17794a", "#c07a00", "#b3261e", "#8a97a5"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _eer(csv: str, split: str, model: str | None = None) -> float:
    d = pd.read_csv(TAB / csv)
    d = d[d.split == split]
    if model is not None:
        d = d[d.model == model]
    return float(d.value_pct.iloc[0])


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _label(ax, bars, fmt="{:.1f} %", dy=0.8, fs=13):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=fs,
            color=INK,
            fontweight="bold",
        )


# 1 -- the honest gap ---------------------------------------------------------------------
def fig_honest_gap() -> None:
    """EER of three detectors on three kinds of audio. Sources: README table;
    baseline_*.csv, cm_aasist-pretrained_*.csv, cm_aasist-AASIST_robust2_*.csv."""
    from matplotlib.lines import Line2D

    groups = [
        "Known attacks\n(same kind as training)",
        "Unseen attacks\n(new synthesis methods)",
        "Real-world audio\n(In-the-Wild)",
    ]
    base = [7.0, 33.4, 58.2]  # README: handcrafted + GBM ensemble (Gen-1)
    pre = [
        _eer("cm_aasist-pretrained_in_domain.csv", "dev_known_attacks"),
        _eer("cm_aasist-pretrained_in_domain.csv", "eval_unknown_attacks"),
        _eer("cm_aasist-pretrained_cross_dataset.csv", "cross_dataset"),
    ]
    ours = [
        _eer("cm_aasist-AASIST_robust2_in_domain.csv", "dev_known_attacks"),
        _eer("cm_aasist-AASIST_robust2_in_domain.csv", "eval_unknown_attacks"),
        _eer("cm_aasist-AASIST_robust2_cross_dataset.csv", "cross_dataset"),
    ]
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6.6))
    b1 = ax.bar(x - w, base, w, color=GREY, label="Handcrafted features + GBM (our first baseline)")
    b2 = ax.bar(
        x, pre, w, color="#7a9be8", label="AASIST, published weights, through our honest front-end"
    )
    b3 = ax.bar(x + w, ours, w, color=BLUE, label="VoiceGuard detector (AASIST fine-tuned here)")
    for bars in (b1, b2, b3):
        _label(ax, bars, fs=11.5)
    ax.axhline(1.0, color=GREEN, ls="--", lw=2)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=GREEN, ls="--", lw=2))
    labels.append("About 1 %: what the AASIST authors report on the benchmark's own test")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Equal error rate (lower is better)")
    ax.set_ylim(0, 80)
    ax.set_title(
        "A detector that looks near-perfect in the lab is far from it outside",
        loc="left",
        fontweight="bold",
        fontsize=17,
    )
    ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=12)
    _save(fig, "1_honest_gap.png")


# 2 -- the silence shortcut ----------------------------------------------------------------
def fig_silence() -> None:
    """Silence-only EER (50 % = chance; below 40 % = a shortcut). Sources: README sec. 2.1,
    docs/evaluation_protocol.md; baseline_silence_ablation.csv, cm_aasist-*_silence_ablation.csv."""
    names = [
        "First pipeline\n(before the fix)",
        "Handcrafted\nbaseline\n(after the fix)",
        "AASIST,\npublished\nweights",
        "VoiceGuard\ndetector",
    ]
    vals = [
        14.0,
        59.0,
        _eer("cm_aasist-pretrained_silence_ablation.csv", "silence_only"),
        _eer("cm_aasist-AASIST_robust2_silence_ablation.csv", "silence_only"),
    ]
    cols = [RED, GREEN, GREEN, GREEN]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.axhspan(0, 40, color="#fbe4e1", zorder=0)
    ax.axhline(50, color=INK, ls="--", lw=1.8)
    ax.text(3.45, 51.2, "50 % = guessing", ha="right", fontsize=13, color=INK)
    ax.text(
        -0.42,
        38.5,
        "below 40 %:\nreading the\nrecording,\nnot the voice",
        fontsize=12.5,
        color=RED,
        ha="left",
        va="top",
        linespacing=1.25,
    )
    bars = ax.bar(names, vals, color=cols, width=0.55, zorder=2)
    _label(ax, bars, "{:.0f} %", dy=1.2)
    ax.errorbar([0], [14.0], yerr=[[1.0], [1.0]], color=INK, capsize=6, lw=2, zorder=3)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Error using only the silent parts")
    ax.set_title(
        "The test that exposed a hidden shortcut, and the gate that guards it",
        loc="left",
        fontweight="bold",
        fontsize=17,
    )
    _save(fig, "2_silence_shortcut.png")


# 3 -- architecture -------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, body, ec=BLUE, title_lines=1, title_size=15.5, body_size=11.5):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#ffffff", ec=ec, lw=2.2
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        linespacing=1.1,
    )
    ax.text(
        x + w / 2,
        y + h - 0.3 - 0.52 * title_lines,
        body,
        ha="center",
        va="top",
        fontsize=body_size,
        color=MUTED,
        linespacing=1.35,
    )


def _arrow(ax, p, q, color=MUTED, ls="-", lw=2.2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=20,
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(19, 8.8))
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 12.2)
    ax.axis("off")
    ax.text(
        0.3,
        11.95,
        "How VoiceGuard decides, several times a second during a call",
        fontsize=23,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.3,
        11.2,
        "No single detector can be trusted on every kind of fake, so four partly independent kinds of evidence are combined, and a person decides.",
        fontsize=14,
        color=MUTED,
        va="top",
    )

    # main flow, left to right
    _box(ax, 0.3, 5.4, 2.8, 2.5, "Live call", "phone, VoIP\nor microphone", ec=GREY, body_size=11)
    _box(
        ax,
        3.7,
        5.4,
        3.7,
        2.5,
        "Input gate",
        "repairs bad audio and\nfinds the speech; says\nWAITING when there\nis nothing to judge",
        ec=GREEN,
        body_size=10.5,
    )
    _box(
        ax,
        8.0,
        5.4,
        3.7,
        2.5,
        "Fair front-end",
        "16 kHz, trim, loudness,\nidentical for real and\nfake, so the recording\ncannot give it away",
        ec=GREEN,
        body_size=10.5,
    )

    # the four kinds of evidence
    ex, ew, eh = 12.5, 5.4, 1.62
    _box(
        ax,
        ex,
        8.55,
        ew,
        eh,
        "1  Voice detector",
        "AASIST neural network, fine-tuned\nfor noise, rooms and phone lines",
        ec=BLUE,
        body_size=10.5,
    )
    _box(
        ax,
        ex,
        6.7,
        ew,
        eh,
        "2  Voice match",
        "WavLM voiceprint compared with\nthe customer's voice on file",
        ec=BLUE,
        body_size=10.5,
    )
    _box(
        ax,
        ex,
        4.85,
        ew,
        eh,
        "3  Recording style",
        "noise, echo, clipping, band: sets\nhow far to trust the detector",
        ec=AMBER,
        body_size=10.5,
    )
    _box(
        ax,
        ex,
        3.0,
        ew,
        eh,
        "4  Call details",
        "known number? amount? earlier\nwarnings? (not from the audio)",
        ec=GREY,
        body_size=10.5,
    )

    _box(
        ax,
        18.7,
        4.75,
        3.3,
        3.6,
        "Trust-weighted\nfusion",
        "clone check, voice\ncheck and call details\nas separate alarms",
        ec=BLUE,
        title_lines=2,
        title_size=14,
        body_size=10.5,
    )
    _box(
        ax,
        22.7,
        7.3,
        3.0,
        2.3,
        "Policy",
        "routine, payment or\nsensitive call: how\nmuch evidence\nis enough",
        ec=BLUE,
        title_size=14.5,
        body_size=10.5,
    )
    ax.add_patch(
        FancyBboxPatch(
            (22.7, 3.7),
            3.0,
            2.95,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc="#f4f7fb",
            ec=INK,
            lw=2.2,
        )
    )
    ax.text(
        24.2, 6.45, "Answer", ha="center", va="top", fontsize=14.5, fontweight="bold", color=INK
    )
    for i, (txt, col) in enumerate([("ALLOW", GREEN), ("VERIFY", AMBER), ("ESCALATE", RED)]):
        ax.text(
            24.2,
            5.55 - i * 0.7,
            txt,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.28", fc=col, ec=col),
        )

    # arrows
    _arrow(ax, (3.1, 6.65), (3.7, 6.65))
    _arrow(ax, (7.4, 6.65), (8.0, 6.65))
    for c in (9.36, 7.51, 5.66):
        _arrow(ax, (11.7, 6.65), (ex, c))
    ax.text(11.05, 4.0, "call system", fontsize=10, color=MUTED, ha="left", va="bottom")
    _arrow(ax, (11.0, 3.81), (ex, 3.81))
    for c, d in ((9.36, 7.9), (7.51, 7.15), (5.66, 6.4), (3.81, 5.6)):
        _arrow(ax, (ex + ew, c), (18.7, d))
    _arrow(ax, (22.0, 6.9), (22.7, 8.2))
    _arrow(ax, (24.2, 7.3), (24.2, 6.65))

    # what surrounds the decision
    ax.add_patch(
        FancyBboxPatch(
            (0.3, 0.35),
            10.6,
            3.85,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc="#f4f7fb",
            ec=LINE,
            lw=1.6,
        )
    )
    ax.text(0.65, 3.95, "Around the decision", fontsize=15, fontweight="bold", color=INK, va="top")
    bullets = [
        'Console: a plain-language answer, the reasons, an "I\'ve seen this" button',
        "One engine over REST, WebSocket and gRPC (tested to agree exactly)",
        "Audit log: scores and reasons only, never audio, text or a voiceprint",
        "A person acts on the answer: decision support, not an automatic block",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.7, 3.2 - i * 0.68, "\u2022  " + b, fontsize=11, color=MUTED, va="center")
    _save(fig, "3_architecture.png")


# 4 -- fusion beats any single signal --------------------------------------------------------
def fig_fusion() -> None:
    """Balanced accuracy on speakers the model never saw (19 situations, 4,157 reads). Sources:
    docs/system_eval.md; baselines from scripts/fit_fusion.py (same folds); shipped row from
    results/tables/clone_head_variants.csv."""
    from matplotlib.lines import Line2D

    v = pd.read_csv(TAB / "clone_head_variants.csv")
    shipped = float(
        v[
            (v.clone_head.str.contains("gain")) & (v.evaluation == "speaker-held-out (OOF)")
        ].balanced.iloc[0]
    )
    dev = float(
        v[(v.clone_head.str.contains("gain")) & (v.evaluation == "untouched dev")].balanced.iloc[0]
    )
    names = [
        "The detector alone",
        "Hand-set mix of the signals",
        "VoiceGuard's learned,\nline-aware fusion",
    ]
    vals = [73.6, 83.9, shipped]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.barh(names[::-1], vals[::-1], color=[BLUE, "#7a9be8", GREY], height=0.55)
    for b in bars:
        ax.text(
            b.get_width() + 0.6,
            b.get_y() + b.get_height() / 2,
            f"{b.get_width():.1f} %",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=INK,
        )
    ax.axvline(dev, color=GREEN, ls="--", lw=2)
    ax.legend(
        [Line2D([0], [0], color=GREEN, ls="--", lw=2)],
        [f"{dev:.1f} % on 10 speakers never used for fitting or choosing"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=12.5,
    )
    ax.set_xlim(50, 100)
    ax.set_xlabel("Balanced accuracy (mean of genuine callers accepted and attacks flagged)")
    ax.set_title(
        "Combining weak signals beats trusting one",
        loc="left",
        fontweight="bold",
        fontsize=17,
        pad=12,
    )
    _save(fig, "4_fusion_beats_single_signals.png")


# 5 -- robustness -----------------------------------------------------------------------------
def fig_robustness() -> None:
    """Left: balanced accuracy over the 14 situations the first system had never seen (docs/stress_eval.md).
    Right: the shipped system per situation on 10 untouched speakers (stress_eval_final_dev.csv)."""
    d = pd.read_csv(TAB / "stress_eval_final_dev.csv").sort_values("bal")
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(15.5, 6.6), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.42}
    )
    labels = ["Before:\nnever seen", "After:\nnever seen", "After:\nuntouched\nspeakers"]
    vals = [78.6, 86.6, 91.3]  # docs/stress_eval.md "The shipped system"
    bars = a.bar(labels, vals, color=[RED, AMBER, GREEN], width=0.6)
    _label(a, bars, "{:.1f} %", dy=0.5)
    a.set_ylim(60, 100)
    a.set_ylabel("Balanced accuracy, mean of 14 situations")
    a.set_title("Fourteen awkward situations", loc="left", fontweight="bold", fontsize=15)
    a.tick_params(axis="x", labelsize=13)
    cols = [GREEN if x >= 90 else AMBER if x >= 82 else RED for x in d.bal * 100]
    b.barh(d.cond, d.bal * 100, color=cols, height=0.62)
    for i, v in enumerate(d.bal * 100):
        b.text(v + 0.4, i, f"{v:.0f} %", va="center", fontsize=12, color=INK)
    b.set_xlim(60, 102)
    b.set_xlabel("Balanced accuracy on untouched speakers")
    b.set_title(
        "Where it is strong, and where it is still weak", loc="left", fontweight="bold", fontsize=15
    )
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
        b.spines[s].set_visible(False)
    fig.suptitle(
        "Rooms, echo, clipping, packet loss, hum, music, phone codecs",
        x=0.01,
        ha="left",
        fontsize=13,
        color=MUTED,
        y=1.0,
    )
    _save(fig, "5_robustness.png")


# 6 -- modern clones ---------------------------------------------------------------------------
def fig_modern() -> None:
    """Share of calls flagged (VERIFY or ESCALATE) for modern cloning tools, shipped system.
    Source: results/tables/modern_by_attack_gain_{dev,eval}.csv (dev 20 calls, eval 48 per cell)."""
    rows = []
    for sp in ("dev", "eval"):
        t = pd.read_csv(TAB / f"modern_by_attack_gain_{sp}.csv")
        t["split"] = sp
        rows.append(t)
    t = pd.concat(rows)
    t = t[t.attack.isin(["knnvc", "speecht5x", "mms"])]

    def pooled(attack: str, cond: str) -> float:
        s = t[(t.attack == attack) & (t.cond == cond)]
        return float((s.flagged * s.n).sum() / s.n.sum() * 100)

    conds = [
        ("clean", "Clean line"),
        ("babble10", "Noisy room\n(voices at 10 dB)"),
        ("ulaw", "Phone line\n(G.711)"),
    ]
    attacks = [
        ("knnvc", "kNN-VC voice conversion\n(open-source cloning)", BLUE),
        ("speecht5x", "SpeechT5 + speaker vector\n(laptop-class cloning)", AMBER),
        ("mms", "MMS-TTS speech\n(nobody's real voice)", GREEN),
    ]
    x = np.arange(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for k, (a, name, col) in enumerate(attacks):
        vals = [pooled(a, c) for c, _ in conds]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=name)
        _label(ax, bars, "{:.0f} %", dy=1.0)
    ax.set_xticks(x, [c[1] for c in conds])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Share of cloned calls flagged (higher is better)")
    ax.set_title(
        "Newer cloning tools: most are caught, a real share is not",
        loc="left",
        fontweight="bold",
        fontsize=17,
    )
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    ax.text(
        0.0,
        -0.22,
        "The detector alone sees only 2-3 % of the kNN-VC and SpeechT5 calls; the voice match does most of the catching.",
        transform=ax.transAxes,
        fontsize=11.5,
        color=MUTED,
    )
    _save(fig, "6_modern_clones.png")


# 7 -- why a real voice note was flagged ----------------------------------------------------------
def fig_lesson() -> None:
    """Log-odds contributions to the clone head (shipped until 2026-10-23) for a real WhatsApp voice
    note and a studio sample call. Source: docs/real_world_audio.md (term-by-term decomposition)."""
    terms = [
        "The detector's score",
        "Recording style\n(noise floor, band, envelope)",
        "Detector x style",
        "Baseline",
    ]
    note = [-0.29, 2.81, -0.04, 1.39]
    studio = [-2.63, -4.01, -7.74, 1.39]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    for ax, vals, title, tot in (
        (axes[0], note, "A real friend's WhatsApp voice note", sum(note)),
        (axes[1], studio, "A clean studio sample call", sum(studio)),
    ):
        cols = [RED if v > 0 else GREEN for v in vals]
        bars = ax.barh(terms[::-1], vals[::-1], color=cols[::-1], height=0.6)
        for b_ in bars:
            v = b_.get_width()
            ax.text(
                v + (0.15 if v >= 0 else -0.15),
                b_.get_y() + b_.get_height() / 2,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=13,
            )
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_title(
            f'{title}\ntotal {tot:+.2f}  (> 0 means "synthetic")',
            loc="left",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(-10.5, 4.5)
    fig.suptitle(
        "Why the old clone check called a real person fake: recording style alone pushed it over the line",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    _save(fig, "7_why_a_real_voice_note_was_flagged.png")


if __name__ == "__main__":
    fig_honest_gap()
    fig_silence()
    fig_architecture()
    fig_fusion()
    fig_robustness()
    fig_modern()
    fig_lesson()
