"""Fit the channel-aware learned fusion and measure it honestly.

    python scripts/fit_fusion.py --tag _v2                      # report only
    python scripts/fit_fusion.py --tag _v2,_stress --ship       # + write models/fusion/fusion.json

Input: results/tables/system_trials{tag}_{eval,dev}.csv from scripts/build_system_trials.py
(genuine / clone / impostor calls replayed under several acoustic situations, with the raw
spoof / speaker reads and channel descriptors cached). ``--tag`` takes a comma-separated
list; their eval tables are fitted together.

Protocol
  * model       two logistic heads (clone, impostor) over [signal, channel, signal x channel]
                (see voiceguard.risk.learned); thresholds chosen on training speakers only;
                channel features clamped to the range seen in training
  * speakers    4-fold cross-validation grouped by SPEAKER on the eval split, so no person
                appears on both sides; pooled out-of-fold (OOF) numbers are reported
  * situations  LEAVE-SITUATIONS-OUT: for each group of acoustic situations (below) the heads
                are fitted on every OTHER situation and on other speakers, then scored on the
                held-out group -- an estimate of behaviour in a situation never seen, which
                the speaker-grouped OOF number cannot give
  * dev split   different speakers, known attacks. Never used for fitting or thresholds
  * baselines   the hand-set weighted mean (risk.fuse) and spoof-only, each with its single
                best threshold, scored on the very same OOF folds
An "attack" call is a clone or a different human. Flag = anything not ALLOW. A read the system
declined to judge (no usable speech) is excluded from every rate and counted in ``ans``.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from voiceguard.config import REPO_ROOT
from voiceguard.eval import compute_eer, git_commit
from voiceguard.risk import Signal, fuse
from voiceguard.risk.learned import (
    ALLOWED_CHANNEL,
    CHANNEL,
    DEFAULT_PATH,
    LearnedFusion,
    head_features,
    tree_features,
)

TAB = REPO_ROOT / "results" / "tables"
STANDARD = ["clean", "babble10", "babble5", "pink10", "ulaw"]
# situations that share a failure mechanism are held out together
SITUATION_GROUPS = {
    "reverberant": ["reverb", "speakerphone", "tel_reverb"],
    "distorted": ["clip", "loss", "burst"],
    "steady background": ["hum", "music", "tel_babble"],
    "level / band": ["narrow", "quiet", "opus8k", "g722", "amr"],
}


def _load(split: str, tags: list[str], names=CHANNEL) -> pd.DataFrame | None:
    parts = []
    for tag in tags:
        p = TAB / f"system_trials{tag}_{split}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if "floor_rel_db" not in d.columns:  # tables built before the gain-invariant descriptor
            d["floor_rel_db"] = d["floor_db"] - d["level_db"]
        if not set(names) <= set(d.columns):
            missing = sorted(set(names) - set(d.columns))
            raise SystemExit(f"{p.name} lacks channel features {missing} -- rerun the builder")
        d["tag"] = tag
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def _detector_of(*tables) -> dict | None:
    """The one (detector, calibration) every table was built with, or None for old tables that
    do not record it. Mixed detectors cannot be fitted together: their scores are on
    different scales, so one set of thresholds would be wrong for all of them."""
    seen, any_recorded = set(), False
    for t in tables:
        if t is None:
            continue
        if {"detector", "op_point"} <= set(t.columns):
            any_recorded = True
            names, ops = t.detector.fillna("(unrecorded)"), t.op_point.round(6).fillna(-1.0)
            seen |= set(zip(names, ops, strict=True))
        else:  # a table that predates the record is a detector of its own
            seen.add(("(unrecorded)", -1.0))
    if len(seen) > 1:
        raise SystemExit(f"the tables were built with different detectors: {sorted(seen)}")
    if not any_recorded:
        return None
    name, op = next(iter(seen))
    return {"name": name, "operating_point": float(op)}


def _answered(d: pd.DataFrame) -> pd.DataFrame:
    return d[d.spoof.notna()].reset_index(drop=True)


def _hand_set(r):
    sig = [
        Signal("spoof", "", r.spoof, 0.45, ""),
        Signal("speaker", "", None if pd.isna(r.speaker_risk) else r.speaker_risk, 0.35, ""),
        Signal("prosody", "", None if pd.isna(r.prosody) else r.prosody, 0.25, ""),
        Signal("context", "", 0.0, 0.20, ""),
    ]
    return fuse(sig).score


def _clamp_of(train: pd.DataFrame, names=CHANNEL):
    return (
        train[list(names)].min().to_numpy(float),
        train[list(names)].max().to_numpy(float),
    )


def _X(d, sig_cols, clamp, names=CHANNEL, main_effects=True):
    one = len(sig_cols) == 1
    return np.stack(
        [
            head_features(
                r[sig_cols[0]] if one else [r[c] for c in sig_cols],
                {k: r[k] for k in names},
                clamp,
                names,
                main_effects,
            )
            for _, r in d.iterrows()
        ]
    )


def _fit_head(d, sig_cols, attack_kind, C, clamp, names=CHANNEL, main_effects=True):
    sub = d[d.kind.isin(["genuine", attack_kind])]
    sub = sub[sub[sig_cols].notna().all(axis=1)]
    X = _X(sub, sig_cols, clamp, names, main_effects)
    y = (sub.kind == attack_kind).astype(int).values
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    return dict(
        mean=mu.tolist(),
        std=sd.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        main_effects=bool(main_effects),
    )


GBM = dict(max_depth=3, max_iter=80, learning_rate=0.08, min_samples_leaf=40, l2_regularization=1.0)


def _export_gbm(clf) -> dict:
    """A fitted HistGradientBoostingClassifier as the plain lists ``risk.learned._TreeHead``
    walks (scikit-learn is needed to fit, never to serve). tests/test_fit_fusion.py checks that
    the export reproduces ``predict_proba``, so a scikit-learn change that breaks it fails
    loudly instead of shipping a wrong model."""
    trees = []
    for stage in clf._predictors:
        n = stage[0].nodes
        trees.append(
            dict(
                feature=np.where(n["is_leaf"], -1, n["feature_idx"]).astype(int).tolist(),
                threshold=np.round(n["num_threshold"].astype(float), 9).tolist(),
                left=n["left"].astype(int).tolist(),
                right=n["right"].astype(int).tolist(),
                value=np.round(n["value"].astype(float), 9).tolist(),
            )
        )
    return dict(type="gbm", baseline=float(np.ravel(clf._baseline_prediction)[0]), trees=trees)


def _fit_gbm_head(d, sig_col, attack_kind, clamp, names=CHANNEL):
    return _export_gbm(_fit_gbm(d, sig_col, attack_kind, clamp, names)[0])


def _fit_gbm(d, sig_col, attack_kind, clamp, names=CHANNEL):
    """-> (fitted classifier, its training inputs). Boosted trees over [logit(signal), channel],
    monotone in the signal: a higher suspicion score can never lower the attack probability."""
    sub = d[d.kind.isin(["genuine", attack_kind])]
    sub = sub[sub[sig_col].notna()]
    X = np.stack(
        [
            tree_features(r[sig_col], {k: r[k] for k in names}, clamp, names)
            for _, r in sub.iterrows()
        ]
    )
    y = (sub.kind == attack_kind).astype(int).values
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # balanced classes
    clf = HistGradientBoostingClassifier(
        **GBM, monotonic_cst=[1] + [0] * len(names), random_state=0
    ).fit(X, y, sample_weight=w)
    return clf, X


def _fit_gain_head(d, sig_col, attack_kind, C, clamp, names=CHANNEL):
    """The trust-weighted head (``risk.learned._GainHead``): ``p = sigmoid(offset + g * logit(s))``
    with ``g = exp(w0 + w . u)`` the channel's gain on the detector. Fitted by L-BFGS on the same
    balanced logistic loss the linear heads use, with the same strength of L2 on the channel
    weights (``w``); the gain is positive by construction, so the head is monotone in the
    detector score and cannot alarm on the channel alone."""
    sub = d[d.kind.isin(["genuine", attack_kind])]
    sub = sub[sub[sig_col].notna()]
    X = np.stack(
        [
            tree_features(r[sig_col], {k: r[k] for k in names}, clamp, names)
            for _, r in sub.iterrows()
        ]
    )
    y = (sub.kind == attack_kind).astype(float).values
    L, D = X[:, 0], X[:, 1:]
    mu, sd = D.mean(0), D.std(0) + 1e-6
    U = (D - mu) / sd
    sw = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))  # balanced; sums to 1
    lam = 1.0 / (2.0 * C * len(y))  # sklearn's C, expressed against the weights above

    def loss(theta):
        off, w0, w = theta[0], theta[1], theta[2:]
        lg = np.clip(w0 + U @ w, -6.0, 6.0)
        g = np.exp(lg)
        z = off + g * L
        ce = np.logaddexp(0.0, z) - y * z
        pz = 1.0 / (1.0 + np.exp(-z))
        dz = sw * (pz - y)
        free = (np.abs(w0 + U @ w) < 6.0).astype(float)  # the clip has no gradient
        dlg = dz * g * L * free
        grad = np.concatenate([[dz.sum(), dlg.sum()], U.T @ dlg + 2.0 * lam * w])
        return float((sw * ce).sum() + lam * (w @ w)), grad

    theta0 = np.zeros(2 + U.shape[1])
    res = minimize(loss, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    off, w0, w = res.x[0], res.x[1], res.x[2:]
    return dict(
        type="gain",
        mean=mu.tolist(),
        std=sd.tolist(),
        w0=float(w0),
        w=w.tolist(),
        offset=float(off),
    )


def _p_attack(spec, d):
    lf = LearnedFusion({**spec, "t_warn": 0.5, "t_esc": 0.9})
    alt = lf.clone_signals == 2
    return np.array(
        [
            lf.probs(
                r.spoof,
                None if pd.isna(r.speaker_risk) else r.speaker_risk,
                {k: getattr(r, k) for k in lf.channel},
                spoof_alt=r.spoof_alt if alt else None,
            )[0]
            for r in d.itertuples()
        ]
    )


def _thresholds(y_attack, p, fpr_cap):
    """t_warn: best balanced accuracy. t_esc: lowest p keeping genuine ESCALATE-rate <= fpr_cap."""
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    bal = [((p[y_attack == 0] < t).mean() + (p[y_attack == 1] >= t).mean()) / 2 for t in ts]
    t_warn = float(ts[int(np.argmax(bal))])
    gen = np.sort(p[y_attack == 0])
    t_esc = float(max(gen[int(np.ceil((1 - fpr_cap) * len(gen))) - 1], t_warn))
    return t_warn, min(t_esc + 1e-6, 0.999)


def _heads(train, C, alt, names, head, clone_form="full"):
    """The two fitted heads plus the channel bookkeeping (no operating points yet).
    ``clone_form``: "full" = linear in the signal, the channel and their products (the form
    shipped up to 2026-10-23); "interactions" = no bare channel terms; "gain" = the channel
    scales the detector's logit by a positive gain (``_fit_gain_head``)."""
    clamp = _clamp_of(train, names)
    spec = {
        "clone_signals": 2 if alt else 1,
        "channel": list(names),
        "channel_range": {k: [float(clamp[0][i]), float(clamp[1][i])] for i, k in enumerate(names)},
        "head": head,
    }
    if head == "gbm":
        if alt:
            raise SystemExit("--alt (two detector scores) needs the logistic head")
        spec["clone"] = _fit_gbm_head(train, "spoof", "clone", clamp, names)
        spec["impostor"] = _fit_gbm_head(train, "speaker_risk", "impostor", clamp, names)
    else:
        if clone_form == "gain":
            if alt:
                raise SystemExit("--clone-channel-terms gain takes a single detector score")
            spec["clone"] = _fit_gain_head(train, "spoof", "clone", C, clamp, names)
        else:
            spec["clone"] = _fit_head(
                train,
                ["spoof", "spoof_alt"] if alt else ["spoof"],
                "clone",
                C,
                clamp,
                names,
                clone_form == "full",
            )
        spec["impostor"] = _fit_head(train, ["speaker_risk"], "impostor", C, clamp, names)
    return spec


def _fit_all(train, C, fpr_cap, alt=False, names=CHANNEL, head="logistic", clone_form="full"):
    spec = _heads(train, C, alt, names, head, clone_form)
    y = (train.kind != "genuine").astype(int).values
    if head == "gbm":
        # trees reproduce their own training data almost perfectly, so thresholds taken from
        # in-sample predictions would sit far too high: use out-of-fold ones, by speaker
        p = np.zeros(len(train))
        for a, b in GroupKFold(3).split(train, y, train.speaker):
            p[b] = _p_attack(_heads(train.iloc[a], C, alt, names, head, clone_form), train.iloc[b])
    else:
        p = _p_attack(spec, train)
    spec["t_warn"], spec["t_esc"] = _thresholds(y, p, fpr_cap)
    return spec


def _table(d, flag, esc, title):
    print(f"\n{title}  (n={len(d)})")
    print(
        f"{'condition':12s} {'acc':>6s} {'bal':>6s} {'genOK':>6s} "
        f"{'clone':>6s} {'imp':>6s} {'ESCgen':>7s} {'ESCatk':>7s}"
    )
    gen_all = (d.kind == "genuine").values
    conds = list(dict.fromkeys(d.cond))
    for c in ["ALL", *conds]:
        m = np.ones(len(d), bool) if c == "ALL" else (d.cond == c).values
        f, g = flag[m], gen_all[m]
        cl, im = (d.kind == "clone").values[m], (d.kind == "impostor").values[m]
        if not g.any() or g.all():
            continue
        gok = (~f[g]).mean()
        print(
            f"{c:12s} {(f != g).mean():6.1%} {(gok + f[~g].mean()) / 2:6.1%} {gok:6.1%} "
            f"{f[cl].mean() if cl.any() else float('nan'):6.1%} "
            f"{f[im].mean() if im.any() else float('nan'):6.1%} {esc[m][g].mean():7.1%} "
            f"{esc[m][~g].mean():7.1%}"
        )


def _best_single(y, s):
    ts = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    bal = [((s[y == 0] < t).mean() + (s[y == 1] >= t).mean()) / 2 for t in ts]
    return float(ts[int(np.argmax(bal))])


def _leave_situations_out(
    ev, C, fpr_cap, names=CHANNEL, head="logistic", folds=4, clone_form="full"
):
    """Speaker-held-out AND situation-held-out: fit on other speakers x other situations."""
    present = [
        (name, [c for c in cs if c in set(ev.cond)]) for name, cs in SITUATION_GROUPS.items()
    ]
    present = [(n, cs) for n, cs in present if cs]
    flag, esc, seen = np.zeros(len(ev), bool), np.zeros(len(ev), bool), np.zeros(len(ev), bool)
    y = (ev.kind != "genuine").astype(int).values
    for _name, conds in present:
        held = ev.cond.isin(conds).values
        for tr_i, te_i in GroupKFold(folds).split(ev, y, ev.speaker):
            te_i = np.intersect1d(te_i, np.where(held)[0])
            tr_i = np.intersect1d(tr_i, np.where(~held)[0])
            if len(te_i) == 0 or len(tr_i) == 0:
                continue
            spec = _fit_all(
                ev.iloc[tr_i], C, fpr_cap, names=names, head=head, clone_form=clone_form
            )
            p = _p_attack(spec, ev.iloc[te_i])
            flag[te_i], esc[te_i], seen[te_i] = p >= spec["t_warn"], p >= spec["t_esc"], True
    if not seen.any():
        return
    sub = ev[seen].reset_index(drop=True)
    _table(
        sub,
        flag[seen],
        esc[seen],
        "UNSEEN SITUATIONS  [fit without the situation AND without the speaker]",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", action="store_true", help="write models/fusion/fusion.json")
    ap.add_argument("--out", default=None, help="write the fitted model to this json instead")
    ap.add_argument("--tag", default="_v2", help="trial table tag(s), comma-separated")
    ap.add_argument(
        "--alt", action="store_true", help="clone head also uses the upstream-AASIST score"
    )
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--esc-fpr", type=float, default=0.01, help="max genuine ESCALATE rate")
    ap.add_argument("--no-lso", action="store_true", help="skip the leave-situations-out check")
    ap.add_argument(
        "--channel",
        default=",".join(CHANNEL),
        help=f"comma-separated channel descriptors the heads see (default: the shipped four; "
        f"any of {', '.join(ALLOWED_CHANNEL)})",
    )
    ap.add_argument(
        "--head",
        default="logistic",
        choices=["logistic", "gbm"],
        help="logistic: linear in the signal x channel (the shipped form); gbm: monotone boosted "
        "trees, exported to json and evaluated in numpy",
    )
    ap.add_argument(
        "--clone-channel-terms",
        default="full",
        choices=["full", "interactions", "gain"],
        help="full: the clone head sees bare channel descriptors as well as their interaction "
        "with the detector score (the shipped form up to 2026-10-23); interactions: the channel "
        "can only reweigh the detector's score, never raise an alarm by itself (but the sign of "
        "that reweighing is free, so it can still flip); gain: the channel scales the detector's "
        "logit by a positive gain: monotone in the detector, no alarm without it",
    )
    args = ap.parse_args()
    clone_form = args.clone_channel_terms
    tags = args.tag.split(",")
    names = tuple(args.channel.split(","))
    if bad := [k for k in names if k not in ALLOWED_CHANNEL]:
        raise SystemExit(f"unknown / gain-dependent channel descriptors: {bad}")

    ev, dv = _load("eval", tags, names), _load("dev", tags, names)
    if ev is None:
        raise SystemExit("run scripts/build_system_trials.py first")
    detector = _detector_of(ev, dv)
    n_all = len(ev)
    ev = _answered(ev)
    y = (ev.kind != "genuine").astype(int).values
    dev_note = "-" if dv is None else f"{dv.speaker.nunique()} speakers, {len(dv)} reads"
    print(
        f"eval: {ev.speaker.nunique()} speakers, {len(ev)} reads "
        f"({n_all - len(ev)} unjudged dropped) | dev: {dev_note}"
    )

    # ---- out-of-fold, grouped by speaker ----
    p_oof = np.zeros(len(ev))
    base_hand, base_spoof = np.zeros(len(ev), bool), np.zeros(len(ev), bool)
    esc_oof, flag_oof = np.zeros(len(ev), bool), np.zeros(len(ev), bool)
    hand = np.array([_hand_set(r) for r in ev.itertuples()])
    for tr_i, te_i in GroupKFold(4).split(ev, y, ev.speaker):
        tr, te = ev.iloc[tr_i], ev.iloc[te_i]
        spec = _fit_all(tr, args.C, args.esc_fpr, args.alt, names, args.head, clone_form)
        p_oof[te_i] = _p_attack(spec, te)
        flag_oof[te_i] = p_oof[te_i] >= spec["t_warn"]
        esc_oof[te_i] = p_oof[te_i] >= spec["t_esc"]
        base_hand[te_i] = hand[te_i] >= _best_single(y[tr_i], hand[tr_i])
        base_spoof[te_i] = ev.spoof.values[te_i] >= _best_single(y[tr_i], ev.spoof.values[tr_i])

    z = np.zeros(len(ev), bool)
    _table(ev, base_spoof, z, "BASELINE spoof score alone, one threshold  [OOF]")
    _table(ev, base_hand, z, "BASELINE hand-set weighted mean, one threshold  [OOF]")
    _table(ev, flag_oof, esc_oof, "LEARNED channel-aware fusion  [OOF, unseen speakers]")
    print(f"\neval OOF EER (genuine vs attack): {compute_eer(1 - y, -p_oof)[0]:.1%}")

    if not args.no_lso:
        _leave_situations_out(ev, args.C, args.esc_fpr, names, args.head, clone_form=clone_form)

    # ---- final model: all eval speakers; dev = untouched test ----
    spec = _fit_all(ev, args.C, args.esc_fpr, args.alt, names, args.head, clone_form)
    print(f"\nshipped operating points: t_warn={spec['t_warn']:.3f}  t_esc={spec['t_esc']:.3f}")
    if dv is not None:
        dv = _answered(dv)
        p = _p_attack(spec, dv)
        _table(
            dv,
            p >= spec["t_warn"],
            p >= spec["t_esc"],
            "LEARNED fusion on DEV  [untouched, known attacks]",
        )
        yd = (dv.kind != "genuine").astype(int).values
        print(f"dev EER (genuine vs attack): {compute_eer(1 - yd, -p)[0]:.1%}")

    if args.ship or args.out:
        if detector:  # which detector + calibration these reads came from (tables, not the CLI)
            spec["detector"] = detector
        else:
            print("note: these tables do not record their detector; the model will not either")
        spec.update(
            git_commit=git_commit(),
            trials_tag=args.tag,
            n_eval_reads=int(len(ev)),
            n_eval_speakers=int(ev.speaker.nunique()),
            situations=sorted(set(ev.cond)),
            note="fit by scripts/fit_fusion.py; prosody excluded (~chance); "
            f"gain-invariant channel; clone head: {clone_form}",
        )
        dest = Path(args.out) if args.out else DEFAULT_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print("wrote ->", dest)


if __name__ == "__main__":
    main()
