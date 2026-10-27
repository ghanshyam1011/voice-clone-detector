"""The poster builder: its numbers come from the results tables, and it builds without a browser."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_poster.py"
_spec = importlib.util.spec_from_file_location("make_poster", _PATH)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

_TABLES = (
    "clone_head_variants.csv",
    "stress_eval_final_dev.csv",
    "modern_by_attack_gain_dev.csv",
    "modern_by_attack_gain_eval.csv",
    "cm_aasist-pretrained_in_domain.csv",
    "cm_aasist-pretrained_cross_dataset.csv",
    "cm_aasist-pretrained_silence_ablation.csv",
    "cm_aasist-AASIST_robust2_in_domain.csv",
    "cm_aasist-AASIST_robust2_cross_dataset.csv",
    "cm_aasist-AASIST_robust2_silence_ablation.csv",
)
_SHOTS = ("console_real.png", "console_fake.png", "console_hard.png")
pytestmark = pytest.mark.skipif(
    not all((mp.TAB / t).exists() for t in _TABLES)
    or not all((mp.SHOTS / s).exists() for s in _SHOTS),
    reason="results tables or console screenshots are not present",
)

FULL = {
    "subject": "Machine Learning",
    "faculty": "Prof. A B",
    "college": "Some Institute",
    "term": "BE Sem 7",
    "members": [("Asha Rao", "21CS001"), ("Ravi Kumar", "21CS002")],
}


def test_rounding_is_half_up():
    assert mp.r0(82.5) == 83  # Python's round() would say 82
    assert mp.r0(35.8) == 36
    assert mp.r0(0.4) == 0


def test_numbers_match_the_documented_results():
    n = mp.load_numbers()
    assert mp.r0(n["gap_pre"][2]) == 36 and mp.r0(n["gap_ours"][2]) == 36  # In-the-Wild
    assert n["fusion"][2] == pytest.approx(90.2) and n["fusion_dev"] == pytest.approx(91.8)
    assert [mp.r0(v) for v in n["modern"]["knnvc"]] == [72, 99, 87]
    assert [mp.r0(v) for v in n["modern"]["speecht5x"]] == [50, 55, 60]
    assert [mp.r0(v) for v in n["modern"]["mms"]] == [100, 100, 100]
    assert [c for c, _ in n["stress_weakest"]] == [
        "tel_babble",
        "clip",
        "tel_reverb",
        "speakerphone",
    ]
    assert [mp.r0(v) for v in n["silence"]] == [14, 59, 41, 45]


def test_html_carries_every_headline_number():
    html = mp.build_html({})
    for needle in ("36%", "13–15%", "90%", "86.6%", "72%", "SIH26104", "268"):
        assert needle in html, needle
    assert "never used for training" in html
    assert html.count("<svg") >= 10  # the charts, the diagram, the icons and the QR code


def test_every_item_on_the_college_checklist_is_present():
    html = mp.build_html(FULL)
    for needle in (
        "VoiceGuard",  # project title
        "Group members",
        "Roll no.",
        "Asha Rao",
        "21CS002",  # members and roll numbers
        "Machine Learning",
        "Prof. A B",  # subject and faculty
        "Problem statement",
        "Objectives",
        "Methodology / workflow diagram",
        "Technologies and tools used",
        "Results",
        "Conclusion",
        "Faculty Signature",
        "External Examiner's Signature",
    ):
        assert needle in html, needle


def test_signature_boxes_come_last():
    html = mp.build_html(FULL)
    assert html.index("Faculty Signature") > html.index("References.")
    assert html.index("External Examiner's Signature") > html.index("Faculty Signature")
    assert html.index("External Examiner's Signature") > html.index("Conclusion")
    assert html.count('class="sigbox"') == 2


def test_placeholders_stay_visible_until_details_are_given():
    bare = mp.build_html({})
    for ph in ("[Subject name]", "[Faculty name]", "[College name]", "[Member name]", "[Roll no.]"):
        assert ph in bare, ph
    full = mp.build_html(FULL)
    assert "[" not in full.split("</header>")[0].split('class="idcard"')[1]


def test_a_single_member_is_labelled_in_the_singular():
    one = mp.build_html({"members": [("Solo Person", "99X")]})
    assert "Group member<" in one and "Group members" not in one
    assert "Group members" in mp.build_html(FULL)


def test_names_are_escaped():
    html = mp.build_html({"subject": "<b>x</b> & y"})
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in html


def test_members_are_parsed_from_the_command_line_form():
    assert mp.parse_members("Asha Rao, 21CS001; Ravi Kumar, 21CS002") == [
        ("Asha Rao", "21CS001"),
        ("Ravi Kumar", "21CS002"),
    ]
    assert mp.parse_members("Solo Name") == [("Solo Name", "")]
    assert mp.parse_members(None) == [] and mp.parse_members("") == []


def test_wrap_never_exceeds_the_box():
    text = "Repairs bad audio; says WAITING when there is nothing to judge."
    for width in (180, 280, 400):
        for line in mp.wrap(text, 22, width):
            assert mp.text_w(line, 22) <= width or " " not in line


def test_page_is_a1_portrait():
    assert (mp.W, mp.H) == (2245, 3178)  # 594 x 841 mm at 96 dpi
    assert "594mm 841mm" in mp.CSS
