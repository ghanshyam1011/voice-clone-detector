"""The poster's QR encoder, checked against published reference values."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qrcode_svg.py"
_spec = importlib.util.spec_from_file_location("qrcode_svg", _PATH)
qr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qr)

URL = "https://github.com/ghanshyam1011/voice-clone-detector"


def test_reed_solomon_matches_the_published_example():
    # the worked "HELLO WORLD" example for version 1-M in the QR tutorials / ISO 18004 annex
    data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17]
    assert qr.rs_remainder(data, 10) == [196, 35, 39, 119, 235, 215, 231, 226, 93, 23]


@pytest.mark.parametrize(
    ("mask", "bits"),
    [
        (0, "101010000010010"),
        (1, "101000100100101"),
        (2, "101111001111100"),
        (3, "101101101001011"),
        (4, "100010111111001"),
        (5, "100000011001110"),
        (6, "100111110010111"),
        (7, "100101010100000"),
    ],
)
def test_format_bits_for_level_m(mask, bits):
    assert format(qr.format_bits(mask), "015b") == bits


def _syndromes_are_zero(block: list[int], n_ec: int) -> bool:
    for i in range(n_ec):
        v = 0
        for c in block:
            v = qr._mul(v, qr._EXP[i]) ^ c
        if v:
            return False
    return True


def test_every_block_of_the_codeword_stream_is_a_valid_reed_solomon_word():
    payload = URL.encode()
    version = qr._pick_version(len(payload))
    assert version == 4
    ec, blocks, per = qr._M[version]
    stream = qr.codewords(payload, version)
    assert len(stream) == blocks * (per + ec)
    data_part, ec_part = stream[: blocks * per], stream[blocks * per :]
    for b in range(blocks):
        block = data_part[b::blocks] + ec_part[b::blocks]
        assert _syndromes_are_zero(block, ec)


def test_data_codewords_start_with_the_byte_mode_header():
    # mode 0100, length 00000010, "A" 01000001, "B" 01000010, terminator 0000, then pad bytes
    stream = qr.codewords(b"AB", 1)
    assert stream[:6] == [0x40, 0x24, 0x14, 0x20, 0xEC, 0x11]


def test_matrix_structure():
    mat = qr.encode(URL)
    n = len(mat)
    assert n == 33  # version 4
    for cx, cy in ((3, 3), (n - 4, 3), (3, n - 4)):  # finder patterns
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                assert mat[cy + dy][cx + dx] == (max(abs(dx), abs(dy)) != 2)
    assert all(mat[6][x] == (x % 2 == 0) for x in range(8, n - 8))  # timing pattern
    assert all(mat[y][6] == (y % 2 == 0) for y in range(8, n - 8))
    assert mat[n - 8][8] is True  # the always-dark module


def _read_back(mat: list[list[bool]]) -> bytes:
    """A minimal reader: format bits -> mask, undo the mask, walk the zigzag, de-interleave."""
    n = len(mat)
    version = (n - 17) // 4
    fb = [mat[i][8] for i in range(6)] + [mat[7][8], mat[8][8], mat[8][7]]
    fb += [mat[8][14 - i] for i in range(9, 15)]
    bits = sum(int(v) << i for i, v in enumerate(fb))
    mask = next(m for m in range(8) if qr.format_bits(m) == bits)
    template = qr._Matrix(version)
    template.mod = [row[:] for row in mat]
    template.mask(mask)  # XOR again: removes the mask from the data modules
    raw = []
    for right in range(n - 1, 0, -2):
        if right <= 6:
            right -= 1
        for vert in range(n):
            for j in range(2):
                x = right - j
                y = n - 1 - vert if ((right + 1) & 2) == 0 else vert
                if not template.fn[y][x]:
                    raw.append(template.mod[y][x])
    cw = [
        int("".join("1" if b else "0" for b in raw[k : k + 8]), 2)
        for k in range(0, len(raw) - 7, 8)
    ]
    ec, blocks, per = qr._M[version]
    data = [cw[j * blocks + b] for b in range(blocks) for j in range(per)]
    bits_s = "".join(f"{c:08b}" for c in data)
    assert bits_s[:4] == "0100"
    count = int(bits_s[4:12], 2)
    return bytes(int(bits_s[12 + 8 * k : 20 + 8 * k], 2) for k in range(count))


@pytest.mark.parametrize("text", [URL, "hi", "https://example.org/a?b=c&d=é"])
def test_a_reader_gets_the_text_back(text):
    assert _read_back(qr.encode(text)) == text.encode()


def test_svg_has_a_quiet_zone_and_scales():
    svg = qr.to_svg(URL)
    assert 'viewBox="0 0 41 41"' in svg  # 33 modules + 4 + 4
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_too_long_text_is_refused():
    with pytest.raises(ValueError):
        qr.encode("x" * 200)
