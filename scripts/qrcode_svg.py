"""A small dependency-free QR code encoder: text in, SVG out.

Byte mode, error-correction level M, versions 1-6 (up to 106 bytes), which is plenty for a URL.
Follows ISO/IEC 18004; ``tests/test_qrcode_svg.py`` checks the Reed-Solomon code and the format
bits against published reference values. Written because the poster needs one QR code and a
package was not worth a download.

    python scripts/qrcode_svg.py https://example.org > code.svg
"""

from __future__ import annotations

import sys

# level M: version -> (error-correction codewords per block, blocks, data codewords per block)
_M = {
    1: (10, 1, 16),
    2: (16, 1, 28),
    3: (26, 1, 44),
    4: (18, 2, 32),
    5: (24, 2, 43),
    6: (16, 4, 27),
}
_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34]}

# GF(256) with the QR polynomial x^8 + x^4 + x^3 + x^2 + 1
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def rs_remainder(data: list[int], degree: int) -> list[int]:
    """The ``degree`` error-correction codewords for ``data`` (polynomial long division)."""
    gen = [1]
    for i in range(degree):  # multiply by (x + alpha^i)
        nxt = [0] * (len(gen) + 1)
        for j, c in enumerate(gen):
            nxt[j] ^= c
            nxt[j + 1] ^= _mul(c, _EXP[i])
        gen = nxt
    rem = [0] * degree
    for b in data:
        factor = b ^ rem[0]
        rem = rem[1:] + [0]
        for i in range(degree):
            rem[i] ^= _mul(gen[i + 1], factor)
    return rem


def format_bits(mask: int) -> int:
    """The 15 format bits for error-correction level M and the given mask (BCH(15,5) + XOR)."""
    data = mask  # level M is 0b00
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return ((data << 10) | rem) ^ 0x5412


def _pick_version(n: int) -> int:
    for v, (_, blocks, per) in _M.items():
        if 12 + 8 * n <= 8 * blocks * per:  # mode (4) + count (8) + payload
            return v
    raise ValueError(f"{n} bytes is too long for this encoder (limit 106)")


def codewords(payload: bytes, version: int) -> list[int]:
    """Data plus error correction, interleaved in transmission order."""
    ec, blocks, per = _M[version]
    bits: list[int] = []

    def put(val: int, n: int) -> None:
        bits.extend((val >> (n - 1 - i)) & 1 for i in range(n))

    put(0b0100, 4)
    put(len(payload), 8)
    for b in payload:
        put(b, 8)
    put(0, min(4, 8 * blocks * per - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    data = [int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8)]
    k = 0
    while len(data) < blocks * per:
        data.append(0xEC if k % 2 == 0 else 0x11)
        k += 1
    chunks = [data[i * per : (i + 1) * per] for i in range(blocks)]
    ecs = [rs_remainder(c, ec) for c in chunks]
    out = [c[j] for j in range(per) for c in chunks]
    return out + [e[j] for j in range(ec) for e in ecs]


class _Matrix:
    def __init__(self, version: int) -> None:
        self.n = 17 + 4 * version
        self.mod = [[False] * self.n for _ in range(self.n)]
        self.fn = [[False] * self.n for _ in range(self.n)]  # function (non-data) modules
        n = self.n
        for i in range(n):  # timing patterns
            self._set(6, i, i % 2 == 0)
            self._set(i, 6, i % 2 == 0)
        for cx, cy in ((3, 3), (n - 4, 3), (3, n - 4)):  # finder patterns + separators
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if 0 <= cx + dx < n and 0 <= cy + dy < n:
                        self._set(cx + dx, cy + dy, max(abs(dx), abs(dy)) not in (2, 4))
        pos = _ALIGN[version]
        for i, cy in enumerate(pos):
            for j, cx in enumerate(pos):
                if (i, j) in ((0, 0), (0, len(pos) - 1), (len(pos) - 1, 0)):
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        self._set(cx + dx, cy + dy, max(abs(dx), abs(dy)) != 1)
        self.draw_format(0)  # reserves the format areas; redrawn once the mask is chosen

    def _set(self, x: int, y: int, dark: bool) -> None:
        self.mod[y][x] = dark
        self.fn[y][x] = True

    def draw_format(self, mask: int) -> None:
        bits, n = format_bits(mask), self.n

        def bit(i: int) -> bool:
            return (bits >> i) & 1 == 1

        for i in range(6):
            self._set(8, i, bit(i))
        self._set(8, 7, bit(6))
        self._set(8, 8, bit(7))
        self._set(7, 8, bit(8))
        for i in range(9, 15):
            self._set(14 - i, 8, bit(i))
        for i in range(8):
            self._set(n - 1 - i, 8, bit(i))
        for i in range(8, 15):
            self._set(8, n - 15 + i, bit(i))
        self._set(8, n - 8, True)  # the always-dark module

    def place(self, data: list[int]) -> None:
        n, i = self.n, 0
        total = len(data) * 8
        for right in range(n - 1, 0, -2):
            if right <= 6:
                right -= 1
            for vert in range(n):
                for j in range(2):
                    x = right - j
                    y = n - 1 - vert if ((right + 1) & 2) == 0 else vert
                    if not self.fn[y][x] and i < total:
                        self.mod[y][x] = (data[i >> 3] >> (7 - (i & 7))) & 1 == 1
                        i += 1

    def mask(self, m: int) -> None:
        tests = (
            lambda x, y: (x + y) % 2 == 0,
            lambda x, y: y % 2 == 0,
            lambda x, y: x % 3 == 0,
            lambda x, y: (x + y) % 3 == 0,
            lambda x, y: (x // 3 + y // 2) % 2 == 0,
            lambda x, y: x * y % 2 + x * y % 3 == 0,
            lambda x, y: (x * y % 2 + x * y % 3) % 2 == 0,
            lambda x, y: ((x + y) % 2 + x * y % 3) % 2 == 0,
        )
        for y in range(self.n):
            for x in range(self.n):
                if not self.fn[y][x] and tests[m](x, y):
                    self.mod[y][x] = not self.mod[y][x]

    def penalty(self) -> int:
        n, mod, score = self.n, self.mod, 0
        lines = [row[:] for row in mod] + [[mod[y][x] for y in range(n)] for x in range(n)]
        finder_a = [True, False, True, True, True, False, True, False, False, False, False]
        finder_b = finder_a[::-1]
        for row in lines:
            run = 1
            for a, b in zip(row, row[1:], strict=False):
                if a == b:
                    run += 1
                else:
                    score += 3 + run - 5 if run >= 5 else 0
                    run = 1
            score += 3 + run - 5 if run >= 5 else 0
            for i in range(n - 10):
                if row[i : i + 11] in (finder_a, finder_b):
                    score += 40
        for y in range(n - 1):
            for x in range(n - 1):
                if mod[y][x] == mod[y][x + 1] == mod[y + 1][x] == mod[y + 1][x + 1]:
                    score += 3
        dark = sum(map(sum, mod))
        score += 10 * (abs(dark * 20 - n * n * 10) // (n * n))
        return score


def encode(text: str) -> list[list[bool]]:
    """The QR matrix (True = dark) for ``text``, quiet zone not included."""
    payload = text.encode("utf-8")
    version = _pick_version(len(payload))
    data = codewords(payload, version)
    best: tuple[int, _Matrix] | None = None
    for m in range(8):
        mx = _Matrix(version)
        mx.place(data)
        mx.mask(m)
        mx.draw_format(m)
        p = mx.penalty()
        if best is None or p < best[0]:
            best = (p, mx)
    assert best is not None
    return best[1].mod


def to_svg(text: str, quiet: int = 4, dark: str = "#000", light: str | None = "#fff") -> str:
    """An SVG (viewBox in modules, scales to any size) with the quiet zone included."""
    mat = encode(text)
    n = len(mat)
    size = n + 2 * quiet
    paths = []
    for y, row in enumerate(mat):
        x = 0
        while x < n:
            if row[x]:
                start = x
                while x < n and row[x]:
                    x += 1
                paths.append(f"M{start + quiet} {y + quiet}h{x - start}v1h-{x - start}z")
            else:
                x += 1
    bg = f'<rect width="{size}" height="{size}" fill="{light}"/>' if light else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'shape-rendering="crispEdges">{bg}<path d="{"".join(paths)}" fill="{dark}"/></svg>'
    )


if __name__ == "__main__":
    sys.stdout.write(to_svg(sys.argv[1] if len(sys.argv) > 1 else "https://example.org"))
