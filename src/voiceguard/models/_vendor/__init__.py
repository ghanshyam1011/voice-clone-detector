"""Vendored reference implementations. Do not edit -- keep faithful to
upstream so we can reproduce published behaviour before changing anything.
Linting/formatting is disabled for this directory (see pyproject.toml).

| file            | source                                                        | licence |
|-----------------|---------------------------------------------------------------|---------|
| aasist_ref.py   | github.com/clovaai/aasist  models/AASIST.py                    | MIT     |
| rawnet2_ref.py  | github.com/clovaai/aasist  models/RawNet2Spoof.py              | MIT     |
| rawboost_ref.py | github.com/TakHemlata/RawBoost-antispoofing  RawBoost.py       | MIT     |
| AASIST*.conf    | github.com/clovaai/aasist  config/                             | MIT     |

Fetched 2026-09-04. `models/__init__.py` wraps these with the repo's
class-0 = bonafide convention and config plumbing.
"""
