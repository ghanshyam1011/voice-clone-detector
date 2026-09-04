"""Models.

* baseline_gbm  -- handcrafted "Generation 1" baseline (comparison row only)
* neural        -- RawNet2 / AASIST / AASIST-L, vendored from clovaai/aasist
* ssl           -- (P1, next) frozen wav2vec2/XLS-R front-end for SSL-AASIST
"""

from voiceguard.models.baseline_gbm import (
    BASELINE_MODELS,
    score_bonafide,
    train_baseline_ensemble,
)
from voiceguard.models.neural import NB_SAMP, build_cm

__all__ = [
    "train_baseline_ensemble",
    "score_bonafide",
    "BASELINE_MODELS",
    "build_cm",
    "NB_SAMP",
]
