import pytest
import torch

from voiceguard.models.neural import NB_SAMP, build_cm

MODELS = ["rawnet2", "aasist", "aasist-l"]


@pytest.mark.parametrize("name", MODELS)
def test_forward_shape_and_score_range(name):
    torch.manual_seed(0)
    model = build_cm(name).eval()
    x = torch.randn(2, NB_SAMP)
    logits = model(x)
    assert logits.shape == (2, 2)
    p = model.score_bonafide(x)
    assert p.shape == (2,)
    assert torch.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("name", MODELS)
def test_accepts_unbatched(name):
    model = build_cm(name).eval()
    logits = model(torch.randn(NB_SAMP))
    assert logits.shape == (1, 2)


def test_param_counts_in_expected_range():
    # AASIST-L is the ~85k-param lightweight variant; AASIST ~300k; RawNet2 ~17M
    assert build_cm("aasist-l").param_count() < 150_000
    assert build_cm("aasist").param_count() < 500_000
    assert 10_000_000 < build_cm("rawnet2").param_count() < 30_000_000


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_cm("nope")
