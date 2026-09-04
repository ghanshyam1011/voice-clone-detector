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


def _wav2vec2_base_cached() -> bool:
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    return (hub / "models--facebook--wav2vec2-base").exists()


@pytest.mark.skipif(not _wav2vec2_base_cached(), reason="facebook/wav2vec2-base not in HF cache")
def test_ssl_aasist_frozen_backbone_and_forward():
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    model = build_cm("ssl-aasist").eval()
    # ~94M total, only the AASIST backend (<1M) trainable
    assert model.param_count() > 90_000_000
    assert model.param_count(trainable_only=True) < 2_000_000

    x = torch.randn(2, NB_SAMP)
    assert model(x).shape == (2, 2)

    model.train()
    torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1])).backward()
    backbone_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "ssl.model" in n and p.grad is not None
    )
    assert backbone_grad == 0.0  # backbone stays frozen
    assert model.backbone.LL.weight.grad is not None  # backend trains
