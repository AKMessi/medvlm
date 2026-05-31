import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("einops")

from medvlm.config import MedVLMConfig
from medvlm.model import MedVLM


def test_tiny_hybrid_forward_pass():
    config = MedVLMConfig(
        img_size=64,
        max_len=16,
        encoder_type="hybrid",
        pretrained_backbone=False,
        d_model=64,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = MedVLM(config, vocab_size=128, bos_id=125, eos_id=126, pad_id=127)
    images = torch.randn(1, 1, 64, 64)
    tokens = torch.randint(0, 100, (1, 8))
    logits = model(tokens, images)
    assert logits.shape == (1, 8, 128)
