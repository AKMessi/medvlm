from medvlm.config import MedVLMConfig


def test_hybrid_defaults_resolve_dimensions():
    config = MedVLMConfig()
    assert config.encoder_type == "hybrid"
    assert config.resolved_d_model() == 512
    assert config.resolved_n_heads() == 8
