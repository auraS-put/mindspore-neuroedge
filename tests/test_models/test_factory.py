"""Tests for model factory and registration."""

from auras.models.factory import list_models, create_model
from omegaconf import OmegaConf


def test_all_models_registered():
    models = list_models()
    expected = ["lstm", "bilstm", "resnet1d", "mobilenetv3_1d", "ghostnet1d", "mobilevit_1d", "autoformer"]
    for name in expected:
        assert name in models, f"{name} not registered"


def test_create_model_from_config():
    cfg = OmegaConf.create({"name": "lstm", "arch": "lstm", "hidden_size": 64, "num_layers": 1, "dropout": 0.1, "bidirectional": False, "num_classes": 2})
    model = create_model(cfg, num_channels=4)
    assert model.num_classes == 2
    assert model.count_params() > 0
