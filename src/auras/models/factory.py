"""Model registry and factory.

Provides a single entry-point ``create_model(cfg)`` that instantiates
any architecture by name using the config from ``configs/model/*.yaml``.
"""

from __future__ import annotations

from typing import Dict, Type

from auras.models.base import BaseSeizureModel

# ── Registry ─────────────────────────────────────────────────────
_REGISTRY: Dict[str, Type[BaseSeizureModel]] = {}


def register(name: str):
    """Decorator to register a model class."""

    def wrapper(cls: Type[BaseSeizureModel]):
        _REGISTRY[name] = cls
        return cls

    return wrapper


def list_models():
    """Return list of registered model names."""
    return list(_REGISTRY.keys())


def create_model(cfg, num_channels: int = 4) -> BaseSeizureModel:
    """Instantiate a model from a config object (OmegaConf or dict).

    Parameters
    ----------
    cfg : OmegaConf / dict
        Must contain ``arch`` key matching a registered name.
    num_channels : int
        Number of EEG input channels.
    """
    from omegaconf import OmegaConf

    if hasattr(cfg, "arch"):
        arch = cfg.arch
        params = OmegaConf.to_container(cfg, resolve=True)
    else:
        params = dict(cfg)
        arch = params.pop("arch")

    params.pop("name", None)
    params["num_channels"] = num_channels

    if arch not in _REGISTRY:
        raise ValueError(f"Unknown model '{arch}'. Available: {list_models()}")

    return _REGISTRY[arch](**params)


# ── Lazy registration (import triggers decorator) ────────────────
from auras.models.lstm import LSTMModel  # noqa: E402
from auras.models.bilstm import BiLSTMModel  # noqa: E402
from auras.models.resnet1d import ResNet1D  # noqa: E402
from auras.models.mobilenetv3_1d import MobileNetV3_1D  # noqa: E402
from auras.models.ghostnet1d import GhostNet1D  # noqa: E402
from auras.models.mobilevit_1d import MobileViT_1D  # noqa: E402
from auras.models.autoformer import Autoformer  # noqa: E402

register("lstm")(LSTMModel)
register("bilstm")(BiLSTMModel)
register("resnet1d")(ResNet1D)
register("mobilenetv3_1d")(MobileNetV3_1D)
register("ghostnet1d")(GhostNet1D)
register("mobilevit_1d")(MobileViT_1D)
register("autoformer")(Autoformer)

# ── Sprint 2 architectures ────────────────────────────────────────
from auras.models.cnn_baseline import CNNBaseline  # noqa: E402
from auras.models.cnn_bilstm import CNNBiLSTM  # noqa: E402
from auras.models.cnn_bilstm_attn import CNNBiLSTMAttn  # noqa: E402
from auras.models.cam_cnn_bilstm import CAMCNNBiLSTM  # noqa: E402
from auras.models.eegformer import EEGformer  # noqa: E402
from auras.models.cnn_informer import CNNInformer  # noqa: E402
from auras.models.ultralight_cnn import UltraLightCNN  # noqa: E402
from auras.models.pyramidal_cnn_bilstm import PyramidalCNNBiLSTM  # noqa: E402

register("cnn_baseline")(CNNBaseline)
register("cnn_bilstm")(CNNBiLSTM)
register("cnn_bilstm_attn")(CNNBiLSTMAttn)
register("cam_cnn_bilstm")(CAMCNNBiLSTM)
register("eegformer")(EEGformer)
register("cnn_informer")(CNNInformer)
register("ultralight_cnn")(UltraLightCNN)
register("pyramidal_cnn_bilstm")(PyramidalCNNBiLSTM)
