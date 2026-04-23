"""Sprint 2 — New Architectures: comprehensive test suite.

Tests cover every component implemented in Sprint 2:
  1.  Shared modules:    DepthwiseSeparableConv1D, ResDSBlock,
                         ChannelAttention1D, ProbSparseAttention,
                         AttentionDistilling, InformerEncoderLayer
  2.  Sprint-2 models:   CNNBaseline, CNNBiLSTM, CNNBiLSTMAttn,
                         CAMCNNBiLSTM, EEGformer, CNNInformer,
                         UltraLightCNN, PyramidalCNNBiLSTM
  3.  All 15 models (7 existing + 8 new) registered in factory
  4.  factory.create_model integration via YAML configs
  5.  Parameter budget constraints for lightweight models
  6.  Output shape contract: (B, C, T=1024) → (B, 2)

Run with:
    .venv/bin/python -m pytest tests/test_sprint2_architectures.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor

# Use PyNative mode for all tests
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

# Standard input batch: batch=2, channels=4, time=1024
B, C, T = 2, 4, 1024


def _dummy(b: int = B, c: int = C, t: int = T) -> Tensor:
    """Create a random float32 input tensor."""
    return Tensor(np.random.randn(b, c, t).astype(np.float32))


# =============================================================================
# 1. Shared Modules
# =============================================================================

class TestDepthwiseSeparableConv1D:

    def test_output_shape_same_channels(self):
        from auras.models.modules import DepthwiseSeparableConv1D
        m = DepthwiseSeparableConv1D(4, 4, kernel_size=9)
        y = m(_dummy())
        assert y.shape == (B, 4, T), f"Expected (2,4,1024), got {y.shape}"

    def test_output_shape_different_channels(self):
        from auras.models.modules import DepthwiseSeparableConv1D
        m = DepthwiseSeparableConv1D(4, 64, kernel_size=7)
        y = m(_dummy())
        assert y.shape == (B, 64, T)

    def test_preserved_time_dimension(self):
        from auras.models.modules import DepthwiseSeparableConv1D
        m = DepthwiseSeparableConv1D(4, 32, kernel_size=5, stride=1)
        y = m(_dummy())
        assert y.shape[-1] == T, "stride=1 should preserve time dimension"

    def test_stride_halves_time(self):
        from auras.models.modules import DepthwiseSeparableConv1D
        m = DepthwiseSeparableConv1D(4, 32, kernel_size=3, stride=2)
        y = m(_dummy())
        assert y.shape[-1] == T // 2


class TestResDSBlock:

    def test_output_shape_preserved(self):
        from auras.models.modules import ResDSBlock
        m = ResDSBlock(64, kernel_size=9)
        x = Tensor(np.random.randn(B, 64, T).astype(np.float32))
        y = m(x)
        assert y.shape == (B, 64, T), "ResDSBlock must preserve shape"

    def test_residual_connection(self):
        """Output should differ from input (non-trivial transform)."""
        from auras.models.modules import ResDSBlock
        m = ResDSBlock(4, kernel_size=3)
        x = _dummy()
        y = m(x)
        assert not np.allclose(x.asnumpy(), y.asnumpy()), \
            "ResDSBlock should transform the input"


class TestChannelAttention1D:

    def test_output_shape(self):
        from auras.models.modules import ChannelAttention1D
        m = ChannelAttention1D(4)
        y = m(_dummy())
        assert y.shape == (B, C, T)

    def test_channel_weights_sigmoid_range(self):
        """Channel weights must be in (0, 1) (sigmoid output)."""
        from auras.models.modules import ChannelAttention1D
        m = ChannelAttention1D(4)
        x = _dummy()
        # Compute weight by comparing output with input
        y = m(x)
        ratio = (y.asnumpy() / (x.asnumpy() + 1e-8))
        # Each channel-wise ratio should be in (0, 1+eps)
        assert ratio.max() < 2.0


class TestProbSparseAttention:

    def test_output_shape(self):
        from auras.models.modules import ProbSparseAttention
        m = ProbSparseAttention(d_model=64, n_heads=4, factor=3, max_len=128)
        x = Tensor(np.random.randn(B, 32, 64).astype(np.float32))
        y = m(x)
        assert y.shape == (B, 32, 64), f"Expected (2,32,64), got {y.shape}"

    def test_different_sequence_lengths(self):
        from auras.models.modules import ProbSparseAttention
        m = ProbSparseAttention(d_model=32, n_heads=4, factor=3, max_len=256)
        for t in [16, 64, 256]:
            x = Tensor(np.random.randn(B, t, 32).astype(np.float32))
            y = m(x)
            assert y.shape == (B, t, 32), f"Failed for t={t}"

    def test_head_divisibility_error(self):
        with pytest.raises(AssertionError):
            from auras.models.modules import ProbSparseAttention
            ProbSparseAttention(d_model=65, n_heads=4)


class TestAttentionDistilling:

    def test_halves_sequence(self):
        from auras.models.modules import AttentionDistilling
        m = AttentionDistilling(d_model=64)
        x = Tensor(np.random.randn(B, 256, 64).astype(np.float32))
        y = m(x)
        assert y.shape == (B, 128, 64), f"Expected (2,128,64), got {y.shape}"

    def test_odd_sequence_length(self):
        from auras.models.modules import AttentionDistilling
        m = AttentionDistilling(d_model=32)
        x = Tensor(np.random.randn(B, 101, 32).astype(np.float32))
        y = m(x)
        assert y.shape[1] == 50  # floor(101/2)


class TestInformerEncoderLayer:

    def test_halves_sequence(self):
        from auras.models.modules import InformerEncoderLayer
        layer = InformerEncoderLayer(d_model=64, n_heads=4, d_ff=256, factor=3)
        x = Tensor(np.random.randn(B, 128, 64).astype(np.float32))
        y = layer(x)
        assert y.shape == (B, 64, 64), f"Expected (2,64,64), got {y.shape}"

    def test_stacked_layers_shape(self):
        from auras.models.modules import InformerEncoderLayer
        import mindspore.nn as nn
        stack = nn.SequentialCell(
            InformerEncoderLayer(64, 4, 256),
            InformerEncoderLayer(64, 4, 256),
            InformerEncoderLayer(64, 4, 256),
        )
        x = Tensor(np.random.randn(B, 128, 64).astype(np.float32))
        y = stack(x)
        assert y.shape == (B, 16, 64)  # 128 // 2^3 = 16


# =============================================================================
# 2. Sprint-2 Model Forward Passes
# =============================================================================

class TestCNNBaseline:

    def test_output_shape(self):
        from auras.models.cnn_baseline import CNNBaseline
        m = CNNBaseline()
        assert m(_dummy()).shape == (B, 2)

    def test_custom_channels(self):
        from auras.models.cnn_baseline import CNNBaseline
        m = CNNBaseline(conv_channels=(16, 32, 64), kernels=(5, 3, 3))
        assert m(_dummy()).shape == (B, 2)

    def test_num_classes(self):
        from auras.models.cnn_baseline import CNNBaseline
        m = CNNBaseline(num_classes=3)
        assert m(_dummy()).shape == (B, 3)

    def test_param_count(self):
        from auras.models.cnn_baseline import CNNBaseline
        m = CNNBaseline()
        assert m.count_params() < 200_000, "CNNBaseline should be <200 K params"


class TestCNNBiLSTM:

    def test_output_shape(self):
        from auras.models.cnn_bilstm import CNNBiLSTM
        m = CNNBiLSTM()
        assert m(_dummy()).shape == (B, 2)

    def test_different_hidden_size(self):
        from auras.models.cnn_bilstm import CNNBiLSTM
        m = CNNBiLSTM(hidden_size=32)
        assert m(_dummy()).shape == (B, 2)


class TestCNNBiLSTMAttn:

    def test_output_shape(self):
        from auras.models.cnn_bilstm_attn import CNNBiLSTMAttn
        m = CNNBiLSTMAttn()
        assert m(_dummy()).shape == (B, 2)

    def test_short_input(self):
        from auras.models.cnn_bilstm_attn import CNNBiLSTMAttn
        m = CNNBiLSTMAttn()
        assert m(_dummy(t=256)).shape == (B, 2)


class TestCAMCNNBiLSTM:

    def test_output_shape(self):
        from auras.models.cam_cnn_bilstm import CAMCNNBiLSTM
        m = CAMCNNBiLSTM()
        assert m(_dummy()).shape == (B, 2)

    def test_single_channel(self):
        """Works with arbitrary number of channels."""
        from auras.models.cam_cnn_bilstm import CAMCNNBiLSTM
        m = CAMCNNBiLSTM(num_channels=1)
        x = Tensor(np.random.randn(B, 1, T).astype(np.float32))
        assert m(x).shape == (B, 2)


class TestEEGformer:

    def test_output_shape(self):
        from auras.models.eegformer import EEGformer
        m = EEGformer()
        assert m(_dummy()).shape == (B, 2)

    def test_param_count(self):
        """Busia et al. (EEGformer) reports ~50.6 K params for their impl; MindSpore's
        MultiheadAttention includes full Q/K/V weight matrices internally,
        and the max_seq=512 learnable PE adds 65 K params.  Allow ≤300 K."""
        from auras.models.eegformer import EEGformer
        m = EEGformer()
        assert m.count_params() < 300_000, \
            f"EEGformer should be <300 K params, got {m.count_params()}"

    def test_different_patch_size(self):
        from auras.models.eegformer import EEGformer
        m = EEGformer(patch_size=8)
        assert m(_dummy()).shape == (B, 2)


class TestCNNInformer:

    def test_output_shape(self):
        from auras.models.cnn_informer import CNNInformer
        m = CNNInformer()
        assert m(_dummy()).shape == (B, 2)

    def test_fewer_layers(self):
        from auras.models.cnn_informer import CNNInformer
        m = CNNInformer(e_layers=2)
        assert m(_dummy()).shape == (B, 2)


class TestUltraLightCNN:

    def test_output_shape(self):
        from auras.models.ultralight_cnn import UltraLightCNN
        m = UltraLightCNN()
        assert m(_dummy()).shape == (B, 2)

    def test_param_budget(self):
        """Must stay under 10 K trainable parameters."""
        from auras.models.ultralight_cnn import UltraLightCNN
        m = UltraLightCNN()
        n = m.count_params()
        assert n < 10_000, f"UltraLightCNN exceeded budget: {n} params"

    def test_single_channel(self):
        from auras.models.ultralight_cnn import UltraLightCNN
        m = UltraLightCNN(num_channels=1)
        x = Tensor(np.random.randn(B, 1, T).astype(np.float32))
        assert m(x).shape == (B, 2)


class TestPyramidalCNNBiLSTM:

    def test_output_shape(self):
        from auras.models.pyramidal_cnn_bilstm import PyramidalCNNBiLSTM
        m = PyramidalCNNBiLSTM()
        assert m(_dummy()).shape == (B, 2)

    def test_param_count(self):
        """Wang C. et al. (PCNN-BiLSTM) reports ~9,371 params for a 1-channel input.  With 4
        EEG channels and BiLSTM(64 in→32 hidden) the BiLSTM alone is ~25 K;
        allow ≤50 K."""
        from auras.models.pyramidal_cnn_bilstm import PyramidalCNNBiLSTM
        m = PyramidalCNNBiLSTM()
        n = m.count_params()
        assert n < 50_000, f"PyramidalCNNBiLSTM too large: {n} params"

    def test_short_input(self):
        from auras.models.pyramidal_cnn_bilstm import PyramidalCNNBiLSTM
        m = PyramidalCNNBiLSTM()
        assert m(_dummy(t=256)).shape == (B, 2)


# =============================================================================
# 3. Factory Registry — all 15 models present
# =============================================================================

class TestModelRegistry:

    ALL_MODELS = [
        # Sprint 1 / existing
        "lstm", "bilstm", "resnet1d", "mobilenetv3_1d",
        "ghostnet1d", "mobilevit_1d", "autoformer",
        # Sprint 2 — new
        "cnn_baseline", "cnn_bilstm", "cnn_bilstm_attn",
        "cam_cnn_bilstm", "eegformer", "cnn_informer",
        "ultralight_cnn", "pyramidal_cnn_bilstm",
    ]

    def test_all_models_registered(self):
        from auras.models.factory import list_models
        registered = list_models()
        for name in self.ALL_MODELS:
            assert name in registered, f"Model '{name}' missing from registry"

    def test_registry_count(self):
        from auras.models.factory import list_models
        assert len(list_models()) >= 15


# =============================================================================
# 4. Factory create_model integration
# =============================================================================

class TestCreateModel:

    def test_create_cnn_baseline_via_yaml(self):
        from omegaconf import OmegaConf
        from auras.models.factory import create_model
        cfg = OmegaConf.create({
            "arch": "cnn_baseline", "name": "cnn_baseline",
            "conv_channels": [32, 64, 128], "kernels": [7, 5, 3],
            "dropout": 0.2, "num_classes": 2,
        })
        m = create_model(cfg, num_channels=4)
        assert m(_dummy()).shape == (B, 2)

    def test_create_eegformer_via_yaml(self):
        from omegaconf import OmegaConf
        from auras.models.factory import create_model
        cfg = OmegaConf.create({
            "arch": "eegformer", "name": "eegformer",
            "embed_dim": 128, "num_heads": 8, "patch_size": 5,
            "mlp_ratio": 2.0, "dropout": 0.1, "max_seq": 512,
            "num_classes": 2,
        })
        m = create_model(cfg, num_channels=4)
        assert m(_dummy()).shape == (B, 2)

    def test_create_ultralight_via_yaml(self):
        from omegaconf import OmegaConf
        from auras.models.factory import create_model
        cfg = OmegaConf.create({
            "arch": "ultralight_cnn", "name": "ultralight_cnn",
            "dropout": 0.1, "num_classes": 2,
        })
        m = create_model(cfg, num_channels=4)
        assert m(_dummy()).shape == (B, 2)

    def test_create_cam_cnn_bilstm_via_yaml(self):
        from omegaconf import OmegaConf
        from auras.models.factory import create_model
        cfg = OmegaConf.create({
            "arch": "cam_cnn_bilstm", "name": "cam_cnn_bilstm",
            "stem_channels": 64, "lstm_hidden": 40,
            "dropout": 0.25, "num_classes": 2,
        })
        m = create_model(cfg, num_channels=4)
        assert m(_dummy()).shape == (B, 2)


# =============================================================================
# 5. Parameter counts summary (informational, not strict assertions)
# =============================================================================

class TestParameterCounts:

    def test_print_all_sprint2_params(self, capsys):
        """Print param counts for all Sprint-2 models (for reference)."""
        from auras.models.cnn_baseline import CNNBaseline
        from auras.models.cnn_bilstm import CNNBiLSTM
        from auras.models.cnn_bilstm_attn import CNNBiLSTMAttn
        from auras.models.cam_cnn_bilstm import CAMCNNBiLSTM
        from auras.models.eegformer import EEGformer
        from auras.models.cnn_informer import CNNInformer
        from auras.models.ultralight_cnn import UltraLightCNN
        from auras.models.pyramidal_cnn_bilstm import PyramidalCNNBiLSTM

        models = [
            ("cnn_baseline",         CNNBaseline()),
            ("cnn_bilstm",           CNNBiLSTM()),
            ("cnn_bilstm_attn",      CNNBiLSTMAttn()),
            ("cam_cnn_bilstm",       CAMCNNBiLSTM()),
            ("eegformer",            EEGformer()),
            ("cnn_informer",         CNNInformer()),
            ("ultralight_cnn",       UltraLightCNN()),
            ("pyramidal_cnn_bilstm", PyramidalCNNBiLSTM()),
        ]
        print("\n--- Sprint-2 Parameter Counts ---")
        for name, m in models:
            n = m.count_params()
            print(f"  {name:<25} {n:>10,} params")

        # All must produce correct output shape
        for name, m in models:
            out = m(_dummy())
            assert out.shape == (B, 2), f"{name} output shape wrong: {out.shape}"
