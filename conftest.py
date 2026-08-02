"""Shared fixtures and testbed registration for repository contracts."""

from __future__ import annotations

import importlib

import pytest

# Register the CPU-only verification provider explicitly for the test process.
# Production composition intentionally exposes only release-supported families.
import mirai.core.models.testbed.sparse_moe  # noqa: F401

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if torch is not None:
    try:
        # Some PyTorch builds lazily expose this module while torch.optim imports
        # Dynamo. Establishing the public submodule first makes probe order neutral.
        importlib.import_module("torch._inductor.custom_graph_pass")
    except (AttributeError, ModuleNotFoundError):
        pass


@pytest.fixture
def tiny_moe_latents():
    if torch is None:  # pragma: no cover
        pytest.skip("torch not installed")
    g = torch.Generator().manual_seed(7)
    return torch.randn((2, 16, 4, 8, 8), generator=g, dtype=torch.float32)


@pytest.fixture
def dummy_captions():
    return [
        "a person walking in the rain",
        "a neon-lit city street at night",
    ]


@pytest.fixture
def four_frame_video_tensor():
    if torch is None:  # pragma: no cover
        pytest.skip("torch not installed")
    g = torch.Generator().manual_seed(17)
    return torch.randn((3, 4, 16, 16), generator=g, dtype=torch.float32)
