from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.core.training.optim.router_fp32_master import RouterFp32Master  # noqa: E402


def test_fp32_master_accumulates_sub_bf16_ulp_and_roundtrips() -> None:
    working = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    manager = RouterFp32Master([("blocks.0.router.weight", working)])
    master = manager.substitute_named_params(
        [("blocks.0.router.weight", working)]
    )[0][1]
    assert master.dtype == torch.float32
    working.grad = torch.tensor([1.0e-3], dtype=torch.bfloat16)
    manager.sync_grads_to_master()
    with torch.no_grad():
        master.add_(master.grad, alpha=-1.0)
    state = manager.state_dict()
    assert float(state["blocks.0.router.weight"]) < 1.0

    restored_working = torch.nn.Parameter(
        torch.tensor([0.0], dtype=torch.bfloat16)
    )
    restored = RouterFp32Master(
        [("blocks.0.router.weight", restored_working)]
    )
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.state_dict()["blocks.0.router.weight"],
        state["blocks.0.router.weight"],
    )


def test_fp32_master_is_empty_without_trainable_router_parameters() -> None:
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    dense = torch.nn.Parameter(torch.ones(1))
    manager = RouterFp32Master(
        [("blocks.0.router.weight", frozen), ("blocks.0.attn.weight", dense)]
    )
    assert not manager
