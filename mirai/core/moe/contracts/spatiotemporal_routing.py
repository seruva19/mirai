from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.core.moe.adaptation.spatiotemporal import (  # noqa: E402
    sampled_spatiotemporal_edges,
    spatiotemporal_routing_consistency_loss,
)


def test_sampled_edges_are_adjacent_and_bounded() -> None:
    source, target = sampled_spatiotemporal_edges(
        (3, 4, 5), max_edges=17, device="cpu"
    )
    assert source.numel() == target.numel() == 17
    delta = target - source
    assert set(delta.tolist()) <= {1, 5, 20}


def test_consistency_ignores_text_spans_and_has_gradients() -> None:
    first_video = torch.full((8, 3), 1.0 / 3.0)
    first_text = torch.randn(2, 3)
    second_video = torch.full((8, 3), 1.0 / 3.0)
    second_text = torch.randn(1, 3)
    probabilities = torch.cat(
        [first_video, first_text, second_video, second_text]
    ).requires_grad_(True)
    loss = spatiotemporal_routing_consistency_loss(
        probabilities,
        video_offsets=(0, 10),
        grid=(2, 2, 2),
        max_edges=5,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert probabilities.grad is not None
    assert probabilities.grad[8:10].count_nonzero() == 0
    assert probabilities.grad[18:].count_nonzero() == 0


def test_consistency_penalizes_adjacent_router_disagreement() -> None:
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
    )
    loss = spatiotemporal_routing_consistency_loss(
        probabilities,
        video_offsets=(0,),
        grid=(1, 2, 2),
        max_edges=4,
    )
    assert float(loss) > 0.0
