"""Reference diversity-aware expert subset routing from co-occurrence covariance.

Conceptual source: https://openreview.net/forum?id=7FsbfQgti4 (MP-MoE).
"""

from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DIVERSITY_ROUTING_STATE_SCHEMA_VERSION = 1


class ExpertCooccurrenceCovariance:
    """Accumulate exact first/second moments of top-k expert co-selection."""

    def __init__(self, num_experts: int) -> None:
        if int(num_experts) < 2:
            raise ValueError("Diversity-aware routing requires at least two experts.")
        self.num_experts = int(num_experts)
        self.samples = 0
        self._sum = torch.zeros(self.num_experts, dtype=torch.float64)
        self._cross = torch.zeros(
            self.num_experts, self.num_experts, dtype=torch.float64
        )

    def update(self, selected_experts: Any) -> None:
        if not torch.is_tensor(selected_experts) or selected_experts.ndim != 2:
            raise ValueError("selected_experts must have shape [tokens, top_k].")
        indices = selected_experts.detach().long().cpu()
        if int(indices.shape[1]) < 2:
            raise ValueError("Co-occurrence covariance requires top_k >= 2.")
        if bool((indices < 0).any()) or bool((indices >= self.num_experts).any()):
            raise ValueError("selected_experts contains an out-of-range expert id.")
        indicators = torch.zeros(
            int(indices.shape[0]), self.num_experts, dtype=torch.float64
        )
        indicators.scatter_(1, indices, 1.0)
        if bool((indicators.sum(dim=1) != int(indices.shape[1])).any()):
            raise ValueError("selected_experts contains a duplicate expert for one token.")
        self.samples += int(indicators.shape[0])
        self._sum += indicators.sum(dim=0)
        self._cross += indicators.T @ indicators

    def covariance(self) -> Any:
        if self.samples == 0:
            raise ValueError("Co-occurrence covariance has no observations.")
        mean = self._sum / float(self.samples)
        return self._cross / float(self.samples) - torch.outer(mean, mean)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIVERSITY_ROUTING_STATE_SCHEMA_VERSION,
            "num_experts": self.num_experts,
            "samples": self.samples,
            "sum": self._sum.clone(),
            "cross": self._cross.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != DIVERSITY_ROUTING_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported diversity-routing state schema.")
        if int(state.get("num_experts", 0)) != self.num_experts:
            raise ValueError("Diversity-routing checkpoint expert topology changed.")
        samples = int(state.get("samples", -1))
        total = torch.as_tensor(state.get("sum"), dtype=torch.float64).cpu()
        cross = torch.as_tensor(state.get("cross"), dtype=torch.float64).cpu()
        if samples < 0 or total.shape != (self.num_experts,) or cross.shape != (
            self.num_experts,
            self.num_experts,
        ):
            raise ValueError("Invalid diversity-routing checkpoint moments.")
        if not torch.isfinite(total).all() or not torch.isfinite(cross).all():
            raise ValueError("Diversity-routing checkpoint moments must be finite.")
        self.samples, self._sum, self._cross = samples, total.clone(), cross.clone()


class DiversityAwareRoutingController:
    """Warm up on native routes, then replay a frozen covariance selector."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        warmup_steps: int,
        ridge: float = 1e-4,
        token_chunk_size: int = 2048,
    ) -> None:
        if int(num_experts) < 2:
            raise ValueError("Diversity-aware routing requires at least two experts.")
        if int(top_k) < 2 or int(top_k) > int(num_experts):
            raise ValueError("Diversity-aware routing requires 2 <= top_k <= experts.")
        if int(warmup_steps) <= 0:
            raise ValueError("Diversity-aware routing warmup_steps must be positive.")
        if float(ridge) <= 0.0:
            raise ValueError("Diversity-aware routing ridge must be positive.")
        if int(token_chunk_size) <= 0:
            raise ValueError("Diversity-aware routing token_chunk_size must be positive.")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.warmup_steps = int(warmup_steps)
        self.ridge = float(ridge)
        self.token_chunk_size = int(token_chunk_size)
        self._step = 0
        self._training = False
        self._trackers: dict[str, ExpertCooccurrenceCovariance] = {}

    def bind_step(self, *, step: int, training: bool) -> None:
        if int(step) < 0:
            raise ValueError("Diversity-aware routing step must be non-negative.")
        self._step = int(step)
        self._training = bool(training)

    def select(
        self,
        layer_name: str,
        scores: Any,
        native_top_indices: Any,
        *,
        training: bool,
    ) -> Any:
        if not self._training or not bool(training):
            return native_top_indices
        if tuple(scores.shape)[:-1] != tuple(native_top_indices.shape)[:-1]:
            raise ValueError("Diversity routing scores and native routes disagree.")
        if int(scores.shape[-1]) != self.num_experts:
            raise ValueError("Diversity routing expert topology changed.")
        if int(native_top_indices.shape[-1]) != self.top_k:
            raise ValueError("Diversity routing top-k topology changed.")
        name = str(layer_name).strip()
        if not name:
            raise ValueError("Diversity routing requires a stable layer name.")
        tracker = self._trackers.setdefault(
            name, ExpertCooccurrenceCovariance(self.num_experts)
        )
        if self._step < self.warmup_steps:
            tracker.update(native_top_indices)
            return native_top_indices
        if tracker.samples == 0:
            raise RuntimeError(
                f"Diversity routing layer '{name}' reached replay before warm-up observations."
            )
        return diversity_aware_batched_topk(
            scores,
            tracker.covariance(),
            top_k=self.top_k,
            ridge=self.ridge,
            token_chunk_size=self.token_chunk_size,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIVERSITY_ROUTING_STATE_SCHEMA_VERSION,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "warmup_steps": self.warmup_steps,
            "ridge": self.ridge,
            "token_chunk_size": self.token_chunk_size,
            "step": self._step,
            "layers": {
                name: tracker.state_dict()
                for name, tracker in sorted(self._trackers.items())
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", 0)) != DIVERSITY_ROUTING_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported diversity-routing controller schema.")
        expected = (
            self.num_experts,
            self.top_k,
            self.warmup_steps,
            self.ridge,
            self.token_chunk_size,
        )
        observed = (
            int(state.get("num_experts", 0)),
            int(state.get("top_k", 0)),
            int(state.get("warmup_steps", 0)),
            float(state.get("ridge", 0.0)),
            int(state.get("token_chunk_size", 0)),
        )
        if observed != expected:
            raise ValueError("Diversity-routing checkpoint configuration changed.")
        layers = state.get("layers", {})
        if not isinstance(layers, Mapping):
            raise ValueError("Diversity-routing checkpoint layers must be a mapping.")
        restored: dict[str, ExpertCooccurrenceCovariance] = {}
        for name, layer_state in layers.items():
            if not str(name).strip() or not isinstance(layer_state, Mapping):
                raise ValueError("Invalid diversity-routing layer checkpoint state.")
            tracker = ExpertCooccurrenceCovariance(self.num_experts)
            tracker.load_state_dict(layer_state)
            restored[str(name)] = tracker
        step = int(state.get("step", 0))
        if step < 0:
            raise ValueError("Diversity-routing checkpoint step must be non-negative.")
        self._step = step
        self._trackers = restored


def mahalanobis_subset_objective(
    scores: Any, covariance: Any, subset: list[int], *, ridge: float
) -> float:
    if not subset:
        return 0.0
    chosen = torch.tensor(subset, dtype=torch.long, device=scores.device)
    vector = scores.index_select(0, chosen).detach().double()
    matrix = covariance.to(device=scores.device, dtype=torch.float64).index_select(
        0, chosen
    ).index_select(1, chosen)
    regularized = matrix + float(ridge) * torch.eye(
        len(subset), device=scores.device, dtype=torch.float64
    )
    return float(torch.dot(vector, torch.linalg.solve(regularized, vector)).item())


def diversity_aware_greedy_topk(
    scores: Any,
    covariance: Any,
    *,
    top_k: int,
    ridge: float = 1e-4,
) -> Any:
    """Deterministic MP-MoE reference selector; ties choose the lowest id."""
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("scores must have shape [tokens, experts].")
    experts = int(scores.shape[1])
    if covariance.shape != (experts, experts):
        raise ValueError("covariance shape must match the score expert dimension.")
    if int(top_k) <= 0 or int(top_k) > experts or float(ridge) <= 0.0:
        raise ValueError("top_k and ridge are invalid for diversity-aware routing.")
    rows: list[list[int]] = []
    for row in scores:
        selected: list[int] = []
        for _ in range(int(top_k)):
            best_expert, best_value = -1, float("-inf")
            for expert in range(experts):
                if expert in selected:
                    continue
                value = mahalanobis_subset_objective(
                    row, covariance, [*selected, expert], ridge=ridge
                )
                if value > best_value:
                    best_expert, best_value = expert, value
            selected.append(best_expert)
        rows.append(selected)
    return torch.tensor(rows, dtype=torch.long, device=scores.device)


def diversity_aware_batched_topk(
    scores: Any,
    covariance: Any,
    *,
    top_k: int,
    ridge: float = 1e-4,
    token_chunk_size: int = 2048,
) -> Any:
    """Chunked device-resident selector matching the scalar reference."""
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("scores must have shape [tokens, experts].")
    experts = int(scores.shape[1])
    if covariance.shape != (experts, experts):
        raise ValueError("covariance shape must match the score expert dimension.")
    if int(top_k) <= 0 or int(top_k) > experts or float(ridge) <= 0.0:
        raise ValueError("top_k and ridge are invalid for diversity-aware routing.")
    if int(token_chunk_size) <= 0:
        raise ValueError("token_chunk_size must be positive.")
    covariance64 = covariance.to(device=scores.device, dtype=torch.float64)
    expert_ids = torch.arange(experts, device=scores.device, dtype=torch.long)
    diagonal = covariance64.diagonal() + float(ridge)
    outputs: list[Any] = []
    for start in range(0, int(scores.shape[0]), int(token_chunk_size)):
        chunk = scores[start : start + int(token_chunk_size)].detach().double()
        tokens = int(chunk.shape[0])
        selected = torch.empty((tokens, 0), device=scores.device, dtype=torch.long)
        inverse = torch.empty((tokens, 0, 0), device=scores.device, dtype=torch.float64)
        for width in range(int(top_k)):
            if width == 0:
                residual = chunk
                schur = diagonal.unsqueeze(0).expand(tokens, -1)
            else:
                selected_scores = chunk.gather(1, selected)
                alpha = torch.bmm(inverse, selected_scores.unsqueeze(-1)).squeeze(-1)
                candidate_ids = expert_ids.view(1, experts, 1).expand(
                    tokens, -1, width
                )
                selected_ids = selected.unsqueeze(1).expand(-1, experts, -1)
                cross = covariance64[selected_ids, candidate_ids]
                inverse_cross = torch.einsum("tij,tej->tei", inverse, cross)
                residual = chunk - (cross * alpha.unsqueeze(1)).sum(dim=-1)
                schur = diagonal.unsqueeze(0) - (cross * inverse_cross).sum(dim=-1)
            values = residual.square() / schur.clamp_min(float(ridge) * 1e-6)
            if selected.numel():
                values.scatter_(1, selected, float("-inf"))
            choice = values.argmax(dim=1, keepdim=True)
            if width == 0:
                chosen_schur = schur.gather(1, choice).reshape(tokens, 1, 1)
                inverse = chosen_schur.reciprocal()
            else:
                choice_ids = choice.expand(-1, width)
                chosen_cross = covariance64[selected, choice_ids]
                update = torch.bmm(inverse, chosen_cross.unsqueeze(-1))
                chosen_schur = schur.gather(1, choice).reshape(tokens, 1, 1)
                top_left = inverse + torch.bmm(
                    update, update.transpose(1, 2)
                ) / chosen_schur
                right = -update / chosen_schur
                inverse = torch.cat(
                    (
                        torch.cat((top_left, right), dim=2),
                        torch.cat(
                            (right.transpose(1, 2), chosen_schur.reciprocal()),
                            dim=2,
                        ),
                    ),
                    dim=1,
                )
            selected = torch.cat((selected, choice), dim=1)
        outputs.append(selected)
    return torch.cat(outputs, dim=0)


__all__ = [
    "DIVERSITY_ROUTING_STATE_SCHEMA_VERSION",
    "DiversityAwareRoutingController",
    "ExpertCooccurrenceCovariance",
    "diversity_aware_batched_topk",
    "diversity_aware_greedy_topk",
    "mahalanobis_subset_objective",
]
