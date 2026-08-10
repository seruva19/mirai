"""Behavioral contracts for the LingBot-Video refiner stage.

Three properties are pinned here because they are the ones a refactor can break
silently:

1. The tail sigma grid is exact. ``compute_refiner_sigmas`` truncates the
   shifted grid at ``t_thresh``, pins the FIRST sigma to ``t_thresh`` itself
   (not to the nearest grid point), and appends ``sigma_tail_steps`` low-noise
   sigmas landing before ``sigma_min``. The expected sequences below are the
   values the reference construction yields, written out literally so a
   refactor cannot re-derive them from the code under test.
2. The refiner conditions on TEXT ONLY. It calls the text-only
   ``encode_prompt`` hook and never ``encode_conditioned_prompt``, while the
   base denoise loop does the opposite.
3. Frame 0 stays pinned to the condition latent for an image-conditioned
   refine — before the loop, before every forward, and after every solver step.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mirai.core.inference.conditioning import (  # noqa: E402
    IMAGE_TO_VIDEO,
    TEXT_TO_VIDEO,
    InferenceConditioningRequest,
    PreparedInferenceConditioning,
)
from mirai.core.models.lingbot_video.refiner import (  # noqa: E402
    LingBotRefiner,
    MAX_REFINER_SIGMA_TAIL_STEPS,
    compute_refiner_sigmas,
    resolve_refiner_conditioning,
    resolve_refiner_request,
    run_refine,
)


# (steps, shift, t_thresh, sigma_tail_steps) -> exact sigma sequence.
_REFERENCE_SCHEDULES: dict[tuple[int, float, float, int], list[float]] = {
    # Released profile. The shifted grid is
    # [1.0, 0.955224, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3]; everything above
    # 0.85 is dropped and 0.85 itself is prepended, then two tail sigmas.
    (8, 3.0, 0.85, 2): [
        0.85, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0.2, 0.1,
    ],
    # Same truncation, no tail: the grid simply ends at the last shifted value.
    (8, 3.0, 0.85, 0): [0.85, 0.833333, 0.75, 0.642857, 0.5, 0.3],
    # t_thresh exactly ON a grid point (0.9 is the third shifted value): it is
    # kept as-is and NOT duplicated by the pin.
    (8, 3.0, 0.9, 2): [
        0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0.2, 0.1,
    ],
    # Upper boundary: t_thresh == 1.0 keeps the whole shifted grid.
    (4, 3.0, 1.0, 2): [1.0, 0.9, 0.75, 0.5, 0.333333, 0.166667],
    # shift == 1.0 leaves the grid unshifted.
    (10, 1.0, 0.5, 0): [0.5, 0.4, 0.3, 0.2, 0.1],
    (12, 7.0, 0.6, 2): [0.6, 0.583333, 0.388889, 0.259259, 0.12963],
}


def test_refiner_sigma_schedule_matches_the_reference_construction() -> None:
    for (steps, shift, t_thresh, tail), expected in _REFERENCE_SCHEDULES.items():
        sigmas = compute_refiner_sigmas(
            num_inference_steps=steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=tail,
        )
        actual = [round(float(v), 6) for v in sigmas]
        assert actual == expected, (
            f"(steps={steps}, shift={shift}, t_thresh={t_thresh}, "
            f"tail={tail}) -> {actual} != {expected}"
        )
        # The tail contributes exactly ``tail`` extra sigmas.
        without_tail = compute_refiner_sigmas(
            num_inference_steps=steps, shift=shift, t_thresh=t_thresh, tail_steps=0
        )
        assert int(sigmas.numel()) == int(without_tail.numel()) + tail


def test_truncation_pins_the_first_sigma_to_t_thresh_not_the_nearest_grid_point() -> None:
    # 0.85 lies strictly between the shifted grid points 0.9 and 0.833333, so a
    # "snap to nearest grid value" implementation would start at 0.833333.
    sigmas = compute_refiner_sigmas(
        num_inference_steps=8, shift=3.0, t_thresh=0.85, tail_steps=0
    )
    assert float(sigmas[0]) == pytest.approx(0.85, abs=1e-6)
    assert float(sigmas[1]) == pytest.approx(0.833333, abs=1e-6)
    # The re-noise level and the first sampled sigma are the same number, which
    # is what makes the hand-off from prepare_refiner_latent exact.
    assert float(sigmas[0]) != pytest.approx(float(sigmas[1]), abs=1e-6)


def test_refiner_request_rejects_out_of_range_schedule_parameters() -> None:
    base = {"steps": 8, "t_thresh": 0.85, "sigma_tail_steps": 2}
    assert resolve_refiner_request(dict(base), scheduler="euler")["t_thresh"] == 0.85

    with pytest.raises(RuntimeError, match="t_thresh must lie in"):
        resolve_refiner_request({**base, "t_thresh": 0.0}, scheduler="euler")
    with pytest.raises(RuntimeError, match="t_thresh must lie in"):
        resolve_refiner_request({**base, "t_thresh": 1.5}, scheduler="euler")
    with pytest.raises(RuntimeError, match="sigma_tail_steps must lie in"):
        resolve_refiner_request({**base, "sigma_tail_steps": -1}, scheduler="euler")
    with pytest.raises(RuntimeError, match="sigma_tail_steps must lie in"):
        resolve_refiner_request(
            {**base, "sigma_tail_steps": MAX_REFINER_SIGMA_TAIL_STEPS + 1},
            scheduler="euler",
        )
    with pytest.raises(RuntimeError, match="steps must be >= 1"):
        resolve_refiner_request({**base, "steps": 0}, scheduler="euler")


def test_refiner_residency_streams_and_evicts_each_forward_block(monkeypatch) -> None:
    class _RecordingManager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.events = []
            self.released = False

        def bind(self, units, *, device) -> None:
            self.units = list(units)
            self.device = device

        def before_block(self, index) -> None:
            self.events.append(("before", index))

        def after_block(self, index) -> None:
            self.events.append(("after", index))

        def release_device(self) -> None:
            self.released = True

    managers = []

    def _manager_factory(**kwargs):
        manager = _RecordingManager(**kwargs)
        managers.append(manager)
        return manager

    import mirai.core.training.residency.block_swap as block_swap

    monkeypatch.setattr(block_swap, "BlockSwapManager", _manager_factory)
    refiner = LingBotRefiner.__new__(LingBotRefiner)
    refiner._transformer = torch.nn.Module()
    refiner._transformer.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2) for _ in range(2)]
    )
    refiner._block_swap_manager = None
    refiner._block_hook_handles = []

    class _Residency:
        enabled = True
        blocks_to_swap = 2
        mode = "async"
        block_residency_planner = "phase_aware"
        block_swap_prefetch_depth = 1
        block_residency_priority = "uniform"
        block_swap_transfer_strategy = "flat_ring"
        offload_dir = None

    refiner._place(device="cpu", residency=_Residency())
    manager = managers[0]
    assert manager.kwargs["block_swap_backward"] is True
    assert manager.kwargs["block_swap_transfer_strategy"] == "flat_ring"
    refiner.transformer.blocks[0](torch.ones(1, 2))
    assert manager.events == [("before", 0), ("after", 0)]
    refiner.release()
    assert manager.released is True


def test_compressed_refiner_loader_preserves_float32_scale_metadata() -> None:
    class _PackedTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("weight_codes", torch.ones(2, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.ones(2, dtype=torch.float32))

    seen = {}

    def _loader(config, *, subfolder, dtype=None):
        seen.update(config=config, subfolder=subfolder, dtype=dtype)
        return _PackedTransformer()

    refiner = LingBotRefiner.__new__(LingBotRefiner)
    refiner.subfolder = "refiner"
    refiner._transformer = None
    refiner._transformer_config = {"hidden_size": 2}
    refiner._block_swap_manager = None
    refiner._block_hook_handles = []
    refiner.has_weights = lambda: True
    refiner.load(
        device="cpu",
        dtype=torch.bfloat16,
        transformer_loader=_loader,
    )
    assert seen == {
        "config": {"hidden_size": 2},
        "subfolder": "refiner",
        "dtype": torch.bfloat16,
    }
    assert refiner.transformer.weight_codes.dtype is torch.uint8
    assert refiner.transformer.weight_scale.dtype is torch.float32


class _RefinerPipeline:
    """Minimal provider-hook surface the refine policy loop is allowed to touch."""

    def __init__(self, *, condition_latent=None) -> None:
        self.condition_latent = condition_latent
        self.text_prompts: list[str] = []
        self.conditioned_prompts: list[str] = []
        self.forward_inputs: list[torch.Tensor] = []

    # -- lifecycle ---------------------------------------------------------
    def release_base_transformer(self) -> None:
        return None

    def load_vae(self, *, device) -> None:
        _ = device

    def offload_vae(self) -> None:
        return None

    def load_text_encoder(self, *, device) -> None:
        _ = device

    def offload_text_encoder(self) -> None:
        return None

    # -- media -------------------------------------------------------------
    def decode_latents_native(self, latents):
        _ = latents
        return torch.zeros(3, 3, 8, 8)

    def encode_video_native(self, video, *, generator=None, sample_posterior=True):
        _ = generator, sample_posterior
        batch, _c, frames, height, width = video.shape
        return torch.full((batch, 2, frames, height // 4, width // 4), 5.0)

    def prepare_inference_conditioning(self, request, *, device, generator):
        _ = device, generator
        assert request.task == IMAGE_TO_VIDEO
        # Upstream re-encodes the condition image at the REFINER geometry.
        assert (int(request.height), int(request.width)) == (16, 16)
        return PreparedInferenceConditioning(
            task=IMAGE_TO_VIDEO,
            prompt_media=object(),
            condition_latent=self.condition_latent,
        )

    # -- conditioning ------------------------------------------------------
    def encode_prompt(self, prompt, *, device):
        _ = device
        self.text_prompts.append(str(prompt))
        return torch.zeros(1, 1, 4)

    def encode_conditioned_prompt(self, prompt, *, prepared, device):
        _ = prepared, device
        self.conditioned_prompts.append(str(prompt))
        return torch.zeros(1, 1, 4)

    # -- base denoise surface (used only by the contrast test) --------------
    def get_training_model(self):
        return torch.nn.Linear(1, 1, bias=False)

    def preview_latent_geometry(self, *, frame_count, height, width):
        _ = frame_count, height, width
        return (2, 3, 4, 4)

    def resolve_flow_shift_for_latent_shape(self, latent_shape):
        _ = latent_shape
        return 3.0

    def forward(self, sample, timestep, text_embeds):
        _ = timestep, text_embeds
        self.forward_inputs.append(torch.as_tensor(sample).detach().clone())
        return torch.ones_like(torch.as_tensor(sample))

    # -- model -------------------------------------------------------------
    def refiner_forward(self, latents, timestep, text_embeds):
        _ = timestep, text_embeds
        self.forward_inputs.append(torch.as_tensor(latents).detach().clone())
        # A non-zero velocity guarantees the solver actually moves frame 0, so an
        # absent re-pin cannot pass by accident.
        return torch.ones_like(torch.as_tensor(latents))

    @staticmethod
    def refiner_residency_request():
        return None

    @staticmethod
    def load_refiner_transformer(config, *, subfolder, dtype=None):
        raise AssertionError(
            "the contract refiner must already be loaded: "
            f"{config!r}, {subfolder!r}, {dtype!r}"
        )


class _LoadedRefiner:
    def __init__(self) -> None:
        self.released = False

    @staticmethod
    def has_weights() -> bool:
        return True

    def load(
        self,
        *,
        device,
        dtype=None,
        residency=None,
        transformer_loader=None,
    ) -> None:
        _ = device, dtype, residency, transformer_loader

    def release(self) -> None:
        self.released = True


_REFINE_KW = dict(
    prompt="a prompt",
    negative_prompt="",
    seed=7,
    height=16,
    width=16,
    steps=6,
    cfg_scale=1.0,
    shift=3.0,
    t_thresh=0.5,
    sigma_tail_steps=1,
    scheduler="euler",
    device="cpu",
)


def test_refiner_conditions_on_text_only_while_the_base_stage_does_not() -> None:
    condition = torch.full((1, 2, 1, 4, 4), 3.0)
    pipeline = _RefinerPipeline(condition_latent=condition)
    run_refine(
        pipeline=pipeline,
        refiner=_LoadedRefiner(),
        base_latent=torch.zeros(2, 3, 4, 4),
        conditioning=InferenceConditioningRequest(
            task=IMAGE_TO_VIDEO,
            input_image=object(),
            frame_count=3,
            height=8,
            width=8,
        ),
        **_REFINE_KW,
    )
    # The condition image is available (it produced the pin), yet the refiner's
    # prompt embeddings were built through the text-only hook.
    assert pipeline.text_prompts == ["a prompt", ""]
    assert pipeline.conditioned_prompts == []

    # The base denoise loop, given the same pipeline and the same conditioning,
    # takes the image-aware hook — so the suppression is a property of the
    # refiner stage, not a capability the pipeline is missing.
    from mirai.core.training.preview.preview import run_native_denoise_loop

    base = _RefinerPipeline(condition_latent=condition)
    run_native_denoise_loop(
        pipeline=base,
        prompt="a prompt",
        negative_prompt="",
        cfg_scale=1.0,
        seed=7,
        step=0,
        denoise_steps=2,
        scheduler="euler",
        frame_count=3,
        height=16,
        width=16,
        conditioning=InferenceConditioningRequest(
            task=IMAGE_TO_VIDEO,
            input_image=object(),
            frame_count=3,
            height=16,
            width=16,
        ),
    )
    assert base.conditioned_prompts == ["a prompt", ""]
    assert base.text_prompts == []


def test_ti2v_refine_keeps_frame_zero_pinned_through_every_step() -> None:
    condition = torch.full((1, 2, 1, 4, 4), 3.0)
    pipeline = _RefinerPipeline(condition_latent=condition)
    refined = run_refine(
        pipeline=pipeline,
        refiner=_LoadedRefiner(),
        base_latent=torch.zeros(2, 3, 4, 4),
        conditioning=InferenceConditioningRequest(
            task=IMAGE_TO_VIDEO,
            input_image=object(),
            frame_count=3,
            height=8,
            width=8,
        ),
        **_REFINE_KW,
    )
    assert pipeline.forward_inputs, "the refine loop never ran a forward"
    expected = torch.full((2, 4, 4), 3.0)
    # Pinned before every forward ...
    for sample in pipeline.forward_inputs:
        assert torch.equal(sample[0, :, 0], expected)
    # ... and bit-identical after the final step.
    assert torch.equal(refined[:, 0], expected)
    # The rest of the clip did move, so the pin is not masking a no-op loop.
    assert not torch.equal(refined[:, 1], expected)


def test_text_only_refine_pins_nothing_and_needs_no_conditioning_hook() -> None:
    class _NoConditioningPipeline(_RefinerPipeline):
        def prepare_inference_conditioning(self, request, *, device, generator):
            raise AssertionError("a text-only refine must not prepare conditioning")

    pipeline = _NoConditioningPipeline()
    refined = run_refine(
        pipeline=pipeline,
        refiner=_LoadedRefiner(),
        base_latent=torch.zeros(2, 3, 4, 4),
        conditioning=InferenceConditioningRequest(
            task=TEXT_TO_VIDEO, frame_count=3, height=8, width=8
        ),
        **_REFINE_KW,
    )
    assert refined.shape[0] == 2
    # ``conditioning=None`` (no media request at all) behaves identically.
    same = run_refine(
        pipeline=_NoConditioningPipeline(),
        refiner=_LoadedRefiner(),
        base_latent=torch.zeros(2, 3, 4, 4),
        conditioning=None,
        **_REFINE_KW,
    )
    assert torch.equal(refined, same)


def test_image_conditioned_refine_fails_when_the_pin_cannot_be_built() -> None:
    request = InferenceConditioningRequest(
        task=IMAGE_TO_VIDEO, input_image=object(), frame_count=3, height=8, width=8
    )

    class _NoHook:
        pass

    with pytest.raises(RuntimeError, match="prepare_inference_conditioning"):
        resolve_refiner_conditioning(
            _NoHook(), request, height=16, width=16, device="cpu", generator=None
        )

    class _EmptyPrepared:
        @staticmethod
        def prepare_inference_conditioning(request, *, device, generator):
            _ = request, device, generator
            return PreparedInferenceConditioning(task=IMAGE_TO_VIDEO)

    with pytest.raises(RuntimeError, match="no condition latent"):
        resolve_refiner_conditioning(
            _EmptyPrepared(), request, height=16, width=16, device="cpu", generator=None
        )

    with pytest.raises(TypeError, match="InferenceConditioningRequest"):
        resolve_refiner_conditioning(
            _NoHook(), {"task": "ti2v"}, height=16, width=16, device="cpu", generator=None
        )
