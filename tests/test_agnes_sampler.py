"""Released Agnes scheduler scalar-device and precision boundaries."""

import os

import pytest
import torch

from mirai.core.models.agnes_video.sampler import AgnesFlowUniPCMultistepSolver


def test_released_first_step_preserves_bfloat16_velocity_rounding():
    solver = AgnesFlowUniPCMultistepSolver(shift=5.0)
    solver.set_timesteps(32, shift=5.0)
    generator = torch.Generator().manual_seed(193)
    sample = torch.randn((2, 8, 7, 9), generator=generator) * 30
    velocity = (torch.randn(sample.shape, generator=generator) * 5).bfloat16()
    current, following = solver._sigmas[:2]
    h = (torch.log(1 - following) - torch.log(following)
         - (torch.log(1 - current) - torch.log(current)))
    clean = sample - current * velocity
    expected = following / current * sample - (1 - following) * torch.expm1(-h) * clean
    actual = solver.step(velocity, solver.timesteps[0], sample).prev_sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(clean, sample - current * velocity.float())
    assert solver._sigmas.device.type == "cpu"


def test_released_zero_sigma_endpoint_is_exact_data_prediction():
    solver = AgnesFlowUniPCMultistepSolver(shift=5.0)
    solver.set_timesteps(1, shift=5.0)
    sample = torch.tensor([[0.7, -12.9, 39.1]])
    velocity = torch.tensor([[0.12, -0.37, 1.9]], dtype=torch.bfloat16)
    expected = sample - solver._sigmas[0] * velocity
    assert torch.isposinf(solver._lambda(solver._sigmas[-1]))
    actual = solver.step(velocity, solver.timesteps[0], sample).prev_sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _check_begin_index_preserves_cpu_sigmas(device):
    solver = AgnesFlowUniPCMultistepSolver(shift=5.0)
    solver.set_timesteps(32, device=device, shift=5.0)
    sigmas = solver._sigmas.clone()
    timesteps = solver.timesteps.clone()
    source_timesteps = solver.source_timesteps.clone()
    solver.set_begin_index(11)
    assert solver._sigmas.device.type == "cpu"
    torch.testing.assert_close(solver._sigmas, sigmas[11:], rtol=0, atol=0)
    torch.testing.assert_close(solver.timesteps, timesteps[11:], rtol=0, atol=0)
    torch.testing.assert_close(solver.source_timesteps, source_timesteps[11:], rtol=0, atol=0)
    assert solver._step_index == 0
    assert solver.last_sample is None
    assert all(output is None for output in solver.model_outputs)


def test_begin_index_preserves_released_cpu_scalar_schedule():
    _check_begin_index_preserves_cpu_sigmas("cpu")


@pytest.mark.skipif(
    os.environ.get("MIRAI_AGNES_CUDA_PROBE") != "1" or not torch.cuda.is_available(),
    reason="CUDA sampler contracts require explicit controlled GPU-worker opt-in",
)
def test_cuda_begin_index_keeps_sigmas_on_cpu():
    _check_begin_index_preserves_cpu_sigmas("cuda")
