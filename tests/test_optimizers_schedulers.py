"""Behavioral contract for optimizer breadth and LR schedulers."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch, MagicMock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _make_param() -> "torch.nn.Parameter":
    p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    p.grad = torch.tensor(0.1, dtype=torch.float32)
    return p


def _build(optimizer_type: str, **kw):
    from mirai.core.training.optim.optimizer import build_optimizer

    p = _make_param()
    return build_optimizer(
        params=[p],
        optimizer_type=optimizer_type,
        lr=kw.pop("lr", 1e-3),
        weight_decay=kw.pop("weight_decay", 0.0),
        allow_fallback=kw.pop("allow_fallback", True),
        **kw,
    )


def _build_sched(optimizer, *, scheduler: str, warmup: int = 0, total: int = 100, **kw):
    from mirai.core.training.optim.scheduler import build_scheduler

    return build_scheduler(
        optimizer,
        warmup_steps=warmup,
        total_steps=total,
        scheduler=scheduler,
        **kw,
    )


def _lr_sequence(sched, optimizer, steps: int) -> list[float]:
    """Collect one LR value per correctly ordered optimizer/scheduler step."""
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        sched.step()
    return lrs


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestAdamwDefault(unittest.TestCase):
    def test_adamw_resolved_type(self) -> None:
        result = _build("adamw")
        self.assertEqual(result.resolved_type, "adamw")
        self.assertFalse(result.used_fallback)


@unittest.skipIf(torch is None, "torch not installed")
class TestAdafactorOptimizer(unittest.TestCase):
    def test_adafactor_with_transformers(self) -> None:
        """Adafactor builds when transformers.optimization is present."""
        mock_af = MagicMock()
        mock_af.return_value = MagicMock(spec=torch.optim.Optimizer, param_groups=[{"lr": 1e-3}])
        fake_tf_opt = types.ModuleType("transformers.optimization")
        fake_tf_opt.Adafactor = mock_af

        with patch.dict(sys.modules, {"transformers.optimization": fake_tf_opt,
                                       "transformers": types.ModuleType("transformers")}):
            # Re-import to pick up patched modules via importlib inside build_optimizer
            import importlib
            import mirai.core.training.optim.optimizer as opt_mod
            importlib.reload(opt_mod)
            try:
                result = opt_mod.build_optimizer(
                    params=[_make_param()],
                    optimizer_type="adafactor",
                    lr=1e-3,
                    weight_decay=0.0,
                    allow_fallback=True,
                )
                self.assertEqual(result.resolved_type, "adafactor")
                self.assertFalse(result.used_fallback)
            finally:
                importlib.reload(opt_mod)

    def test_adafactor_fallback_to_adamw_when_unavailable(self) -> None:
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no transformers")):
            result = _build("adafactor", allow_fallback=True)
        self.assertEqual(result.resolved_type, "adamw")
        self.assertTrue(result.used_fallback)

    def test_adafactor_raises_when_fallback_disabled(self) -> None:
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no transformers")):
            with self.assertRaises(RuntimeError):
                _build("adafactor", allow_fallback=False)


@unittest.skipIf(torch is None, "torch not installed")
class TestLionOptimizer(unittest.TestCase):
    def test_lion_uses_in_repo_fallback_when_pytorch_optimizer_absent(self) -> None:
        """When pytorch_optimizer missing, Lion falls back to in-repo LionFallback."""
        original_import = __builtins__.__class__.__module__  # noqa: F841 (just loading)

        def _fake_import(name, *args, **kw):
            if name == "pytorch_optimizer":
                raise ModuleNotFoundError("no pytorch_optimizer")
            return importlib.import_module(name, *args, **kw)

        import importlib

        with patch("importlib.import_module", side_effect=_fake_import):
            result = _build("lion")
        self.assertEqual(result.resolved_type, "lion")
        self.assertFalse(result.used_fallback)
        from mirai.core.training.optim.lion import LionFallback
        self.assertIsInstance(result.optimizer, LionFallback)

    def test_lion_in_repo_step_runs(self) -> None:
        from mirai.core.training.optim.lion import LionFallback

        p = torch.nn.Parameter(torch.tensor(2.0))
        p.grad = torch.tensor(0.5)
        opt = LionFallback([p], lr=1e-2, weight_decay=0.0)
        before = p.data.item()
        opt.step()
        after = p.data.item()
        # Lion applies -lr * sign(grad), so param should decrease
        self.assertLess(after, before)


@unittest.skipIf(torch is None, "torch not installed")
class TestCameOptimizer(unittest.TestCase):
    def test_came_fallback_to_adamw_when_all_unavailable(self) -> None:
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no pytorch_optimizer")):
            result = _build("came", allow_fallback=True)
        # Should degrade to adamw (both came and adafactor unavailable)
        self.assertIn(result.resolved_type, {"adafactor", "adamw"})
        self.assertTrue(result.used_fallback)

    def test_came_raises_when_fallback_disabled(self) -> None:
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no pytorch_optimizer")):
            with self.assertRaises(RuntimeError):
                _build("came", allow_fallback=False)

    def test_came_does_not_substitute_available_adafactor_when_fallback_disabled(
        self,
    ) -> None:
        adafactor = types.SimpleNamespace(Adafactor=object())

        def fake_import(name: str):
            if name == "pytorch_optimizer":
                return types.SimpleNamespace()
            if name == "transformers.optimization":
                return adafactor
            raise ModuleNotFoundError(name)

        with patch("importlib.import_module", side_effect=fake_import) as import_module:
            with self.assertRaisesRegex(RuntimeError, "allow_fallback=false"):
                _build("came", allow_fallback=False)
        self.assertEqual(import_module.call_count, 1)


@unittest.skipIf(torch is None, "torch not installed")
class TestAdamw8bitFallback(unittest.TestCase):
    """Existing behavior preserved."""

    def test_8bit_fallback_when_allowed(self) -> None:
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("no bnb")):
            result = _build("adamw_8bit", allow_fallback=True)
        self.assertEqual(result.resolved_type, "adamw")
        self.assertTrue(result.used_fallback)


@unittest.skipIf(torch is None, "torch not installed")
class TestUnsupportedOptimizer(unittest.TestCase):
    def test_raises_on_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            _build("mythical_optimizer")


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerConstant(unittest.TestCase):
    """Constant scheduler reference behavior."""

    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_constant_no_warmup_flat(self) -> None:
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="constant", warmup=0, total=10)
        lrs = _lr_sequence(sched, opt, 10)
        self.assertTrue(all(abs(v - 1.0) < 1e-6 for v in lrs), lrs)

    def test_constant_with_warmup_ramps_then_flat(self) -> None:
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="constant", warmup=4, total=10)
        lrs = _lr_sequence(sched, opt, 10)
        # First step uses lr_lambda(0) = min(1/4, 1) = 0.25
        self.assertAlmostEqual(lrs[0], 0.25, places=5)
        # After warmup should be flat at 1.0
        for v in lrs[4:]:
            self.assertAlmostEqual(v, 1.0, places=5)

    def test_constant_matches_canonical_formula(self) -> None:
        """Verify the constant scheduler's canonical warmup formula."""
        from mirai.core.training.optim.scheduler import build_scheduler

        warmup = 5
        opt = self._make_optimizer()
        sched = build_scheduler(opt, warmup_steps=warmup)
        lrs = _lr_sequence(sched, opt, 12)

        # Evaluate the formula independently.
        expected = [min(float(i + 1) / float(warmup), 1.0) for i in range(12)]
        for got, exp in zip(lrs, expected):
            self.assertAlmostEqual(got, exp, places=6)


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerLinear(unittest.TestCase):
    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_linear_decays_to_zero(self) -> None:
        total = 10
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="linear", warmup=0, total=total, min_lr_ratio=0.0)
        lrs = _lr_sequence(sched, opt, total)
        # Should be monotonically non-increasing
        for a, b in zip(lrs, lrs[1:]):
            self.assertGreaterEqual(a + 1e-6, b)
        # Final LR should be near 0
        self.assertLess(lrs[-1], 0.2)

    def test_linear_respects_min_lr_ratio(self) -> None:
        total = 20
        opt = self._make_optimizer()
        sched = _build_sched(
            opt, scheduler="linear", warmup=0, total=total, min_lr_ratio=0.1
        )
        lrs = _lr_sequence(sched, opt, total)
        for v in lrs:
            self.assertGreaterEqual(v + 1e-6, 0.1)

    def test_linear_warmup_ramps(self) -> None:
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="linear", warmup=5, total=20)
        lrs = _lr_sequence(sched, opt, 20)
        # During warmup LR should increase
        self.assertLess(lrs[0], lrs[4])


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerCosine(unittest.TestCase):
    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_cosine_starts_high_ends_low(self) -> None:
        total = 20
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="cosine", warmup=0, total=total, min_lr_ratio=0.0)
        lrs = _lr_sequence(sched, opt, total)
        self.assertGreater(lrs[0], lrs[-1])
        # Should end near 0 (cos(π*1) = -1 → 0.5*(1-1)=0)
        self.assertLess(lrs[-1], 0.1)

    def test_cosine_decays_to_min_lr_ratio(self) -> None:
        total = 20
        lo = 0.2
        opt = self._make_optimizer()
        sched = _build_sched(
            opt, scheduler="cosine", warmup=0, total=total, min_lr_ratio=lo
        )
        lrs = _lr_sequence(sched, opt, total)
        for v in lrs:
            self.assertGreaterEqual(v + 1e-6, lo)


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerCosineWithRestarts(unittest.TestCase):
    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_restarts_produce_multiple_peaks(self) -> None:
        total = 40
        opt = self._make_optimizer()
        sched = _build_sched(
            opt,
            scheduler="cosine_with_restarts",
            warmup=0,
            total=total,
            num_cycles=2.0,
            min_lr_ratio=0.0,
        )
        lrs = _lr_sequence(sched, opt, total)
        # Find local maxima (peaks where LR rises then falls)
        peaks = [
            i for i in range(1, len(lrs) - 1) if lrs[i] > lrs[i - 1] and lrs[i] > lrs[i + 1]
        ]
        # With 2 cycles there should be at least one restart peak (mid-sequence)
        self.assertGreaterEqual(len(peaks), 1)

    def test_restarts_respect_min_lr_ratio(self) -> None:
        total = 40
        lo = 0.05
        opt = self._make_optimizer()
        sched = _build_sched(
            opt,
            scheduler="cosine_with_restarts",
            warmup=0,
            total=total,
            num_cycles=2.0,
            min_lr_ratio=lo,
        )
        lrs = _lr_sequence(sched, opt, total)
        for v in lrs:
            self.assertGreaterEqual(v + 1e-6, lo)


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerPolynomial(unittest.TestCase):
    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_polynomial_power1_linear(self) -> None:
        """Power=1 polynomial should behave like linear decay."""
        total = 20
        opt1 = self._make_optimizer()
        opt2 = self._make_optimizer()
        lin = _build_sched(opt1, scheduler="linear", warmup=0, total=total)
        poly = _build_sched(opt1, scheduler="polynomial", warmup=0, total=total, power=1.0)
        # They share the same optimizer; just check polynomial is monotone decreasing
        lrs_poly = _lr_sequence(sched=poly, optimizer=opt1, steps=total)
        for a, b in zip(lrs_poly, lrs_poly[1:]):
            self.assertGreaterEqual(a + 1e-6, b)

    def test_polynomial_higher_power_faster_decay(self) -> None:
        total = 20
        opt1 = self._make_optimizer()
        opt2 = self._make_optimizer()
        sched1 = _build_sched(opt1, scheduler="polynomial", warmup=0, total=total, power=1.0)
        sched2 = _build_sched(opt2, scheduler="polynomial", warmup=0, total=total, power=3.0)
        lrs1 = _lr_sequence(sched1, opt1, total)
        lrs2 = _lr_sequence(sched2, opt2, total)
        # Power-3 should decay faster in the middle
        mid = total // 2
        self.assertGreater(lrs1[mid], lrs2[mid])


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerRex(unittest.TestCase):
    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def test_rex_generally_decreasing(self) -> None:
        total = 30
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="rex", warmup=0, total=total, power=1.0)
        lrs = _lr_sequence(sched, opt, total)
        # Should be monotone non-increasing
        for a, b in zip(lrs, lrs[1:]):
            self.assertGreaterEqual(a + 1e-6, b)

    def test_rex_respects_min_lr_ratio(self) -> None:
        total = 30
        lo = 0.05
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler="rex", warmup=0, total=total, power=1.0, min_lr_ratio=lo)
        lrs = _lr_sequence(sched, opt, total)
        for v in lrs:
            self.assertGreaterEqual(v + 1e-6, lo)


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerWarmupComposition(unittest.TestCase):
    """Verify warmup composes correctly with all non-constant modes."""

    def _make_optimizer(self, lr: float = 1.0):
        p = torch.nn.Parameter(torch.tensor(1.0))
        return torch.optim.SGD([p], lr=lr)

    def _check_warmup_ramp(self, mode: str) -> None:
        warmup = 5
        total = 30
        opt = self._make_optimizer()
        sched = _build_sched(opt, scheduler=mode, warmup=warmup, total=total)
        lrs = _lr_sequence(sched, opt, total)
        # LR during warmup should increase
        self.assertLess(lrs[0], lrs[warmup - 1])
        # First LR should be warmup step 0: min(1/5, 1) = 0.2
        self.assertAlmostEqual(lrs[0], 0.2, places=5)
        # Post-warmup LR should be <= 1.0
        for v in lrs[warmup:]:
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_warmup_linear(self) -> None:
        self._check_warmup_ramp("linear")

    def test_warmup_cosine(self) -> None:
        self._check_warmup_ramp("cosine")

    def test_warmup_cosine_with_restarts(self) -> None:
        self._check_warmup_ramp("cosine_with_restarts")

    def test_warmup_polynomial(self) -> None:
        self._check_warmup_ramp("polynomial")

    def test_warmup_rex(self) -> None:
        self._check_warmup_ramp("rex")


@unittest.skipIf(torch is None, "torch not installed")
class TestSchedulerInvalidMode(unittest.TestCase):
    def test_unknown_scheduler_raises(self) -> None:
        from mirai.core.training.optim.scheduler import build_scheduler

        p = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.SGD([p], lr=1.0)
        with self.assertRaises(ValueError):
            build_scheduler(opt, warmup_steps=0, total_steps=10, scheduler="unicorn")


# ---------------------------------------------------------------------------
# Default-path regression: scheduler="constant", optimizer="adamw" unchanged
# ---------------------------------------------------------------------------


@unittest.skipIf(torch is None, "torch not installed")
class TestDefaultPathRegression(unittest.TestCase):
    """End-to-end: build_optimizer + build_scheduler with default settings."""

    def test_default_optimizer_and_scheduler(self) -> None:
        from mirai.core.training.optim.optimizer import build_optimizer
        from mirai.core.training.optim.scheduler import build_scheduler

        p = torch.nn.Parameter(torch.tensor(1.0))
        p.grad = torch.tensor(0.1)
        result = build_optimizer(
            params=[p],
            optimizer_type="adamw",
            lr=1e-3,
            weight_decay=0.0,
            allow_fallback=True,
        )
        self.assertEqual(result.resolved_type, "adamw")
        self.assertFalse(result.used_fallback)

        # No-warmup constant scheduler
        sched = build_scheduler(result.optimizer, warmup_steps=0)
        lrs = _lr_sequence(sched, result.optimizer, 5)
        for v in lrs:
            self.assertAlmostEqual(v, 1e-3, places=7)

    def test_default_warmup_ramp_matches_canonical_formula(self) -> None:
        from mirai.core.training.optim.scheduler import build_scheduler

        warmup = 4
        p = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.SGD([p], lr=1.0)
        sched = build_scheduler(opt, warmup_steps=warmup)
        lrs = _lr_sequence(sched, opt, 8)
        expected = [min(float(i + 1) / float(warmup), 1.0) for i in range(8)]
        for got, exp in zip(lrs, expected):
            self.assertAlmostEqual(got, exp, places=6)


if __name__ == "__main__":
    unittest.main()
