"""Single binding point for the optional MagiCompiler dependency.

MagiCompiler (``magi_compiler``) ships with the released MAGI-2 runtime and is
not installable from PyPI. The vendored MAGI-2 modules decorate module-scope
callables with it, so every one of them needs the names to exist at import time
whether or not the package is present. The fallback decorators are identity:
the decorated callable keeps its eager PyTorch definition.

Identity registration has one consequence a caller cannot observe from the
decorated function itself. ``magi_register_custom_op("magi2::<name>", ...)``
publishes ``torch.ops.magi2.<name>`` only under the real compiler; under the
fallback the ``magi2`` operator namespace stays empty, and any module that
dispatches through ``torch.ops.magi2.*`` fails inside its forward with an
``_OpNamespace`` ``AttributeError``. Registration is a graph-boundary concern
and is independent of whether an implementation exists: the decorated callables
are plain module-level functions that already carry SandAI's own eager path.
:func:`resolve_magi2_op` therefore binds a call site to the registered operator
when one exists and to that eager callable otherwise, and
:func:`require_magi2_custom_ops` fails only when neither is reachable.

Attribution: SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview).
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
from typing import Callable, Sequence

import torch


# Operator namespace ``magi_register_custom_op`` publishes into.
MAGI2_OP_NAMESPACE = "magi2"

# Operators the vendored refiner reaches through ``torch.ops.magi2``.
MAGI2_REFINER_REQUIRED_OPS: tuple[str, ...] = (
    "flash_attn_func",
    "flex_flash_attn_func",
)

# Module holding the eager callables the operators are registered from. Each
# name in ``MAGI2_REFINER_REQUIRED_OPS`` is a module-level attribute there.
MAGI2_EAGER_IMPL_MODULE = "mirai.vendors.magi2_preview.model.magi2_refiner"

# Distribution the eager attention path calls into. Both eager implementations
# reach FlashAttention-2 through ``flash_attn.flash_attn_interface``.
MAGI2_EAGER_IMPL_DEPENDENCY = "flash_attn"


@functools.lru_cache(maxsize=1)
def magi_attention_flex_flash_attn_func() -> Callable[..., object] | None:
    """MagiAttention's flex entry point, or ``None`` when it is not usable.

    ``find_spec`` proves only that the distribution is on the path. MagiAttention
    carries CUDA extensions built for compute capability 9.0, so an install that
    is present can still fail to import on any other architecture; importing the
    entry point is the probe that separates the two. Resolution stays lazy
    because the import touches CUDA.
    """
    if importlib.util.find_spec("magi_attention") is None:
        return None
    try:
        from magi_attention.api import flex_flash_attn_func
    except Exception:
        return None
    return flex_flash_attn_func

try:
    from magi_compiler import magi_compile
    from magi_compiler.api import magi_register_custom_op
    from magi_compiler.config import CompileConfig

    MAGI_COMPILER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):

    def _identity_decorator(*args, **kwargs):
        """Return the decorated object unchanged, bare or parameterized."""
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda value: value

    class CompileConfig:  # type: ignore[no-redef]
        """Placeholder for the compiler's config type.

        Vendored modules annotate ``config_patch`` hooks with this type, and the
        annotations are evaluated at definition time. Nothing constructs it
        under the fallback because no patch hook is ever invoked.
        """

    magi_compile = _identity_decorator
    magi_register_custom_op = _identity_decorator
    MAGI_COMPILER_AVAILABLE = False


def missing_magi2_custom_ops(op_names: Sequence[str]) -> tuple[str, ...]:
    """Names in ``op_names`` absent from the ``torch.ops.magi2`` namespace.

    ``torch.ops.<ns>`` resolves lazily, so attribute access is the only probe
    that reflects whether an operator was actually registered.
    """
    namespace = getattr(torch.ops, MAGI2_OP_NAMESPACE)
    return tuple(name for name in op_names if not hasattr(namespace, name))


@functools.lru_cache(maxsize=None)
def resolve_magi2_op(name: str, eager_impl: Callable[..., object]) -> Callable[..., object]:
    """Bind one ``torch.ops.magi2.<name>`` call site to a reachable callable.

    The registered operator wins whenever MagiCompiler published one, so a real
    compiler install keeps its graph boundary and its dispatch semantics. When
    the namespace is empty the same computation is still available as
    ``eager_impl``, the module-level function the operator was registered from.
    """
    namespace = getattr(torch.ops, MAGI2_OP_NAMESPACE)
    registered = getattr(namespace, name, None)
    return eager_impl if registered is None else registered


def unimplemented_magi2_ops(
    op_names: Sequence[str] = MAGI2_REFINER_REQUIRED_OPS,
) -> tuple[str, ...]:
    """Names in ``op_names`` with neither a registered operator nor an eager one.

    The eager module imports FlashAttention-2 at module scope, so an import
    failure there leaves every unregistered name without an implementation.
    """
    missing = missing_magi2_custom_ops(op_names)
    if not missing:
        return ()
    try:
        module = importlib.import_module(MAGI2_EAGER_IMPL_MODULE)
    except ImportError:
        return missing
    return tuple(name for name in missing if not callable(getattr(module, name, None)))


def require_magi2_custom_ops(
    component: str,
    op_names: Sequence[str] = MAGI2_REFINER_REQUIRED_OPS,
) -> None:
    """Fail before ``component`` runs when no attention implementation is reachable.

    Registration alone is not the precondition: a missing ``torch.ops.magi2``
    operator still leaves the eager implementation, which needs its
    FlashAttention-2 dependency. The stage cannot run only when one of the two
    is unreachable for an operator it dispatches.
    """
    if not missing_magi2_custom_ops(op_names):
        return
    if importlib.util.find_spec(MAGI2_EAGER_IMPL_DEPENDENCY) is None:
        raise RuntimeError(
            f"{component} has no registered torch.ops.{MAGI2_OP_NAMESPACE} "
            "attention operators, so it runs the eager implementation vendored "
            f"beside them, and that path imports '{MAGI2_EAGER_IMPL_DEPENDENCY}' "
            f"which is not installed. Install '{MAGI2_EAGER_IMPL_DEPENDENCY}', "
            "install the MagiCompiler distribution shipped with the MAGI-2 "
            "release, or run the preview stage without refinement."
        )
    unimplemented = unimplemented_magi2_ops(op_names)
    if unimplemented:
        listed = ", ".join(
            f"torch.ops.{MAGI2_OP_NAMESPACE}.{name}" for name in unimplemented
        )
        cause = (
            "MagiCompiler ('magi_compiler') is not installed, so the identity "
            "fallback registered none of them"
            if not MAGI_COMPILER_AVAILABLE
            else "the installed MagiCompiler build registers none of them"
        )
        verb = "is" if len(unimplemented) == 1 else "are"
        raise RuntimeError(
            f"{component} dispatches its attention through torch.ops."
            f"{MAGI2_OP_NAMESPACE}, and {listed} {verb} unavailable: {cause}. No "
            f"eager implementation is exported by {MAGI2_EAGER_IMPL_MODULE} "
            "either, so the stage cannot run. Install the MagiCompiler "
            "distribution shipped with the MAGI-2 release into this "
            "environment, or run the preview stage without refinement."
        )


__all__ = [
    "MAGI2_EAGER_IMPL_DEPENDENCY",
    "MAGI2_EAGER_IMPL_MODULE",
    "MAGI2_OP_NAMESPACE",
    "MAGI2_REFINER_REQUIRED_OPS",
    "MAGI_COMPILER_AVAILABLE",
    "CompileConfig",
    "magi_attention_flex_flash_attn_func",
    "magi_compile",
    "magi_register_custom_op",
    "missing_magi2_custom_ops",
    "require_magi2_custom_ops",
    "resolve_magi2_op",
    "unimplemented_magi2_ops",
]
