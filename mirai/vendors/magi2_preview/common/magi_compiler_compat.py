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
``_OpNamespace`` ``AttributeError``. :func:`require_magi2_custom_ops` turns that
into an explicit precondition at the seam where such a module is constructed.

Attribution: SandAI MAGI-2-preview, Apache-2.0
(https://github.com/SandAI-org/MAGI-2-preview).
"""

from __future__ import annotations

from typing import Sequence

import torch


# Operator namespace ``magi_register_custom_op`` publishes into.
MAGI2_OP_NAMESPACE = "magi2"

# Operators the vendored refiner reaches through ``torch.ops.magi2``.
MAGI2_REFINER_REQUIRED_OPS: tuple[str, ...] = (
    "flash_attn_func",
    "flex_flash_attn_func",
)

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


def require_magi2_custom_ops(
    component: str,
    op_names: Sequence[str] = MAGI2_REFINER_REQUIRED_OPS,
) -> None:
    """Fail before ``component`` runs when its ``magi2`` operators are absent."""
    missing = missing_magi2_custom_ops(op_names)
    if not missing:
        return
    cause = (
        "MagiCompiler ('magi_compiler') is not installed, so the identity "
        "fallback registered none of them"
        if not MAGI_COMPILER_AVAILABLE
        else "the installed MagiCompiler build registers none of them"
    )
    listed = ", ".join(f"torch.ops.{MAGI2_OP_NAMESPACE}.{name}" for name in missing)
    verb = "is" if len(missing) == 1 else "are"
    raise RuntimeError(
        f"{component} dispatches its attention through torch.ops."
        f"{MAGI2_OP_NAMESPACE}, and {listed} {verb} unavailable: {cause}. There "
        "is no eager substitute for these operators, so the stage cannot run. "
        "Install the MagiCompiler distribution shipped with the MAGI-2 release "
        "into this environment, or run the preview stage without refinement."
    )


__all__ = [
    "MAGI2_OP_NAMESPACE",
    "MAGI2_REFINER_REQUIRED_OPS",
    "MAGI_COMPILER_AVAILABLE",
    "CompileConfig",
    "magi_compile",
    "magi_register_custom_op",
    "missing_magi2_custom_ops",
    "require_magi2_custom_ops",
]
