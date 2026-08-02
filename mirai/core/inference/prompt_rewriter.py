"""Provider-neutral inference prompt rewriting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mirai.core.registry import Registry


@dataclass(frozen=True)
class PromptRewriteRequest:
    prompt: str
    negative_prompt: str = ""


@dataclass(frozen=True)
class PromptRewriteResult:
    prompt: str
    negative_prompt: str


PromptRewriter = Callable[[PromptRewriteRequest], PromptRewriteResult]
PromptRewriterRegistry: Registry[PromptRewriter] = Registry("inference prompt rewriter")


def register_prompt_rewriter(name: str):
    """Register a named prompt rewriter implementation."""

    def decorator(rewriter: PromptRewriter) -> PromptRewriter:
        PromptRewriterRegistry.register(name, rewriter)
        return rewriter

    return decorator


def rewrite_inference_prompts(
    name: str,
    request: PromptRewriteRequest,
) -> PromptRewriteResult:
    """Rewrite a prompt pair, preserving exact strings when disabled."""
    key = str(name or "none").strip().lower()
    if key == "none":
        return PromptRewriteResult(request.prompt, request.negative_prompt)
    return PromptRewriterRegistry.get(key)(request)
