"""Executable release contracts for the native Agnes runtime."""

from .report import validate_inference_report, validate_train_report

__all__ = ["validate_inference_report", "validate_train_report"]
