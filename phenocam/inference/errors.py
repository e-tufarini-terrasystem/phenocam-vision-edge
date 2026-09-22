"""Inference errors with fixed meanings; the CLI supplies messages and exit codes."""


class InferenceError(RuntimeError):
    """Represent a model, image, runtime, or internal inference failure."""


class OutputWriteError(RuntimeError):
    """Represent failure to create a non-empty regular output image."""
