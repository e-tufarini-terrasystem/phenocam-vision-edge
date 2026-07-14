"""Define the two inference-domain errors that cross into the CLI boundary.

The errors carry only fixed, non-sensitive meanings; ``run.py`` owns their
user-facing messages and process status handling.
"""


class InferenceError(RuntimeError):
    """Represent a model, image, runtime, or internal inference failure."""


class OutputWriteError(RuntimeError):
    """Represent failure to create a non-empty regular output image."""
