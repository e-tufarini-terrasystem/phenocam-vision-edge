"""Embeddings entry points; implementation is grouped by responsibility."""

from .manifest import DEDUP_MANIFEST_FIELDS, EMBEDDING_FIELDS, DUPLICATE_PAIR_FIELDS, CALIBRATION_FIELDS, _load_downloads, _identity, combine_manifests
from .model import _atomic_npz, _preprocess, compute_embeddings
from .pairs import _pairs_for_equal, duplicate_pairs
