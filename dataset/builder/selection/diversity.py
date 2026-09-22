"""Select diverse candidates with deterministic ties and optional group quotas."""

from collections import Counter
import numpy as np
from ..common import DatasetError, stable_rank
from ..dedup import DEDUP_FIELDS


SELECTION_FIELDS = DEDUP_FIELDS + (
    "primary_stratum",
    "size_tags",
    "small_target",
    "occluded_or_truncated",
    "multiple_targets",
    "easy",
    "selection_status",
)


_RARE_CLASSES = {"bicycle", "motorcycle", "bus", "truck"}


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


class _FarthestSelector:
    """Greedily minimize similarity to the nearest selected unit embedding."""

    def __init__(
        self,
        rows,
        embeddings,
        seed,
        group_keys=None,
        group_limit=None,
        initial_group_counts=None,
    ):
        self.rows = rows
        self.embeddings = embeddings
        self.seed = seed
        self.selected = []
        self.selected_set = set()
        self.maximum_similarity = np.full(len(rows), -np.inf, dtype=np.float32)
        self.stable = np.asarray(
            [stable_rank(seed, _identity(row)) for row in rows], dtype=object
        )
        self.group_keys = group_keys
        self.group_limit = group_limit
        self.group_counts = Counter(initial_group_counts or {})

    def choose(self, eligible):
        indexes = [index for index in eligible if index not in self.selected_set]
        if self.group_keys is not None and self.group_limit is not None:
            indexes = [
                index
                for index in indexes
                if self.group_counts[self.group_keys[index]] < self.group_limit
            ]
        if not indexes:
            raise DatasetError("no candidate remains for an unsatisfied provisional quota")
        if not self.selected:
            chosen = min(indexes, key=lambda index: self.stable[index])
        else:
            chosen = min(
                indexes,
                key=lambda index: (self.maximum_similarity[index], self.stable[index]),
            )
        self.selected.append(chosen)
        self.selected_set.add(chosen)
        if self.group_keys is not None:
            self.group_counts[self.group_keys[chosen]] += 1
        # With unit embeddings, the largest dot product is the nearest selected image.
        similarities = self.embeddings @ self.embeddings[chosen]
        self.maximum_similarity = np.maximum(self.maximum_similarity, similarities)
        return chosen
