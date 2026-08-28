"""Choose the smallest diverse PhenoCam negative supplement."""

from pathlib import Path

import numpy as np

from ..common import DatasetError, stable_rank


def identity(row):
    return f"{row['source_dataset']}::{row['source_id']}"


def diverse(candidates, initial, embeddings_path, count, seed, identity_function=identity):
    with np.load(embeddings_path, allow_pickle=False) as archive:
        identities = archive["identities"].astype(str)
        embeddings = archive["embeddings"].astype(np.float32)
    indexes = {value: index for index, value in enumerate(identities)}
    candidate_ids = [identity_function(row) for row in candidates]
    initial_ids = [identity_function(row) for row in initial]
    if any(value not in indexes for value in (*candidate_ids, *initial_ids)):
        raise DatasetError("global embeddings do not cover final PhenoCam negatives")
    candidate_embeddings = embeddings[[indexes[value] for value in candidate_ids]]
    initial_embeddings = embeddings[[indexes[value] for value in initial_ids]]
    maximum_similarity = np.max(candidate_embeddings @ initial_embeddings.T, axis=1)
    selected = []
    selected_set = set()
    stable = [stable_rank(seed, "final-negative", value) for value in candidate_ids]
    while len(selected) < count:
        eligible = (index for index in range(len(candidates)) if index not in selected_set)
        chosen = min(eligible, key=lambda index: (maximum_similarity[index], stable[index]))
        selected.append(chosen)
        selected_set.add(chosen)
        maximum_similarity = np.maximum(
            maximum_similarity, candidate_embeddings @ candidate_embeddings[chosen]
        )
    return [candidates[index] for index in selected]


def select(rows, imported, embeddings_path, target, seed):
    """Reuse every CVAT-confirmed empty frame, then diversify the remainder."""
    by_identity = {identity(row): row for row in rows}
    prior_receipts = [row for row in imported if int(row["annotation_count"]) == 0]
    prior = [by_identity[row["source_identity"]] for row in prior_receipts]
    candidates = [row for row in rows if row["provisional_role"] == "negative"]
    selected_new = diverse(
        candidates, prior, Path(embeddings_path), target - len(prior), seed
    )
    final = sorted((*prior, *selected_new), key=identity)
    if len(final) != target:
        raise DatasetError("final PhenoCam negative selection has an invalid size")
    return prior_receipts, prior, selected_new, final
