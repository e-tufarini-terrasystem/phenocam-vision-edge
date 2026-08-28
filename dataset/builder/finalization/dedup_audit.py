"""Run exact and SSCD checks over the final accepted composition."""

from collections import Counter
from pathlib import Path

import numpy as np

from ..common import DatasetError, hamming64


def audit(records, dataset_root, config):
    for field in ("source_sha256", "decoded_sha256"):
        values = [record["row"][field] for record in records]
        if len(set(values)) != len(values):
            raise DatasetError(f"final composition contains duplicate {field}")
    selected = {record["source_identity"] for record in records}
    embeddings = {}
    identities_by_file = (
        "work/global/embeddings.npz",
        "work/openimages/embeddings-small-supplement.npz",
        "work/openimages/embeddings.npz",
    )
    model_identities = set()
    for relative in identities_by_file:
        archive = np.load(Path(dataset_root) / relative, allow_pickle=False)
        model_identities.add(str(archive["model_identity"]))
        mask = np.isin(archive["identities"], list(selected))
        embeddings.update(
            (str(identity), vector)
            for identity, vector in zip(archive["identities"][mask], archive["embeddings"][mask])
        )
    if set(embeddings) != selected or len(model_identities) != 1:
        raise DatasetError("final SSCD audit has missing or incompatible embeddings")
    ordered = sorted(selected)
    matrix = np.stack([embeddings[identity] for identity in ordered])
    threshold = float(config["deduplication"]["embedding_review_cosine_min"])
    suspicious = []
    for start in range(0, len(ordered), 128):
        similarities = matrix[start : start + 128] @ matrix.T
        for left_offset, right in np.argwhere(similarities >= threshold):
            left = start + int(left_offset)
            if left < int(right):
                suspicious.append((ordered[left], ordered[int(right)], float(similarities[left_offset, right])))
    if suspicious:
        raise DatasetError("final composition contains unresolved SSCD duplicate candidates")
    phashes = [(record["source_identity"], record["row"]["phash"]) for record in records]
    phash_candidates = sum(hamming64(left_hash, right_hash) <= int(config["deduplication"]["phash_hamming_max"]) for index, (_, left_hash) in enumerate(phashes) for _, right_hash in phashes[index + 1 :])
    return {
        "decoded_duplicates": 0,
        "embedding_model": next(iter(model_identities)),
        "phash_candidates": phash_candidates,
        "source_byte_duplicates": 0,
        "sscd_candidates_at_0_95": 0,
    }
