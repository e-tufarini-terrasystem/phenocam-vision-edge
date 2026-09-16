"""Embeddings: pairs responsibility extracted without changing the data contract."""

import heapq
import itertools
from collections import defaultdict
import numpy as np
from ..common import DatasetError, hamming64, stable_rank, write_csv
from .manifest import CALIBRATION_FIELDS, DUPLICATE_PAIR_FIELDS, _identity, _load_downloads


def _pairs_for_equal(rows, field):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[field]].append(index)
    return {
        pair
        for indexes in groups.values()
        if indexes and len(indexes) > 1
        for pair in itertools.combinations(indexes, 2)
    }


def duplicate_pairs(download_manifest, embeddings_path, output_pairs, calibration_path, config):
    rows = _load_downloads(download_manifest)
    minimum = int(config["deduplication"]["calibration_pairs_minimum"])
    # Distinct unordered pairs are finite; reject impossible quotas before sampling.
    if len(rows) * (len(rows) - 1) // 2 < minimum:
        raise DatasetError("not enough distinct pairs for SSCD calibration")
    with np.load(embeddings_path, allow_pickle=False) as archive:
        identities = archive["identities"].astype(str)
        embeddings = archive["embeddings"].astype(np.float32)
        model_identity = str(archive["model_identity"])
    expected = np.asarray([_identity(row) for row in rows])
    if not np.array_equal(identities, expected) or embeddings.shape[0] != len(rows):
        raise DatasetError("embedding rows do not match the download manifest")
    configured = config["deduplication"]["embedding_model"]
    expected_model = f"{configured['identifier']}@sha256:{configured['sha256']}"
    if model_identity != expected_model:
        raise DatasetError("embedding model identity mismatch")

    signals = defaultdict(set)
    for pair in _pairs_for_equal(rows, "source_sha256"):
        signals[pair].add("source_sha256")
    for pair in _pairs_for_equal(rows, "decoded_sha256"):
        signals[pair].add("decoded_sha256")

    phash_limit = int(config["deduplication"]["phash_hamming_max"])
    phashes = [row["phash"] for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if hamming64(phashes[left], phashes[right]) <= phash_limit:
                signals[(left, right)].add("phash")

    review_min = float(config["deduplication"]["embedding_review_cosine_min"])
    auto_min = float(config["deduplication"]["embedding_auto_cosine_min"])
    similarities = embeddings @ embeddings.T
    for left in range(len(rows)):
        for right in np.flatnonzero(similarities[left, left + 1 :] >= review_min):
            right = left + 1 + int(right)
            signals[(left, right)].add(
                "sscd_auto" if similarities[left, right] >= auto_min else "sscd_review"
            )

    pair_rows = []
    for (left, right), pair_signals in sorted(signals.items()):
        pair_rows.append(
            {
                "left_identity": identities[left],
                "right_identity": identities[right],
                "left_path": rows[left]["local_path"],
                "right_path": rows[right]["local_path"],
                "signals": ";".join(sorted(pair_signals)),
                "phash_hamming": hamming64(phashes[left], phashes[right]),
                "embedding_cosine": f"{float(similarities[left, right]):.8f}",
                "review_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "decision": "",
                "note": "",
            }
        )
    write_csv(output_pairs, DUPLICATE_PAIR_FIELDS, pair_rows)

    bands = (
        ("auto_threshold", 0.98, 1.01, 80),
        ("review_threshold", 0.95, 0.98, 80),
        ("near_control", 0.85, 0.95, 40),
    )
    calibration = []
    used = set()
    seed = config["seed"]
    for band, lower, upper, limit in bands:
        choices = []
        for left in range(len(rows)):
            indexes = np.flatnonzero(
                (similarities[left, left + 1 :] >= lower)
                & (similarities[left, left + 1 :] < upper)
            )
            for offset in indexes:
                right = left + 1 + int(offset)
                choices.append((left, right))
        choices.sort(key=lambda pair: stable_rank(seed, band, identities[pair[0]], identities[pair[1]]))
        for left, right in choices[:limit]:
            used.add((left, right))
            calibration.append(
                {
                    "left_identity": identities[left],
                    "right_identity": identities[right],
                    "left_path": rows[left]["local_path"],
                    "right_path": rows[right]["local_path"],
                    "embedding_cosine": f"{float(similarities[left, right]):.8f}",
                    "similarity_band": band,
                    "review_status": "pending",
                    "reviewer": "",
                    "reviewed_at": "",
                    "is_copy": "",
                    "note": "",
                }
            )
    top_heap = []
    top_limit = max(minimum, 200)
    top_target = min(minimum, max(len(calibration), minimum // 2))
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            entry = (float(similarities[left, right]), left, right)
            if len(top_heap) < top_limit:
                heapq.heappush(top_heap, entry)
            elif entry[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, entry)
    for similarity, left, right in sorted(top_heap, reverse=True):
        if len(calibration) >= top_target:
            break
        if (left, right) in used:
            continue
        used.add((left, right))
        calibration.append(
            {
                "left_identity": identities[left],
                "right_identity": identities[right],
                "left_path": rows[left]["local_path"],
                "right_path": rows[right]["local_path"],
                "embedding_cosine": f"{similarity:.8f}",
                "similarity_band": "natural_top_similarity",
                "review_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "is_copy": "",
                "note": "",
            }
        )
    generator = np.random.default_rng(seed)
    while len(calibration) < minimum and len(rows) > 1:
        left, right = sorted(generator.choice(len(rows), size=2, replace=False).tolist())
        if (left, right) in used:
            continue
        used.add((left, right))
        calibration.append(
            {
                "left_identity": identities[left],
                "right_identity": identities[right],
                "left_path": rows[left]["local_path"],
                "right_path": rows[right]["local_path"],
                "embedding_cosine": f"{float(similarities[left, right]):.8f}",
                "similarity_band": "deterministic_control",
                "review_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "is_copy": "",
                "note": "",
            }
        )
    if len(calibration) < minimum:
        raise DatasetError(
            f"only {len(calibration)} SSCD calibration pairs available; {minimum} required"
        )
    write_csv(calibration_path, CALIBRATION_FIELDS, calibration[:minimum])
    return {"duplicate_candidates": len(pair_rows), "calibration_pairs": minimum}
