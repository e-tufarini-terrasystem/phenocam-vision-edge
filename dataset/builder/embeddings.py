"""Pinned SSCD descriptors and duplicate-review pair generation."""

import csv
import heapq
import itertools
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import DatasetError, hamming64, require_columns, sha256_file, stable_rank, write_csv


DEDUP_MANIFEST_FIELDS = (
    "source_dataset",
    "source_subset",
    "source_id",
    "local_path",
    "source_rotation_ccw",
    "source_sha256",
    "decoded_sha256",
    "phash",
)


EMBEDDING_FIELDS = (
    "row_index",
    "source_identity",
    "local_path",
    "embedding_model",
)

DUPLICATE_PAIR_FIELDS = (
    "left_identity",
    "right_identity",
    "left_path",
    "right_path",
    "signals",
    "phash_hamming",
    "embedding_cosine",
    "review_status",
    "reviewer",
    "reviewed_at",
    "decision",
    "note",
)

CALIBRATION_FIELDS = (
    "left_identity",
    "right_identity",
    "left_path",
    "right_path",
    "embedding_cosine",
    "similarity_band",
    "review_status",
    "reviewer",
    "reviewed_at",
    "is_copy",
    "note",
)


def _load_downloads(path):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, DEDUP_MANIFEST_FIELDS, path)
        return list(reader)


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def combine_manifests(input_paths, output_path):
    combined = []
    identities = set()
    for input_path in input_paths:
        with Path(input_path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = (
                "source_dataset",
                "source_id",
                "local_path",
                "source_sha256",
                "decoded_sha256",
                "phash",
            )
            require_columns(reader, required, input_path)
            for row in reader:
                output = {
                    "source_dataset": row["source_dataset"],
                    "source_subset": row.get("source_subset", ""),
                    "source_id": row["source_id"],
                    "local_path": row["local_path"],
                    "source_rotation_ccw": row.get("source_rotation_ccw", ""),
                    "source_sha256": row["source_sha256"],
                    "decoded_sha256": row["decoded_sha256"],
                    "phash": row["phash"],
                }
                identity = _identity(output)
                if identity in identities:
                    raise DatasetError(f"duplicate source identity in dedup manifest: {identity}")
                if not Path(output["local_path"]).is_file():
                    raise DatasetError(f"dedup source image is missing: {identity}")
                identities.add(identity)
                combined.append(output)
    combined.sort(key=_identity)
    write_csv(output_path, DEDUP_MANIFEST_FIELDS, combined)
    return {
        "image_count": len(combined),
        "source_counts": {
            source: sum(row["source_dataset"] == source for row in combined)
            for source in sorted({row["source_dataset"] for row in combined})
        },
    }


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _preprocess(path, model_config, rotation_ccw=0):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if rotation_ccw:
            image = image.rotate(int(rotation_ccw), expand=True)
        image = image.resize(
            (int(model_config["input_width"]), int(model_config["input_height"])),
            Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(model_config["mean"], dtype=np.float32)
    standard_deviation = np.asarray(
        model_config["standard_deviation"], dtype=np.float32
    )
    normalized = (pixels - mean) / standard_deviation
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def compute_embeddings(download_manifest, model_path, output_npz, output_manifest, config, batch_size=16):
    try:
        import torch
    except ImportError as error:
        raise DatasetError("install dataset/requirements.txt before computing embeddings") from error

    rows = _load_downloads(download_manifest)
    model_config = config["deduplication"]["embedding_model"]
    if sha256_file(model_path) != model_config["sha256"]:
        raise DatasetError("SSCD model checksum mismatch")
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    matrices = []
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, int(batch_size))):
            batch_rows = rows[start : start + max(1, int(batch_size))]
            batch = np.stack(
                [
                    _preprocess(
                        row["local_path"],
                        model_config,
                        int(row["source_rotation_ccw"] or 0),
                    )
                    for row in batch_rows
                ]
            )
            result = model(torch.from_numpy(batch)).detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            if np.any(norms <= 0) or not np.all(np.isfinite(result)):
                raise DatasetError("SSCD emitted an invalid descriptor")
            matrices.append(result / norms)
    matrix = np.concatenate(matrices, axis=0) if matrices else np.empty((0, 512), np.float32)
    identities = np.asarray([_identity(row) for row in rows])
    model_identity = f"{model_config['identifier']}@sha256:{model_config['sha256']}"
    _atomic_npz(
        output_npz,
        identities=identities,
        embeddings=matrix,
        model_identity=np.asarray(model_identity),
        preprocessing=np.asarray(json.dumps(model_config, sort_keys=True)),
    )
    write_csv(
        output_manifest,
        EMBEDDING_FIELDS,
        (
            {
                "row_index": index,
                "source_identity": identity,
                "local_path": row["local_path"],
                "embedding_model": model_identity,
            }
            for index, (identity, row) in enumerate(zip(identities, rows))
        ),
    )
    return {"image_count": len(rows), "dimensions": int(matrix.shape[1]) if len(rows) else 0}


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
    minimum = int(config["deduplication"]["calibration_pairs_minimum"])
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
