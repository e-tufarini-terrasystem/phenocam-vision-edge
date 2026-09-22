"""Compute normalized SSCD embeddings with a checksum-verified model."""

import json
import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ..common import DatasetError, sha256_file, write_csv
from .manifest import EMBEDDING_FIELDS, _identity, _load_downloads


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
