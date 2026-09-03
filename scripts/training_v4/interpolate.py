"""Create the fixed interpolation family between original and best fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v4"
BASE = ROOT / "models/yolo26n.pt"
ALPHAS = (0.10, 0.25, 0.50, 0.75)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--alpha", type=float, choices=ALPHAS)
    arguments = parser.parse_args()
    parent = arguments.parent.resolve()
    allowed = (WORK / "runs").resolve()
    if not parent.is_file() or not parent.is_relative_to(allowed) or not arguments.name.replace("-", "").isalnum():
        raise SystemExit("parent must be a recorded v4 fine-tuning checkpoint")
    base, fine = YOLO(BASE), YOLO(parent)
    if dict(base.names) != dict(fine.names):
        raise RuntimeError("interpolation parents have different class mappings")
    base_state, fine_state = base.model.float().state_dict(), fine.model.float().state_dict()
    if base_state.keys() != fine_state.keys():
        raise RuntimeError("interpolation parents have different architectures")
    alphas = (arguments.alpha,) if arguments.alpha is not None else ALPHAS
    summary = {
        "method": "linear weight interpolation", "selection_split": "val", "alphas": alphas,
        "base": str(BASE.relative_to(ROOT)), "base_sha256": _sha256(BASE),
        "fine_tuned": str(parent.relative_to(ROOT)), "fine_tuned_sha256": _sha256(parent), "candidates": [],
    }
    for alpha in alphas:
        name = f"{arguments.name}-alpha-{int(alpha * 100):02d}"
        weights = WORK / "interpolation" / name / "weights"
        if weights.parent.exists():
            raise SystemExit(f"{name} exists; refusing to overwrite it")
        weights.mkdir(parents=True)
        state = {
            key: base_value * (1.0 - alpha) + fine_state[key] * alpha
            if base_value.is_floating_point() else fine_state[key]
            for key, base_value in base_state.items()
        }
        candidate = YOLO(BASE)
        candidate.model.float().load_state_dict(state, strict=True)
        checkpoint = weights / "best.pt"
        candidate.save(checkpoint)
        row = {"name": name, "alpha": alpha, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": _sha256(checkpoint)}
        summary["candidates"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    path = WORK / "interpolation" / f"{arguments.name}-summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
