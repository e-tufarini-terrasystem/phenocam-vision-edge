"""Create a fixed interpolation grid from recorded v5 checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v5"
BASE = ROOT / "models/yolo26n.pt"
ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.50, 0.75)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    # Keep the workstation dependency optional when importing the fixed grid.
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, action="append", required=True)
    parser.add_argument("--name", required=True)
    arguments = parser.parse_args()
    parents = [path.resolve() for path in arguments.parent]
    runs = (WORK / "runs").resolve()
    if (
        any(not path.is_file() or not path.is_relative_to(runs) for path in parents)
        or len(set(parents)) != len(parents)
        or not arguments.name.replace("-", "").isalnum()
    ):
        raise SystemExit("parents must be unique recorded v5 checkpoints")

    base = YOLO(BASE)
    fine_tuned = [YOLO(path) for path in parents]
    if any(dict(base.names) != dict(model.names) for model in fine_tuned):
        raise RuntimeError("interpolation parents have different class mappings")
    base_state = base.model.float().state_dict()
    fine_states = [model.model.float().state_dict() for model in fine_tuned]
    if any(base_state.keys() != state.keys() for state in fine_states):
        raise RuntimeError("interpolation parents have different architectures")
    # All parents started from the same base. Uniform averaging aggregates seed
    # variance before the conservative base-to-adapted interpolation.
    fine_state = {
        key: sum(state[key] for state in fine_states) / len(fine_states)
        if base_value.is_floating_point()
        else fine_states[0][key]
        for key, base_value in base_state.items()
    }

    summary = {
        "method": "linear weight interpolation",
        "selection_split": "val",
        "alphas": ALPHAS,
        "base": str(BASE.relative_to(ROOT)),
        "base_sha256": sha256(BASE),
        "fine_tuned": [str(path.relative_to(ROOT)) for path in parents],
        "fine_tuned_sha256": [sha256(path) for path in parents],
        "candidates": [],
    }
    for alpha in ALPHAS:
        name = f"{arguments.name}-alpha-{int(alpha * 100):02d}"
        weights = WORK / "interpolation" / name / "weights"
        if weights.parent.exists():
            raise SystemExit(f"{name} exists; refusing to overwrite it")
        weights.mkdir(parents=True)
        # Non-floating buffers are copied from the adapted model; learned tensors
        # stay on the straight path between the exact recorded parents.
        state = {
            key: base_value * (1.0 - alpha) + fine_state[key] * alpha
            if base_value.is_floating_point()
            else fine_state[key]
            for key, base_value in base_state.items()
        }
        candidate = YOLO(BASE)
        candidate.model.float().load_state_dict(state, strict=True)
        checkpoint = weights / "best.pt"
        candidate.save(checkpoint)
        row = {
            "name": name,
            "alpha": alpha,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
        }
        summary["candidates"].append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    destination = WORK / "interpolation" / f"{arguments.name}-summary.json"
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
