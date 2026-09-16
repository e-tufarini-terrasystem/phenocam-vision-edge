"""Verify immutable experiment inputs and snapshot the actual dirty worktree."""

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.training_v5.selection import sha256

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/training-v6"
DATASET = ROOT / "dataset/dataset-v6"
CONFIG = Path(__file__).with_name("experiments.json")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def code_hashes():
    paths = [p for directory in ("phenocam", "scripts", "dataset/builder", "tests")
             for p in (ROOT / directory).rglob("*") if p.suffix in {".py", ".json"}]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify_dataset():
    import yaml
    from ultralytics import YOLO
    from scripts.training_v5.dataset import image_properties
    from .sources import check_separation, local_path, reviewed_items

    config = json.loads(CONFIG.read_text())
    checksums = (DATASET / "metadata/checksums.sha256").read_text().splitlines()
    fingerprint_lines = []
    for line in checksums:
        digest, name = line.split("  ", 1)
        if sha256(local_path(DATASET, name)) != digest:
            raise RuntimeError("Dataset checksum changed")
        if name != "metadata/audit.json":
            fingerprint_lines.append(line + "\n")
    fingerprint = hashlib.sha256("".join(fingerprint_lines).encode()).hexdigest()
    if fingerprint != config["fingerprint"]:
        raise RuntimeError("Unexpected v6 fingerprint")
    data = yaml.safe_load((DATASET / "dataset.yaml").read_text())
    names = YOLO(ROOT / config["base"]).names
    if (data["names"] != names or len(names) != 80 or data["train"] != "images/train"
            or data["val"] != "images/val" or Path(data["path"]).resolve() != DATASET
            or "test" in data or "download" in data):
        raise RuntimeError("Dataset class or split contract changed")
    rows = list(csv.DictReader((DATASET / "metadata/source-images.csv").open()))
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 2020, "val": 206, "pklot_holdout": 6}:
        raise RuntimeError("Unexpected split counts")
    groups, pixels, labels, image_names = defaultdict(set), set(), 0, set()
    for index, row in enumerate(rows):
        image = local_path(DATASET, row["image_path"])
        _, decoded, _ = image_properties(image)
        if decoded != row["decoded_sha256"] or decoded in pixels or sha256(image) != row["compiled_sha256"]:
            raise RuntimeError("Image identity or decoded pixels differ")
        pixels.add(decoded)
        image_names.add(row["image_path"])
        groups[row["group_id"] or row["source_identity"].split("#")[0]].add(row["split"])
        label = local_path(DATASET, row["label_path"]) if row["label_path"] else None
        lines = label.read_text().splitlines() if label else []
        if len(lines) != int(row["annotation_count"]):
            raise RuntimeError("Annotation count differs")
        if not label and (DATASET / row["image_path"].replace("images/", "labels/")).with_suffix(".txt").exists():
            raise RuntimeError("Negative image has an untracked label")
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError("Malformed label")
            cls, x, y, w, h = map(float, fields)
            if (not all(math.isfinite(v) for v in (cls, x, y, w, h))
                    or not cls.is_integer() or int(cls) not in names or w <= 0 or h <= 0
                    or min(x - w / 2, y - h / 2) < -1e-7
                    or max(x + w / 2, y + h / 2) > 1 + 1e-7):
                raise RuntimeError("Invalid label geometry or class")
        labels += len(lines)
        if index % 250 == 0:
            print(f"Verified {index}/{len(rows)} images", flush=True)
    actual = {str(p.relative_to(DATASET)) for p in (DATASET / "images").rglob("*") if p.is_file()}
    if actual != image_names or any(len(s) != 1 for s in groups.values()) or labels != 7119:
        raise RuntimeError("Unmanifested images, split overlap or box count mismatch")
    base = ROOT / "dataset/dataset-v5"
    inherited = list(csv.DictReader((base / "metadata/source-images.csv").open()))
    by_id = {r["image_id"]: r for r in rows}
    for row in inherited:
        if by_id.get(row["image_id"]) != row:
            raise RuntimeError("Inherited v5 manifest changed")
        for field in ("image_path", "label_path"):
            if row[field] and sha256(base / row[field]) != sha256(DATASET / row[field]):
                raise RuntimeError("Inherited v5 content changed")
    settings = json.loads((ROOT / "dataset/config/training-v6.json").read_text())
    for name, digest in settings["source_sha256"].items():
        if sha256(local_path(ROOT, name)) != digest:
            raise RuntimeError("Pinned annotation source changed")
    additions, _ = reviewed_items(settings)
    canonical = ROOT / "dataset/dataset-v3"
    check_separation(inherited, additions, canonical)
    tests = [r for r in csv.DictReader((canonical / "metadata/source-images.csv").open())
             if r["split"].startswith("test")]
    # Metadata-only leakage checks do not inspect predictions or tune on tests.
    for key in ("source_identity", "source_sha256", "decoded_sha256", "group_id"):
        known = {r[key] for r in tests if r.get(key)}
        values = {r[key].split("#")[0] for r in rows if r.get(key)}
        if known & values:
            raise RuntimeError("Canonical test contamination")
    if {r["parent_source_sha256"] for r in rows} & {r["source_sha256"] for r in tests}:
        raise RuntimeError("Test parent reused by a training crop")
    return {"status": "passed", "fingerprint": fingerprint, "splits": dict(counts),
            "labels": labels, "checksums_verified": len(checksums), "historical_test_rows_screened": len(tests),
            "classes": names, "inherited_v5_images_verified": len(inherited)}


def main():
    import torch

    if (WORK / "preflight.json").exists():
        raise SystemExit("Preflight exists; preserve its original provenance")
    WORK.mkdir(parents=True, exist_ok=True)
    report = verify_dataset()
    hashes = code_hashes()
    for name in hashes:
        destination = WORK / "provenance/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (WORK / "provenance/worktree.patch").write_bytes(diff)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (WORK / "provenance/environment.txt").write_text(freeze)
    report.update({"code_sha256": hashes, "config_sha256": sha256(CONFIG),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "worktree_status": subprocess.check_output(["git", "status", "--short"], text=True),
                   "python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "mps": torch.backends.mps.is_available(),
                   "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                   "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
                   "environment_sha256": sha256(WORK / "provenance/environment.txt"),
                   "model_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "models").glob("*") if p.is_file()}})
    write(WORK / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"code_sha256", "classes", "model_hashes"}}, indent=2))


if __name__ == "__main__":
    main()
