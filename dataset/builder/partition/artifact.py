"""Materialize the approved partition atomically, using hard links when possible."""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

from ..common import DatasetError, sha256_file
from .allocation import PHASH_THRESHOLD, SEED, allocate
from .records import CLASS_NAMES, inventory, load_records, phenocam_inventory
from .report import render_report, split_statistics
from .verification import verify_artifact


PATHS = {
    "train": "train",
    "val": "val",
    "test_id": "test/id",
    "test_ood": "test/ood",
}
YAML = """train: images/train
val: images/val
test:
  - images/test/id
  - images/test/ood
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: __unused_class_4
  5: bus
  6: __unused_class_6
  7: truck
"""


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public_row(record, split):
    row = {key: value for key, value in record.items() if not key.startswith("_")}
    row["source_image_path"] = record["_source_image_path"]
    row["source_label_path"] = record["_source_label_path"]
    row["partition_group_id"] = record["_partition_group_id"]
    row["classes_present"] = ";".join(sorted(CLASS_NAMES[class_id] for class_id in record["_classes"]))
    row["annotation_count"] = str(sum(record["_classes"].values()))
    row["split"] = split
    name = Path(record["_source_image_path"]).name
    row["image_path"] = f"images/{PATHS[split]}/{name}"
    row["label_path"] = f"labels/{PATHS[split]}/{Path(name).stem}.txt" if record["_source_label_path"] else ""
    return row


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_readme(root, statistics):
    values = statistics
    text = f"""# Phenocam Vision dataset v3

Artifact YOLO riproducibile con split leakage-aware.

- Train: {values['train']['images']} immagini / {values['train']['annotations']} annotazioni
- Validation: {values['val']['images']} immagini / {values['val']['annotations']} annotazioni
- Test ID: {values['test_id']['images']} immagini / {values['test_id']['annotations']} annotazioni
- Test OOD: {values['test_ood']['images']} immagini / {values['test_ood']['annotations']} annotazioni

`dataset.yaml` valuta per default il test combinato. `dataset-test-id.yaml` e
`dataset-test-ood.yaml` isolano i due protocolli. I manifest sono in
`manifests/`; metodologia e misure sono in `reports/dataset-analysis.md`.

Rigenerazione e verifica dalla directory `dataset/`:

```sh
.venv/bin/python -m builder.partition build dataset-v3 dataset-v3-rebuilt
.venv/bin/python -m builder.partition verify dataset-v3
```

Aprire `viewer.html`, scegliere questa directory e selezionare lo split.
I dati `internal` sono privati e non devono essere redistribuiti.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_yamls(root):
    (root / "dataset.yaml").write_text(YAML, encoding="utf-8")
    (root / "dataset-test-id.yaml").write_text(YAML.replace("test:\n  - images/test/id\n  - images/test/ood", "test: images/test/id"), encoding="utf-8")
    (root / "dataset-test-ood.yaml").write_text(YAML.replace("test:\n  - images/test/id\n  - images/test/ood", "test: images/test/ood"), encoding="utf-8")


def _write_checksums(root):
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "metadata" / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_artifact(source_root, destination):
    source = Path(source_root).resolve()
    destination = Path(destination).resolve()
    if source == destination or destination.exists():
        raise DatasetError("destination must be a new directory distinct from the source")
    records = load_records(source, measure_light=True)
    measured = inventory(records, source)
    phenocam = phenocam_inventory(records)
    assignments = allocate(records)
    statistics = split_statistics(assignments)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        rows_by_split, rows = {}, []
        for split, split_records in assignments.items():
            rows_by_split[split] = []
            for record in sorted(split_records, key=lambda item: item["image_id"]):
                row = _public_row(record, split)
                _link(source / record["_input_image_path"], temporary / row["image_path"])
                if row["label_path"]:
                    _link(source / record["_input_label_path"], temporary / row["label_path"])
                rows.append(row)
                rows_by_split[split].append(row)
        fields = list(rows[0])
        _write_csv(temporary / "metadata" / "source-images.csv", fields, sorted(rows, key=lambda row: row["image_id"]))
        for split, filename in (("train", "train.csv"), ("val", "val.csv"), ("test_id", "test-id.csv"), ("test_ood", "test-ood.csv")):
            _write_csv(temporary / "manifests" / filename, fields, rows_by_split[split])
        _write_csv(temporary / "manifests" / "test.csv", fields, rows_by_split["test_id"] + rows_by_split["test_ood"])

        metadata = temporary / "metadata"
        _link(source / "metadata" / "source-annotations.jsonl", metadata / "source-annotations.jsonl")
        source_build = _read_json(source / "metadata" / "build.json")
        source_acceptance = _read_json(source / "metadata" / "acceptance-audit.json")
        if source_build is not None:
            (metadata / "source-build.json").write_text(json.dumps(source_build, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if source_acceptance is not None:
            (metadata / "source-acceptance-audit.json").write_text(json.dumps(source_acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "seed": SEED,
            "strategy": "hybrid_grouped_camera_site_time_class",
            "phash_hamming_maximum": PHASH_THRESHOLD,
            "source_manifest_sha256": sha256_file(source / "metadata" / "source-images.csv"),
            "split_statistics": statistics,
        }
        (metadata / "build.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (metadata / "dataset-statistics.json").write_text(json.dumps({"original": measured, "phenocam": phenocam, "splits": statistics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        acceptance = {"status": "passed", "test_frozen": True, "test_id_role": "final_evaluation_only", "test_ood_role": "final_evaluation_only", "internal_training_allowed": False}
        (metadata / "acceptance-audit.json").write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_yamls(temporary)
        _write_readme(temporary, statistics)
        reports = temporary / "reports"
        reports.mkdir()
        (reports / "dataset-analysis.md").write_text(render_report(measured, phenocam, statistics), encoding="utf-8")
        verification = verify_artifact(temporary, check_checksums=False)
        (reports / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (reports / "dataset-analysis.md").write_text(render_report(measured, phenocam, statistics, verification), encoding="utf-8")
        _write_checksums(temporary)
        verify_artifact(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"artifact": str(destination), "statistics": statistics, "verification": verify_artifact(destination)}
