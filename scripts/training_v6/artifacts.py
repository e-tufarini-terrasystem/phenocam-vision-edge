"""Export candidate copies, verify real-image contracts, and deliver without promotion."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from ultralytics import YOLO

from phenocam.inference.pipeline import process_image
from phenocam.inference.runtime import create_session, model_contract, run_tensor
from phenocam.inference.views import iter_views, load_image
from scripts.runtime_evaluation.metrics import iou
from scripts.runtime_evaluation.prediction import Predictor
from .evaluation import guard_runtime
from .preflight import CONFIG, DATASET, ROOT, WORK, sha256, write


def export(checkpoint, name):
    guard_runtime()
    if not name.replace("-", "").isalnum():
        raise RuntimeError("Invalid export name")
    folder = WORK / "exports" / name
    receipt = folder / "export.json"
    if receipt.exists():
        report = json.loads(receipt.read_text())
        if (report["pt_sha256"] != sha256(checkpoint) or report["onnx_sha256"] != sha256(folder / "model.onnx")):
            raise RuntimeError("Existing export changed")
        return folder / "model.onnx"
    if folder.exists():
        raise RuntimeError("Partial export exists; inspect before retrying")
    folder.mkdir(parents=True)
    destination = folder / "model.pt"
    shutil.copy2(checkpoint, destination)
    started = time.monotonic()
    exported = Path(YOLO(destination).export(format="onnx", opset=20, imgsz=640,
                                           batch=1, dynamic=False, simplify=True, device="cpu"))
    if exported.resolve() != destination.with_suffix(".onnx").resolve():
        raise RuntimeError("Unexpected export destination")
    onnx.checker.check_model(onnx.load(exported))
    session = create_session(exported)
    input_name, output_name, width, height, names = model_contract(session)
    rows, _ = run_tensor(session, input_name, output_name, np.zeros((1, 3, 640, 640), dtype=np.float32))
    if ((width, height) != (640, 640) or rows.shape != (300, 6) or not np.isfinite(rows).all()
            or names != YOLO(ROOT / "models/yolo26n.pt").names
            or session.get_inputs()[0].shape != [1, 3, 640, 640]
            or session.get_outputs()[0].shape != [1, 300, 6]):
        raise RuntimeError("Export contract violation")
    write(receipt, {"pt_sha256": sha256(destination), "onnx_sha256": sha256(exported),
                    "source": str(checkpoint.relative_to(ROOT)), "input": [1, 3, 640, 640],
                    "output": [1, 300, 6], "classes": names, "opset": 20,
                    "elapsed_seconds": time.monotonic() - started, "finite": True})
    return exported


def verify():
    guard_runtime()
    frozen = json.loads((WORK / "frozen-selection.json").read_text())
    pt, model = ROOT / frozen["checkpoint"], ROOT / frozen["onnx"]
    if sha256(pt) != frozen["model_sha256"] or sha256(model) != frozen["onnx_sha256"]:
        raise RuntimeError("Frozen artifacts changed")
    output = WORK / "verification"
    if output.exists():
        raise RuntimeError("Verification exists; preserve its evidence")
    output.mkdir()
    session = create_session(model)
    input_name, output_name, _, _, _ = model_contract(session)
    predictor = Predictor(pt, device="cpu")
    manifest = list(csv.DictReader((DATASET / "metadata/source-images.csv").open()))
    selected = [next(r for r in manifest if r["split"] == "val" and r["source_dataset"] == s
                     and int(r["annotation_count"]) > 0) for s in ("open_images", "phenocam", "pklot")]
    evidence = []
    for source in selected:
        image_path = DATASET / source["image_path"]
        image = load_image(image_path)
        overlaps, differences, total = [], [], 0
        for view in iter_views(image, 640, 640):
            ort_rows, _ = run_tensor(session, input_name, output_name, view.tensor)
            pt_rows, _ = predictor._run(view.tensor)
            if not np.isfinite(ort_rows).all() or not np.isfinite(pt_rows).all():
                raise RuntimeError("Nonfinite real-image predictions")
            # Match by geometry/class; top-k rank ties need not preserve row order.
            left = [r for r in pt_rows if r[4] >= 0.2]
            right = [r for r in ort_rows if r[4] >= 0.2]
            if len(left) != len(right):
                raise RuntimeError("PT/ONNX real-image detection counts differ")
            for row in left:
                candidates = [(iou(row[:4], other[:4]), j) for j, other in enumerate(right) if row[5] == other[5]]
                overlap, index = max(candidates, default=(0, -1))
                if overlap < 0.99:
                    raise RuntimeError("PT/ONNX geometry mismatch")
                other = right.pop(index)
                overlaps.append(overlap)
                differences.append(abs(float(row[4]) - float(other[4])))
            total += 1
        if total != 16 or max(differences, default=0) > 0.001:
            raise RuntimeError("PT/ONNX parity or view count failed")
        # Also exercise the public transaction, including rendered products.
        annotated = output / f"{source['source_dataset']}-annotated.jpg"
        privacy = output / f"{source['source_dataset']}-privacy.jpg"
        elapsed = process_image(model, image_path, annotated, privacy)
        if load_image(annotated).size != image.size or load_image(privacy).size != image.size:
            raise RuntimeError("Production rendering dimensions changed")
        evidence.append({"image": source["image_path"], "image_sha256": sha256(image_path),
                         "views": total, "matched_detections": len(overlaps), "minimum_iou": float(min(overlaps)) if overlaps else None,
                         "maximum_confidence_difference": max(differences, default=0),
                         "production_inference_seconds": elapsed})
    write(output / "contract.json", {"status": "passed", "pt_sha256": sha256(pt), "onnx_sha256": sha256(model),
          "images": evidence, "comparison_floor": 0.2, "finite_all_rows": True,
          "production_smoke_threshold": 0.47, "selected_evaluation_threshold": frozen["confidence"],
          "note": "Production API retains its operational threshold; evaluation passes the selected threshold explicitly."})


def deliver():
    guard_runtime()
    frozen = json.loads((WORK / "frozen-selection.json").read_text())
    verification = json.loads((WORK / "verification/contract.json").read_text())
    if verification["status"] != "passed" or verification["onnx_sha256"] != frozen["onnx_sha256"]:
        raise RuntimeError("Verified frozen export required")
    final = {}
    for split in ("pklot_holdout", "test_id", "test_ood"):
        final[split] = {}
        for name in ("base", "v5", "v6"):
            runtime = WORK / f"final/{split}/{name}-onnx/runtime/runtime-metrics.json"
            standard = WORK / f"final/{split}/{name}-pt/standard/standard-metrics.json"
            row, std = json.loads(runtime.read_text()), json.loads(standard.read_text())
            receipt = json.loads((WORK / f"frozen/{name}-onnx.json").read_text())
            pt_receipt = json.loads((WORK / f"frozen/{name}-pt.json").read_text())
            if (row["model_sha256"] != receipt["model_sha256"] or set(map(float, row["thresholds"])) != {receipt["confidence"]}
                    or std["model_sha256"] != pt_receipt["model_sha256"] or row["split"] != split or std["split"] != split):
                raise RuntimeError("Historical results differ from frozen inputs")
            final[split][name] = {"pipeline": row["thresholds"][str(receipt["confidence"])]["pipeline"], "standard": std["subsets"]}
    paths = [ROOT / f"models/yolo26n-v6.{suffix}" for suffix in ("pt", "onnx", "json")]
    pt_validation = json.loads((WORK / "evaluation/v6-pt-fixed/runtime/runtime-metrics.json").read_text())
    if pt_validation["model_sha256"] != frozen["model_sha256"] or pt_validation["images"] != 206:
        raise RuntimeError("Complete PT/ONNX validation comparison required")
    if any(p.exists() for p in paths):
        raise RuntimeError("Deliverables already exist; refusing replacement")
    for destination, key, hash_key in zip(paths[:2], ("checkpoint", "onnx"), ("model_sha256", "onnx_sha256")):
        source = ROOT / frozen[key]
        if sha256(source) != frozen[hash_key]:
            raise RuntimeError("Frozen source changed before delivery")
        shutil.copy2(source, destination)
    report = {"training_status": "completed", "acceptance_status": frozen["status"],
              "improvement_demonstrated": frozen["acceptance"]["passed"], "automatic_promotion": False,
              "pt": str(paths[0].relative_to(ROOT)), "pt_sha256": sha256(paths[0]),
              "onnx": str(paths[1].relative_to(ROOT)), "onnx_sha256": sha256(paths[1]),
              "configuration": json.loads(CONFIG.read_text()), "selection": frozen, "verification": verification,
              "historical_benchmarks": final, "preflight": json.loads((WORK / "preflight.json").read_text()),
              "training_receipts": [json.loads(p.read_text()) for p in sorted((WORK / "receipts").glob("*.json"))],
              "created_unix": time.time(), "limitations": ["Historical benchmarks already used", "No independent-day validation for new PhenoZero/TS02 cameras", "MPS is not bitwise deterministic"]}
    report["validation_pt_onnx"] = {"pt": pt_validation["thresholds"][str(frozen["confidence"])]["pipeline"],
                                    "onnx": frozen["replicas"][0]["pipeline"]}
    write(paths[2], report)
    write(WORK / "final-model.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify", "deliver"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.action == "export":
        if not args.checkpoint or not args.name:
            parser.error("Export requires checkpoint and name")
        export(args.checkpoint.resolve(), args.name)
    else:
        {"verify": verify, "deliver": deliver}[args.action]()
