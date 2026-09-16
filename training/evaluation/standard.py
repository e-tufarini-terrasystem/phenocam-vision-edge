"""Standard validation metrics for one explicit model and image list."""

import torch
import yaml

def metrics(model, dataset, paths, output, name):
    image_list = output / f"{name}.txt"
    image_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
    config = output / f"{name}.yaml"
    config.write_text(yaml.safe_dump({"path": str(dataset), "train": str(image_list), "val": str(image_list), "names": dict(model.names)}, sort_keys=False), encoding="utf-8")
    result = model.val(
        data=str(config), imgsz=640, batch=16, device="mps" if torch.backends.mps.is_available() else "cpu", workers=0,
        plots=False, verbose=False, project=str(output), name=name, exist_ok=True,
    )
    box = result.box
    per_class = {}
    for index, class_id in enumerate(map(int, box.ap_class_index)):
        precision, recall = float(box.p[index]), float(box.r[index])
        per_class[model.names[class_id]] = {
            "map50_95": float(box.all_ap[index].mean()),
            "map50": float(box.all_ap[index, 0]),
            "precision_at_max_f1": precision,
            "recall_at_max_f1": recall,
        }
    precision, recall = float(box.mp), float(box.mr)
    return {
        "images": len(paths),
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "precision_at_max_f1": precision,
        "recall_at_max_f1": recall,
        "f1_at_max_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "per_class": per_class,
        "speed_ms_per_image": result.speed,
    }
