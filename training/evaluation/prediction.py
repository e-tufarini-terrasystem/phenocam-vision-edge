"""Run exact runtime views and fusion while retaining detection provenance."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from phenocam.inference.detections import _overlaps, deduplicate, normalize_rows
from phenocam.inference.runtime import create_session, model_contract, run_tensor
from phenocam.inference.views import iter_views, load_image


def _same_suppression_domain(left, right, names):
    road = {"car", "bus", "truck"}
    left_name, right_name = names[left.class_id], names[right.class_id]
    return left.class_id == right.class_id or left_name in road and right_name in road


class Predictor:
    """Own one PT or ONNX model and expose identical 16-view geometry."""

    def __init__(self, model_path, device="mps", image_size=640, confidence_floor=0.001):
        self.path = Path(model_path)
        self.device = device
        self.floor = float(confidence_floor)
        if not self.path.is_file() or not 0 <= self.floor < 1:
            raise ValueError("invalid evaluation model or confidence floor")
        self.kind = self.path.suffix.lower()
        if self.kind == ".onnx":
            self.model = create_session(self.path)
            self.input_name, self.output_name, self.width, self.height, self.names = model_contract(self.model)
        elif self.kind == ".pt":
            import torch
            from ultralytics import YOLO

            self.torch = torch
            self.model = YOLO(self.path)
            self.names = {int(key): value for key, value in self.model.names.items()}
            self.width = self.height = int(image_size)
            self.input_name = self.output_name = None
        else:
            raise ValueError("evaluation model must be PT or ONNX")

    def _run(self, tensor):
        if self.kind == ".onnx":
            return run_tensor(self.model, self.input_name, self.output_name, tensor)
        if self.device == "mps":
            self.torch.mps.synchronize()
        started = perf_counter()
        result = self.model.predict(
            self.torch.from_numpy(tensor), imgsz=self.width, conf=self.floor,
            device=self.device, verbose=False,
        )[0]
        if self.device == "mps":
            self.torch.mps.synchronize()
        elapsed = perf_counter() - started
        rows = np.column_stack((
            result.boxes.xyxy.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
            result.boxes.cls.detach().cpu().numpy(),
        )).astype(np.float32, copy=False)
        return rows, elapsed

    def raw_image(self, image_path):
        image = load_image(Path(image_path))
        full, crops, full_seconds, crop_seconds = [], [], 0.0, 0.0
        started = perf_counter()
        for view in iter_views(image, self.width, self.height):
            rows, elapsed = self._run(view.tensor)
            detections = normalize_rows(
                rows, view, image.width, image.height, self.names, self.floor,
            )
            if view.priority == 0:
                full.extend(detections)
                full_seconds += elapsed
            else:
                crops.extend(detections)
                crop_seconds += elapsed
        return {
            "width": image.width,
            "height": image.height,
            "full": tuple(full),
            "crop": tuple(crops),
            "full_inference_seconds": full_seconds,
            "crop_inference_seconds": crop_seconds,
            "wall_seconds": perf_counter() - started,
        }

    def merge(self, raw, threshold):
        full = tuple(item for item in raw["full"] if item.confidence >= threshold)
        crop = tuple(item for item in raw["crop"] if item.confidence >= threshold)
        merged = deduplicate((*full, *crop), self.names)
        modes = {
            "full_image": deduplicate(full, self.names),
            "crop_only": deduplicate(crop, self.names),
            "pipeline": merged,
        }
        output = {}
        for mode, detections in modes.items():
            values = []
            for item in detections:
                if mode == "pipeline":
                    in_full = any(_same_suppression_domain(item, other, self.names) and _overlaps(item, other) for other in full)
                    in_crop = any(_same_suppression_domain(item, other, self.names) and _overlaps(item, other) for other in crop)
                    source = "shared" if in_full and in_crop else "full" if in_full else "crop"
                else:
                    source = "full" if mode == "full_image" else "crop"
                values.append({**asdict(item), "view_source": source})
            raw_count = len(full) if mode == "full_image" else len(crop) if mode == "crop_only" else len(full) + len(crop)
            output[mode] = {
                "detections": values,
                "before_dedup": raw_count,
                "duplicates_suppressed": raw_count - len(values),
            }
        return output
