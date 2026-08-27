<!--
This document owns the inference architecture, model compatibility contract,
local verification procedure, and optional model-export workflow.
-->

# Development

## Inference architecture

Inference uses the original `models/yolo26n.onnx` model and its end-to-end ONNX
graph. The runtime calls it directly through ONNX Runtime.

Each image is processed sequentially in one ONNX session using one full-image
view plus fifteen adaptive overlapping crops. Horizontal and square images use
a `5×3` crop grid, while vertical images use `3×5`; both use 20% nominal
overlap. The EXIF-normalized RGB source supplies all sixteen views and remains
the final rendering background.

Crop detections are converted to global image coordinates, all model classes
are merged, and all rows require confidence greater than or equal to 0.30.
Duplicates are suppressed when IoU or smaller-box coverage reaches 0.50.
`car`, `bus`, and `truck` compete across labels, while other classes compete
only with themselves. `phenocam/classes/configuration.py` selects the final
annotations and privacy regions.

The ONNX model must expose exactly the standard 80 COCO classes and the
end-to-end six-column detection output used by the included YOLO26n model.
Incompatible metadata or tensor shapes are rejected before inference.

## Verification

Run the complete test suite from the project directory with the runtime
environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions runs this deterministic suite with Python 3.13 on Ubuntu x86-64
for pull requests and pushes to `main`, then syntax-checks both runtime shell
scripts. This CI does not qualify Raspberry Pi hardware.

The six real-image integration cases and their inventory check are skipped when
the named, untracked reference images are absent from `input/`. When available,
they verify valid annotated outputs and the final suppression-domain overlap
contract. Person and car counts are not asserted because they are not ground
truth or a measurement of accuracy, precision, recall, or mAP.

## Optional model export

Export is a workstation task because Ultralytics brings a much larger Python
stack. Install it in a separate environment only when regeneration is needed:

```sh
python3 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/export.txt
.venv-export/bin/python scripts/export/fp32.py
```

The included and tested `models/yolo26n.onnx` does not need to be exported on
the Raspberry Pi.
