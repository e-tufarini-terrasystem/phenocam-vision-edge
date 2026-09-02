<!--
This document owns the inference architecture, model compatibility contract,
local verification procedure, and optional model-export workflow.
-->

# Development

## Inference architecture

Inference uses the validation-selected `models/yolo26n-v3-extended.onnx` and
its end-to-end ONNX graph. The original and v2 models remain available for
comparisons. The runtime calls the selected model directly through ONNX Runtime.

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

The included and tested `models/yolo26n-v3-extended.onnx` does not need to be
exported on the Raspberry Pi. Test evaluation and operational validation on
deployment imagery remain pending.

## Optional v2 training on Apple Silicon

`notebooks/training-v2.ipynb` fine-tunes the committed checkpoint through MPS, keeps its
80-class COCO runtime contract, compares both checkpoints on a deterministic
group-safe validation split, plots the training curves, exports
`models/yolo26n-v2.onnx`, and runs one end-to-end application inference. From
the repository root:

```sh
python3.13 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/export.txt jupyterlab
.venv-export/bin/python -m ipykernel install --sys-prefix --name phenocam-training-v2
.venv-export/bin/jupyter lab notebooks/training-v2.ipynb
```

Run the cells in order. Generated splits, metrics, plots, and the runtime sample
stay under ignored `output/training-v2/`. Validation in the notebook reuses part
of the public training dataset and does not satisfy the pending operational
validation on internal deployment imagery.

## V3 training notebooks

Both v3 notebooks start from the original COCO checkpoint rather than v2. This
prevents a v2 training image from leaking into the canonical TEST-ID benchmark.
They preserve the 80-class runtime contract, select `best.pt` on Validation,
report TEST-ID and TEST-OOD separately, export ONNX, and run the application
contract check.

```sh
.venv-export/bin/jupyter lab notebooks/training-v3.ipynb
.venv-export/bin/jupyter lab notebooks/training-v3-expanded.ipynb
```

The canonical notebook trains on the 1,600-image v3 Train split. The expanded
notebook adds only the 18 reviewed public PhenoCam images whose sites already
belong to Train. It reuses the canonical Validation and test splits; the 240
private operational images remain excluded from training. Outputs stay under
ignored `output/training-v3*/`, while accepted checkpoints and ONNX exports are
copied to `models/`.

## V3 operational mining

V3 keeps the read-only operational root outside the committed configuration.
Set it only in the current shell, then create the deterministic inventory and
sealed site-day split:

```sh
OPERATIONAL_ROOT=/path/to/read-only/phenocam
dataset/.venv/bin/python -m dataset.builder.mining inventory \
  --root "$OPERATIONAL_ROOT" \
  --output dataset/workspace/training-v3/internal/inventory.csv \
  --rejections dataset/workspace/training-v3/internal/inventory-rejections.csv
dataset/.venv/bin/python -m dataset.builder.mining split \
  --inventory dataset/workspace/training-v3/internal/inventory.csv \
  --output dataset/workspace/training-v3/internal/operational-split.csv \
  --audit dataset/workspace/training-v3/internal/operational-split.json
```

The screening command writes one atomic JSON record per image. Pass only
`operational_dev` and `operational_mining` for internal runs; never pass
`sealed_test` before thresholds and rules are frozen. Matching run/model hashes
make `--resume` idempotent and reject stale checkpoints.

Public PhenoCam recovery uses the separate YOLO26x workstation environment and
excludes both the existing public dataset and the completed negative pilot:

```sh
.venv-export/bin/python -m dataset.builder.mining teacher-screen \
  --candidates dataset/workspace/sources/phenocam/baseline-screened.csv \
  --existing dataset/training-dataset/metadata/source-images.csv \
  --reviewed dataset/workspace/training-v3/selection/public-pilot.csv \
  --model dataset/workspace/models/yolo26x.pt \
  --output-dir dataset/workspace/training-v3/screening/public-teacher \
  --resume
```

The acquisition floor is `0.30`, while selection and CVAT suggestions require
`0.50`. Selection is confidence-first, limited to five frames per camera-day
group, and remains subject to visual false-positive review. The clean task is
uploaded idempotently with:

```sh
dataset/commands/cvat-tasks.sh upload-v3-public-teacher
```

Teacher boxes are proposals only. They become dataset labels only after task
completion, verified export and attributed import.

Task 11 was completed with `stage=annotation`, `state=completed`. Reproduce its
export, attributed import and admission decision with:

```sh
dataset/commands/cvat-tasks.sh export 11 v3-public-teacher
dataset/.venv/bin/python -m dataset.builder.mining import-reviewed \
  --selection dataset/workspace/training-v3/selection/public-teacher-clean.csv \
  --bundle-dir dataset/workspace/annotation/cvat/v3-public-teacher \
  --export dataset/workspace/annotation/exports/v3-public-teacher-reviewed.coco.zip \
  --output-dir dataset/workspace/training-v3/reviewed/public-phenocam-teacher \
  --task-id 11 --annotator Emanuele --reviewer Emanuele --task-completed
dataset/.venv/bin/python -m dataset.builder.mining select-reviewed-public \
  --reviewed-dir dataset/workspace/training-v3/reviewed/public-phenocam-teacher \
  --existing dataset/dataset-v3/metadata/source-images.csv \
  --embeddings dataset/workspace/deduplication/embeddings.npz \
  --output dataset/workspace/training-v3/selection/public-teacher-reviewed-decisions.csv \
  --statistics dataset/workspace/training-v3/selection/public-teacher-reviewed-statistics.json
```

The decision CSV preserves every image as `included`, `reserved`, or
`rejected`. Admission is capped at 18 images, eight per site and one per
camera-day; SSCD similarity `>=0.95` prevents internal near-duplicates. Human
annotation content is ranked before deterministic tie-breaking. Build the first
expanded artifact beside the current v3 so it can be verified without an
overwrite:

```sh
dataset/.venv/bin/python -m dataset.builder.finalization build-v3 \
  --destination dataset/dataset-v3-expanded \
  --include-public-expansion
```

Generated inventories, predictions, selections, copied CVAT images and exports
remain under ignored `dataset/workspace/`. They must not be added to Git.

After human completion and a verified final COCO export, import each operational
task with its selection and original bundle:

```sh
dataset/.venv/bin/python -m dataset.builder.mining import-reviewed \
  --selection dataset/workspace/training-v3/selection/internal-representative.csv \
  --bundle-dir dataset/workspace/annotation/cvat/v3-internal-representative-clean \
  --export /path/to/task-9-final.coco.zip \
  --output-dir dataset/workspace/training-v3/reviewed/operational-dev-representative \
  --task-id 9 --annotator Emanuele --reviewer Emanuele --task-completed
```

The import validates identities, source hashes, dimensions, classes, boxes and
task completeness. It writes an attributed manifest, a preannotation/ground
truth report, a COCO archive, and verified image copies. Image names expose the
site without relying on local paths, for example
`raspberrypi2.local--2025-10-30T121905--f4ab422e2908.jpg`. Internal reviewed
images remain operational development/mining data and do not enter first-cycle
v3 training.

After both reviewed operational tasks have been imported, materialize the
viewer-compatible current v3 artifact:

```sh
dataset/commands/dataset-finalization.sh build-v3
```

The command now produces the intermediate `dataset/dataset-v3-source/`. Build
and verify the final leakage-aware artifact with:

```sh
cd dataset
.venv/bin/python -m builder.partition build dataset-v3-source dataset-v3
.venv/bin/python -m builder.partition verify dataset-v3
```

The final artifact exposes Train, Validation, TEST-ID and TEST-OOD. Internal
frames remain TEST-OOD and never enter training. Open `dataset/viewer.html`,
select `dataset-v3`, and use the split selector to inspect each subset.
