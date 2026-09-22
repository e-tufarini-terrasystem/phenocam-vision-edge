<!--
Own the current development workflow and preserve earlier implementation context.
-->

# Development

For a source checkout and the runtime environment, follow the
[operating guide](cli.md#manual-setup-from-source). Workstation commands below
require the full source repository; dataset and training tools are not shipped
in the runtime archive.

## Current workflow

The repository maintains one model: **YOLO26n specialized for PhenoCam**.
`models/yolo26n-phenocam.{pt,onnx,json}` replace the former v6 filenames.
Model bytes are unchanged. The runtime JSON receipt contains only `model_id`,
`model_version` and `onnx_sha256`. Acceptance status and provenance remain in the
[training report](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/docs/status/training-v6-experiment.md)
and [model comparison](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/docs/status/model-comparison-2026-09-15.md).
The model remains experimental, with no demonstrated overall improvement.
Software `v0.2.3` is considered stable by the maintainer following manual testing
on Raspberry Pi, as recorded in the [README](../README.md#supported-environments).
Runtime testing on the device and evaluation of model accuracy are separate;
the release's manual test does not qualify later source changes.
The original COCO `models/yolo26n.pt` initializes fresh training;
its ONNX and receipt are also tracked for optional diagnostic inference.

Code responsibilities:

- `phenocam/`: deployed inference; no training or dataset dependency.
- `dataset/builder/`: source acquisition, human review and current artifact.
- `training/`: one bounded training workflow and evaluation.
- `scripts/`: runtime batch, installation, packaging and optional base export.
- `dataset/data/`: one reviewed catalog and five disjoint splits.

Set up the workstation environment with the existing export requirements plus
runtime dependencies. Ultralytics supplies its training dependencies; the ONNX
export tools are explicitly included in the export requirements.
The historical environment is recorded in the original report. Reproducing its
exact numerical results requires that environment and the original code.

```sh
python3.13 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/runtime.txt -r requirements/export.txt
```

Reconstruct the catalog's local images using the approved SHA-256-addressed pool,
then verify them. The pool includes frozen crops; public downloads alone cannot
recreate human annotations or private images. See [dataset instructions](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md).

```sh
.venv-export/bin/python -m dataset.builder.artifact hydrate \
  --sources dataset/workspace/sources/approved/images
.venv-export/bin/python -m dataset.builder.artifact verify
```

Hydration refuses an existing image directory. If images are already present,
run only verification. The versioned catalog has no machine-specific root;
`dataset.yaml` is generated locally. Training generates its own absolute YAML
inside the named run directory, so moving a checkout does not reuse stale roots.

Start a fresh protocol from the repository root with a unique run name:

```sh
PHENOCAM_TRAINING_RUN=experiment-01 \
  .venv-export/bin/python -m training.workflow
```

Preflight freezes the current specialized PT/ONNX pair as the reference, verifies
the full dataset, records the actual environment and snapshots the current code.
Two initial recipes plus two replicas retain the four-training budget, fixed
seeds, complete 2,020-image training split and validation-only selection.
New runs compare against the frozen current model, not the historical v5 model.
The original v6 experiment and its acceptance decision are not reinterpreted.

Run artifacts stay under `output/training/<run>/`. Delivery goes to that run's
`delivery/` directory, never automatically replacing the shipped model.
Failed or partial runs are preserved. Running the same workflow can reuse
completed stages only when their recorded inputs still match; an incomplete
training or partial export requires inspection or a new run name.
Preflight also freezes the checksum inventory covering human annotations.
Training and evaluation reject a changed inventory, including when the image
manifest is unchanged. Receipts without this annotation identity cannot be
resumed; preserve the old run and choose a new run name.

Standalone validation does not require training:

```sh
YOLO_NUM_THREADS=4 .venv/bin/python -m training.evaluation \
  --model models/yolo26n-phenocam.onnx --dataset dataset/data --split val \
  --threshold 0.47 --output output/validation-current
```

Benchmark splits require a frozen receipt for the exact model and threshold.
They are historical benchmarks, not new independent validation.

## Verification

From the repository root with the runtime environment:

```sh
.venv/bin/python -m pip install -r requirements/test.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s dataset/tests -v
sh -n scripts/batch.sh
sh -n scripts/installer.sh
for script in dataset/commands/*.sh; do sh -n "$script"; done
git diff --check
```

CI runs both suites and checks all maintained shell scripts. Dataset tests use
fixtures and the committed catalog, so private images and the training stack
are not required. Image integration cases remain optional when their named
reference inputs are absent.

## Runtime architecture

One ONNX session processes the EXIF-normalized RGB source sequentially: one full
view, then fifteen crops. Horizontal/square inputs use a `5×3` grid; vertical
inputs use `3×5`, with 20% nominal overlap. Crop boxes map back to the source
before global suppression and enabled-class filtering.

The model must expose the standard 80 COCO classes and an end-to-end float
output with six columns (box, confidence, class). Input and output metadata and
runtime tensors are validated. The shipped model uses `1×3×640×640` input and
`1×300×6` output. Filesystem effects and failure ordering are specified in the
[operating guide](cli.md#outputs-and-conditional-deletion).

## Model export and release identity

Optional base export runs on the workstation, after creating `.venv-export`:

```sh
.venv-export/bin/python scripts/export/fp32.py
```

It writes under `output/export/`; it does not replace the shipped specialized
model. Keep each `(model_id, model_version)` bound to exactly one ONNX SHA-256.
A new export with different bytes needs a new version and matching receipt.
Software releases alone do not change model versions or experimental status.
A stable software release may bundle an experimental model when its status and
limitations are clearly documented. Software promotion does not change the
model's acceptance status or historical evaluation results.

Software version lives once in `phenocam/__init__.py`, in literal form
`__version__ = "MAJOR.MINOR.PATCH"`. Before a release, update that declaration and
current installation/metadata examples together. `scripts/package.py` reads the
selected commit without executing its code and rejects an archive version that
disagrees; historical commits without the declaration remain supported.

```sh
.venv/bin/python scripts/package.py VERSION COMMIT EXISTING_OUTPUT_DIRECTORY
```

Replace all three arguments with the intended version, immutable commit and an
existing destination outside the repository. The packager selects runtime files,
README, CLI/development guides and `LICENSE` when present in that commit;
historical `docs/manual.md` is included only for commits that still contain it.
It excludes dataset, training, tests and report artifacts. Updated packaging does
not alter previously published archives. Publication requires the exact commit's
CI, archive/checksum and temporary `.meta` identity checks; target-device evidence
must remain explicit. See `AGENTS.md` and `.skills/release/SKILL.md` in the source
repository for the release procedure and authorization boundaries.

## Historical evidence

The repository's `docs/status/` retains experimental protocols, measured results,
annotation decisions and handoffs. Its v6 training report records the original
80% cross-view coverage; the 15 September comparison uses 50% at confidence 0.47.
Do not combine those measurements or describe historical tests as new blind tests.
Raw evidence under ignored `output/` and `dataset/workspace/` needs the original
local artifacts or an authorized backup; it is not distributed with a clone.

## Historical workflow before consolidation

<details>
<summary>Pre-consolidation development context and commands (da14b7d)</summary>

The following material preserves development context at commit `da14b7d`.
Versioned model paths, notebooks, build commands and dataset names below describe
that revision, not the supported current entry points above. Original tracked
files can be read with `git show da14b7d:<path>`; the local backup branch is
`backup/before-single-model-consolidation`. Historical measurements retain their
original model, threshold, code and dataset identities.


### Inference architecture

The v5 reference model is `models/yolo26n-v5.onnx`, with an end-to-end ONNX
graph. Pass this path explicitly with `--model` when using v5; older models
remain available for comparisons. The runtime suppression rule applies to
every compatible model.

Each image is processed sequentially in one ONNX session using one full-image
view plus fifteen adaptive overlapping crops. Horizontal and square images use
a `5×3` crop grid, while vertical images use `3×5`; both use 20% nominal
overlap. The EXIF-normalized RGB source supplies all sixteen views and remains
the final rendering background.

Crop detections are converted to global image coordinates, all model classes
are merged, and all rows require confidence greater than or equal to 0.47.
Duplicates are suppressed when IoU reaches 0.50. Smaller-box coverage also
suppresses at 0.50, but only between different views, to reconcile crop
fragments without discarding adjacent occluded objects from the same view.
`car`, `bus`, and `truck` compete across labels, while other classes compete
only with themselves. `phenocam/classes/configuration.py` selects the final
annotations and privacy regions.

The ONNX model must expose exactly the standard 80 COCO classes and the
end-to-end six-column detection output used by the included YOLO26n model.
Incompatible metadata or tensor shapes are rejected before inference.

The v6 experiment completed four independent training runs from the base.
Its selected `models/yolo26n-v6.onnx` remains experimental because it failed
the validation acceptance criteria. `scripts/batch.sh` currently selects this
experimental v6 model explicitly; this setting does not change its acceptance
status. The [v6 training report](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/docs/status/training-v6-experiment.md) records its
fixed protocol, environment, commands, metrics, and verification evidence.
Checkpoints, logs, and completed run receipts are under `output/training-v6/`.

### Occlusion calibration

Historical results below use cross-view coverage 0.80. On 2026-09-14 the
runtime coverage threshold was changed to 0.50 at the user's request;
these metrics have not been remeasured with that threshold.

On 2026-09-08, v5 predictions were evaluated with the legacy suppression rule
and cross-view containment thresholds of 0.80, 0.90, and 0.95, each at confidence
0.47, 0.40, 0.35, and 0.25. The selected combination maximized validation F1:
IoU 0.50, cross-view coverage 0.80, and confidence 0.47. Lowering confidence
recovered more targets but reduced F1 at every tested containment threshold.

All candidates reused the same ONNX predictions from the 206-image v5
validation split, acquired at a 0.20 confidence floor. This preserves every
candidate needed by the threshold grid. Matching used IoU 0.50 and the five
enabled classes: person, car, motorcycle, bus, and truck. Model, 640x640 input,
sixteen views, CPU provider, and four threads stayed fixed. The environment was
Python 3.13.14 on an Apple Silicon Mac; this is not Raspberry Pi qualification.
The ONNX SHA-256 remained
`252f302257759cae6d40579fb76b74d66f87bdce2997d44f89dba85d03420379`.

| Validation rule | Confidence | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 0.47 | 0.42607 | 0.44785 | 0.43669 |
| Cross-view 0.80 | 0.47 | 0.42077 | 0.48875 | 0.45222 |
| Cross-view 0.80 | 0.40 | 0.38720 | 0.51943 | 0.44367 |
| Cross-view 0.80 | 0.35 | 0.36630 | 0.53783 | 0.43579 |
| Cross-view 0.80 | 0.25 | 0.32681 | 0.58078 | 0.41826 |
| Cross-view 0.90 | 0.47 | 0.41267 | 0.49284 | 0.44921 |
| Cross-view 0.95 | 0.47 | 0.40641 | 0.49284 | 0.44547 |

The implementation was replayed against the cached validation predictions and
matched the selected result. Rule, confidence, and model/code hashes were
recorded before opening the regression splits. Both sides below use the same
v5 ONNX and confidence 0.47; only suppression differs.

| Split (images) | TP before/after | FP before/after | Recall before/after | F1 before/after |
| --- | ---: | ---: | ---: | ---: |
| Validation (206) | 219 / 239 | 295 / 329 | 0.44785 / 0.48875 | 0.43669 / 0.45222 |
| PKLot holdout (6) | 282 / 282 | 62 / 63 | 0.21478 / 0.21478 | 0.34037 / 0.34017 |
| TEST-ID (200) | 235 / 246 | 299 / 353 | 0.53047 / 0.55530 | 0.48106 / 0.47217 |
| TEST-OOD (240) | 2384 / 2456 | 117 / 178 | 0.40510 / 0.41733 | 0.56857 / 0.57659 |

This is a recall trade-off, not a uniform accuracy improvement. TEST-ID
precision falls from 0.44007 to 0.41068; TEST-OOD precision falls from 0.95322
to 0.93242. Duplicate detections rise from 0 to 1 on validation, 4 to 5 on
TEST-ID, and 1 to 13 on TEST-OOD. Validation's 78 negative images retain eight
false-positive boxes; TEST-ID negatives increase from 16 to 17 boxes, and
TEST-OOD negatives remain at zero. These are existing benchmarks, not a new
independent occlusion-labelled test; repeated camera scenes limit independence.
No training or model export was performed, and no rule was retuned on test.

In `input/phenozero1_2026_09_08_095832.jpg`, the final car count increases from
three to four. The recovered full-view candidate has confidence 0.519031:
its overlap with the adjacent car is IoU 0.158367 and smaller-box coverage
0.560375. The old coverage rule discarded it; the new rule retains it. Other
cars below confidence 0.47 remain missed. This does not establish recovery of
foliage-occluded objects that the model never predicts.

Local receipts, metrics, error events, cached validation predictions, test log,
and before/after images are under ignored
`output/occlusion-calibration-2026-09-08/`. The baseline source revision is
`3396bdfe1bb01bb8bf9a5f14aee7a0558cb76a91`. Reproduce current validation with
the existing evaluator and a new output directory:

```sh
YOLO_NUM_THREADS=4 .venv/bin/python -m scripts.runtime_evaluation.evaluate \
  --model models/yolo26n-v5.onnx --dataset dataset/dataset-v5 --split val \
  --threshold 0.47 --threshold 0.40 --threshold 0.35 --threshold 0.25 \
  --output output/occlusion-validation
```

### Historical verification

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

### Optional model export

Export is a workstation task because Ultralytics brings a much larger Python
stack. Install it in a separate environment only when regeneration is needed:

```sh
python3 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/export.txt
.venv-export/bin/python scripts/export/fp32.py
```

The included and tested `models/yolo26n-v4.onnx` does not need to be exported
on the Raspberry Pi. The frozen evaluation is documented in
`docs/status/training-v4-experiment-2026-09-03.md`; target-device latency remains
pending because a Raspberry Pi was not reachable during the cycle.

### Optional v2 training on Apple Silicon

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

### V3 training notebooks

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

### V3 operational mining

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

</details>
