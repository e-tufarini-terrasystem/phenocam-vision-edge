# YOLO26n training report — 0.1.3-extended

> Historical artifact labels use normalized model versions, not filesystem paths.
> Exact commands, identifiers and paths remain in the original document:
> `git cat-file blob 97c09e41205543cb7bae392f60b5c3a9b4b01411` from the
> [source snapshot](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/tree/cc63857c12c5553c2e3451854863edf7c0705e2a).

> Historical record for this iteration; use the [current guide](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md).
> Commands refer to the original workflow; ignored artifacts require local evidence.

Status: model selected and exported on 2026-09-02; test evaluation was stopped
at the user's request. The reviewed 18-image dataset extension is named
0.1.3 expanded; this report describes its 0.1.3 extended training experiment.
The dataset extension and the selected model are distinct artifacts.

## Executive Summary

Full-network fine-tuning underperformed the untouched COCO checkpoint. A
head-only low-rate refinement was selected and reproduced after an external
cleanup deleted the first checkpoint. The recovered model reached validation
mAP50-95 0.5404 and was exported with its 80-class runtime contract intact.
The user accepted the 0.0002 difference from the first run and stopped the
remaining stability and test stages.

## Environment

| Item | Value |
| --- | --- |
| Git commit | `7a9d73061f7ad8e1ce6ac7d7a614e35a472abf4d` (dirty worktree recorded per run) |
| Dataset artifact | dataset 0.1.3 expanded plus frozen dataset 0.1.3 split |
| Dataset fingerprint | `e87592ab2a4d614319518a199aaae52f038ee98a46b40a37f4fc4b06ba484c89` |
| Python | 3.13.14 |
| Ultralytics | 8.4.48 |
| PyTorch | 2.13.0 |
| CUDA | unavailable |
| Hardware | Apple M4, 10-core GPU, 24 GiB unified memory |
| Training device | MPS |
| Initialization | `models/yolo26n.pt`, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Seed | 42 |

PyTorch warns that `index_put_with_accumulate_mps` has no deterministic MPS
implementation. Seeds and `deterministic=True` are preserved, but bitwise
reproducibility is therefore not claimed. A seed-17 stability run was started
but intentionally stopped before completion. PyTorch MPS uses unified memory; per-batch
driver allocation is sampled because CUDA-style reserved VRAM is not available.

## Dataset

The approved canonical split remains unchanged. The 18 reviewed expansion
images are added only to canonical train. Tests never appear in the development
YAML.

The runtime-compatible detection head retains all 80 COCO class indices. This
dataset annotates and evaluates the six configured targets at their original
indices: person (0), bicycle (1), car (2), motorcycle (3), bus (5), and truck
(7). Keeping the original mapping avoids an incompatible head or silent class
renumbering in the edge application.

| Split | Images | Positive | Negative | Objects |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,618 | 1,013 | 605 | 3,854 |
| Validation | 200 | 125 | 75 | 464 |
| TEST-ID | 200 | 124 | 76 | 463 |
| TEST-OOD | 240 | 231 | 9 | 5,885 |

| Class | Train | Validation | TEST-ID | TEST-OOD |
| --- | ---: | ---: | ---: | ---: |
| person | 2,443 | 306 | 303 | 180 |
| bicycle | 160 | 20 | 20 | 0 |
| car | 714 | 74 | 74 | 5,558 |
| motorcycle | 163 | 20 | 21 | 37 |
| bus | 145 | 18 | 18 | 0 |
| truck | 229 | 26 | 27 | 110 |

## Dataset Integrity Verification

The historical dataset audit module independently performed the following
checks before the new search started; its invocation is in the original document:

- verified every entry in the canonical and extended checksum manifests;
- decoded all selected images with Pillow;
- parsed every label as five numeric YOLO fields;
- required target class IDs and normalized, positive-size boxes;
- reconciled absent labels exactly with intentional negative images;
- verified unique paths and no path crossing between splits;
- invoked the repository partition verifier for group, camera, SHA-256 and
  pHash leakage invariants;
- required every expansion camera to belong exclusively to canonical train;
- recomputed image and object counts globally and per class.

All 3,760 extended-artifact checksums passed. The canonical verifier reported
2,240 images, 10,562 objects, 1,417 partition groups, 157 PhenoCam cameras,
zero duplicate hashes and zero cross-split pHash-near-duplicate pairs. Measured
validation (464), TEST-ID (463), and TEST-OOD (5,885) object totals exactly
match the approved expected counts. The authoritative machine-readable receipt
is training output 0.1.3 extended (`dataset-audit.json`).

## Web Research and Methodology

Authoritative sources consulted:

- [Ultralytics train mode](https://docs.ultralytics.com/modes/train/)
- [Ultralytics YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [Ultralytics fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide/)
- [Ultralytics hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [Ultralytics Ray Tune integration](https://docs.ultralytics.com/integrations/ray-tune/)
- [Ultralytics validation mode](https://docs.ultralytics.com/modes/val/)
- [Ultralytics performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)
- [Ultralytics model testing](https://docs.ultralytics.com/guides/model-testing/)
- [Ultralytics end-to-end detection](https://docs.ultralytics.com/guides/end2end-detection/)
- [COCO detection evaluation overview](https://presentations.cocodataset.org/COCO17-Detect-Overview.pdf)
- [WiSE-FT: Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)

Decisions influenced by the research:

- Start from the official COCO-pretrained `yolo26n.pt`, not random weights.
- Preserve YOLO26n because it is the repository's edge-deployment architecture;
  YOLO26x remains an annotation teacher, not a silent architecture change.
- Use mAP50-95 as primary selection metric, with per-class AP, precision and
  recall as safeguards.
- Keep 640 as the baseline resolution, test 960 separately for small-object
  sensitivity, and keep batch size a hardware concern rather than a quality
  hyperparameter.
- Compare AdamW learning rates, the official MuSGD recipe, reduced augmentation,
  moderate class weighting, backbone freezing, and higher resolution. Rotation,
  shear, vertical flip, mixup and copy-paste are not searched because the fixed
  camera geometry and dataset size do not justify them.
- Use non-zero one-epoch warmup for explicit fine-tunes and disable mosaic in
  the final five epochs.
- Use validation-only early stopping and freeze the checkpoint and operating
  threshold before the final test command becomes available.
- Select the operating confidence in the validation maximum-mean-F1 region.
- Retain YOLO26's end-to-end, NMS-free head because the ONNX runtime consumes
  its `(N, 300, 6)` output. Consequently there is no NMS IoU threshold to tune;
  IoU 0.50 in object accounting is an evaluation matching threshold, not NMS.
- Because ordinary fine-tuning consistently moved away from a stronger
  pretrained solution, test a small, fixed validation-only grid of linear
  weight interpolations. WiSE-FT motivates this preservation hypothesis, but it
  is evidence from image classification rather than a claim that object
  detection must behave identically; the experiment is retained only if this
  dataset's validation result supports it.

Ray Tune and Optuna are not installed. Adding them would introduce dependencies
without enabling useful parallelism on the single MPS device. The search is a
predeclared, deterministic sequence of isolated trials instead. Each run starts
from the same checkpoint and records its complete arguments, environment,
runtime, checkpoint hash, and validation history.

## Baseline

| Run | mAP50-95 | mAP50 | mAP75 | Precision | Recall | F1 | Best epoch | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched COCO checkpoint | 0.5318 | 0.6574 | 0.5367 | 0.7017 | 0.6439 | 0.6716 | n/a | validation only |
| Historical default 0.1.3-expanded fine-tune | 0.5178 | 0.6415 | 0.5350 | 0.6725 | 0.6175 | 0.6438 | 1 | 1,599 s |

The historical baseline used 30 maximum epochs, patience 8, batch 8, 640 px,
seed 42, `optimizer=auto`, default augmentation, and stopped after nine epochs.
Ultralytics selected AdamW with learning rate 0.000119 and momentum 0.9. Its
accepted checkpoint and export are retained under `models/`; ignored run output
was removed by the cleanup incident documented below.

## Hyperparameter Search

Search definition: training implementation 0.1.3 extended (`experiments.json`). Broad runs
use 15 maximum epochs and patience 5. Except for the named dimension, they use
batch 8, 640 px, seed 42, default YOLO26n loss weights and augmentation, and
close mosaic for the final five epochs.

| Rank | Run | mAP50-95 | mAP50 | Precision | Recall | Best epoch | Runtime |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `narrow-head-only-00010` | 0.5398 | 0.6680 | 0.6482 | 0.6376 | 12 | 1,995 s |
| 2 | class-weighted head | 0.5388 | 0.6625 | 0.6435 | 0.6550 | 6 | 1,202 s |
| 3 | `narrow-head-only-00005` | 0.5382 | 0.6655 | 0.6446 | 0.6493 | 9 | 1,768 s |
| 4 | interpolation alpha 0.25 | 0.5370 | 0.6575 | 0.6984 | 0.6326 | n/a | 2.8 s validation |
| 5 | interpolation alpha 0.50 | 0.5366 | 0.6594 | 0.7021 | 0.6369 | n/a | 3.5 s validation |
| 6 | no-mosaic ablation | 0.5329 | 0.6586 | 0.6574 | 0.6534 | 11 | 1,532 s |
| 7 | Untouched pretrained reference | 0.5318 | 0.6574 | 0.7017 | 0.6439 | n/a | validation only |
| 8 | interpolation alpha 0.75 | 0.5308 | 0.6511 | 0.6954 | 0.6205 | n/a | 2.7 s validation |
| 9 | interpolation alpha 0.10 | 0.5279 | 0.6542 | 0.7128 | 0.6147 | n/a | 13.4 s validation |
| 10 | Historical default baseline | 0.5178 | 0.6415 | 0.6725 | 0.6175 | 1 | 1,599 s |
| 11 | `broad-freeze-backbone` | 0.4902 | 0.6445 | 0.6819 | 0.6026 | 14 | 2,073 s |
| 12 | `broad-adamw-0002` | 0.4889 | 0.6166 | 0.6147 | 0.6280 | 10 | 2,608 s |
| 13 | `broad-adamw-0005` | 0.4526 | 0.6100 | 0.7153 | 0.5351 | 10 | 2,760 s |

The 0.001 AdamW branch was pruned before execution: increasing the rate from
0.0002 to 0.0005 reduced the primary metric by 0.0363 and both were already
below the reference. This is a declared early-termination decision, not an
unreported failed run. Freezing the first ten modules did not arrest the loss,
so a validation-stage refinement tested the head alone (`freeze=23`) at smaller
rates. The pending mosaic and class-weight controls were moved from 0.00005 to
the new 0.0001 leader before either ran, keeping each mechanism comparison at
the strongest measured rate. The interpolation grid was fixed at 0.10, 0.25,
0.50 and 0.75 before it
ran; it preserves most pretrained weights while incorporating the historical
fine-tune. Alpha 0.25 narrowly beat alpha 0.50 on the primary metric, while
alpha 0.50 retained slightly better mAP50, mAP75 and recall.

Removing mosaic reached 0.5329: it retained slightly more recall than the
leader but lost 0.0069 mAP50-95 and materially weakened motorcycle AP. Default
mosaic is therefore retained; the lower-augmentation branch is not a finalist.

`cls_pw=0.25` reached 0.5388 but did not lead and improved only truck AP among
the six targets. Source inspection found that Ultralytics computes inverse
frequency over all 80 retained COCO outputs and substitutes count one for the
74 unannotated classes. The resulting weights were 0.151–0.306 for the six
annotated classes and 1.060 for each absent class. It is thus a mixed
rare-target/absent-output regularizer, not a clean six-class balance control.
The simpler unweighted candidate has the higher primary metric and more stable
per-class AP, so weighting is rejected.

A fixed-resolution check of the current head-only leader measured 0.5390 at
640 and 0.3949 at 960; inference rose from 9.3 to 20.1 ms/image. The 960 branch
was therefore pruned. A short high-resolution training run would not be a fair
full-budget comparison, while a full run would spend substantial compute on a
deployment point already contradicted by both accuracy and latency evidence.
The remaining stability stage was stopped at the user's request after the
selected result had been reproduced within 0.0002 mAP50-95.

## Final Hyperparameters

| Parameter | Value |
| --- | --- |
| Base checkpoint | `models/yolo26n.pt` |
| Trainable modules | detection head only (`freeze=23`) |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Final learning-rate factor | 0.1 |
| Warmup | disabled |
| Image size | 640 |
| Batch | 8 |
| Seed | 42 |
| Mosaic | default; close for final 5 planned epochs |
| Selected epoch | 6 of 9 completed epochs |

The configured maximum was 20 epochs with patience 6. Training was stopped by
the user during epoch 10 after the recovered result was accepted; selection
always used the separately saved `best.pt`.

## Validation Results

An independent validation pass over 200 images and 464 objects produced:

| mAP50-95 | mAP50 | mAP75 | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5404 | 0.6650 | 0.5675 | 0.6673 | 0.6400 | 0.6534 |

Per-class mAP50-95 was 0.2483 person, 0.5694 bicycle, 0.4961 car,
0.6319 motorcycle, 0.7943 bus, and 0.5021 truck. The validation-selected
operating confidence is 0.2182.

## Validation Object Accounting

At confidence 0.2182 and matching IoU 0.50, the accounting reconciled all 464
ground-truth objects: 228 TP, 129 FP, and 236 FN. This fixed-threshold view has
precision 0.6387, recall 0.4914, F1 0.5554, and count error -107. It is not the
threshold-integrated COCO metric reported above.

## Test Results

Not run. The user stopped the workflow after validation and export.

## Test Object Accounting

Not produced because the test split was not run.

## Validation vs Test

Not available because the test split was not run.

## Previous Model Comparison

Validation-only comparisons are reported in the Baseline and Hyperparameter
Search sections. No new test comparison was run.

## Error Analysis

The validation accounting found the largest count deficit on person (-101).
Motorcycle (+5) and truck (+7) were over-counted. Generated ranked examples
remain local under ignored `output/` and are not part of the committed model.

## Issues Found and Resolved

- At 17:12 CEST, after the narrow search completed and before the first
  stability run, the ignored `output/` tree was deleted and recreated by an
  external process. No command in this workflow deleted it, no recoverable
  Trash or open-file copy was available, and repository source/model files were
  unaffected. The selected configuration was rerun and exported; earlier
  search measurements remain transcribed in this report, but
  their deleted machine artifacts are not claimed to survive. This is an
  artifact-retention incident, not a metric or dataset correction.
- The historical training notebook had already evaluated TEST-ID and TEST-OOD
  before this request. Therefore the repository's test data was not literally
  unseen at the start of this workflow. The new search definition was fixed
  without using those results, training receives a YAML with only train/val,
  and test execution is code-gated behind a hashed frozen selection. This
  limitation will remain disclosed; strict historical test secrecy cannot be
  retroactively restored.
- The first metric exporter treated `Metric.ap` as a class-by-IoU matrix. In
  Ultralytics 8.4.48 it is one-dimensional; `Metric.all_ap` is authoritative.
  The failed export happened after a valid validation pass, changed no model,
  and the short pass was rerun with the correct field.
- A stricter audit initially rejected boxes touching an image boundary because
  eight-decimal YOLO serialization reconstructs some endpoints about `5e-9`
  outside `[0, 1]`. Inspection showed rounding error rather than malformed
  annotations. The audit now accepts a documented `1e-6` geometric tolerance;
  all labels pass and no dataset file was changed.

## Final Model

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| PT checkpoint 0.1.3 extended | `7b9a545a335e785a79292373aba0cad8686653c3cd87814c64f593a779a49bae` | 7,500,800 bytes |
| ONNX model 0.1.3 extended | `a38224fcb45cf90e4f898d64ce0254b5a1f1007309ac9bcd089f20a4961b7ad3` | 9,941,992 bytes |

The ONNX export uses opset 20, static input `[1, 3, 640, 640]`, output
`[1, 300, 6]`, and all 80 COCO classes. ONNX Runtime contract validation and
20 measured CPU inference passes succeeded; median zero-tensor latency on the
recorded Apple M4 host was 21.1 ms.

## Conclusions

The low-rate head-only update is the retained 0.1.3-extended model. It reproduces
the deleted winner within measurement noise and preserves the edge runtime
contract. Stability and test conclusions are deliberately not claimed because
those stages were stopped.
