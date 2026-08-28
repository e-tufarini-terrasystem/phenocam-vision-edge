<!--
Architectural purpose: authoritative, executable specification for building and
auditing the first mixed object-detection dataset used by this repository.
System context: it fixes data, ontology, provenance, split, and quality gates
while deliberately leaving download and training implementation to later work.
-->

# External mixed detection dataset: iteration 1

Status: approved specification

Decision date: 2026-08-26

Deterministic sampling seed: `20260826`

Implementation decision, 2026-08-27: this iteration builds only the public
training dataset. Internal inventory, operational validation, and any blind
test are deferred to a second phase and must remain explicitly recorded as
`operational_validation: pending`. Deferral does not authorize a substitute
public validation split or the inclusion of internal imagery in training.

## 1. Experiment and non-negotiable boundary

Iteration 1 must answer one question:

> How well does a carefully designed external mixed dataset generalize to the
> real internal PhenoCam/Agrocam deployment domain before target-domain images
> are introduced into training?

The training dataset must therefore contain only approved external/public
sources. Its download, filtering, annotation normalization, deduplication,
balancing, sampling, and manifests must complete without access to the internal
archive.

Internal imagery is cleared for model development, training, validation, and
testing, but its first-iteration role is operational validation, plus a blind
held-out test set only when enough independent groups exist. It must not be
silently or automatically added to training.

If no internal imagery is available, publish the completed public training
dataset and report `operational_validation: pending`. Absence of internal data
is not a reason to delay or change the public mixture.

This document specifies dataset construction only. It does not authorize a
model download, dataset download, training run, runtime change, or license
decision for commercial deployment.

## 2. Runtime compatibility invariant

The compiled labels must remain compatible with the current repository:

- Ultralytics detection model with the canonical 80 COCO class-name table;
- `640 × 640` RGB letterboxed model input;
- operational class IDs `0`, `1`, `2`, `3`, `5`, and `7`;
- current full-frame and adaptive-crop inference path;
- current confidence and duplicate-suppression behavior.

The dataset `data.yaml` must retain all 80 canonical COCO names in their
canonical order. Annotation rows may use only these six IDs:

| ID | Compiled class | Accepted source meaning |
|---:|---|---|
| 0 | `person` | person |
| 1 | `bicycle` | bicycle, including a bicycle without a rider |
| 2 | `car` | car, van, pickup |
| 3 | `motorcycle` | motorcycle, motor scooter |
| 5 | `bus` | bus, coach |
| 7 | `truck` | truck, tractor, combine, other self-propelled agricultural vehicle |

Do not collapse these classes into `person` and `vehicle`. A two-class ontology
requires a separate model/runtime experiment and is out of scope here.

Every object must retain its original fine-grained source label in source
metadata. For example, a `combine` becomes class ID `7` in YOLO while
`source_class=combine` and `vehicle_subtype=combine` remain available for audit
and error analysis.

## 3. Existing-system baseline

The current artifact is an Ultralytics YOLO26n detector exported to ONNX. Its
metadata identifies Ultralytics `8.4.48`, detection task, 80 COCO classes, an
FP32 `1 × 3 × 640 × 640` RGB input, and a `[1, 300, 6]` output of bounding box,
confidence, and class. Runtime deployment targets a Raspberry Pi 3 CPU.

Inference applies EXIF orientation, RGB conversion, and `(114, 114, 114)`
letterboxing. It evaluates one full image and 15 overlapping adaptive crops:
`5 × 3` for horizontal/square images or `3 × 5` for vertical images, with 20%
overlap. The default confidence threshold is `0.30`. Cross-view duplicates are
suppressed when intersection-over-union or smaller-box coverage is at least
`0.50`; car, bus, and truck detections compete during this suppression.

These facts shape sampling and evaluation. In particular, an object is measured
in its best production view, and dataset construction must include difficult
small objects, borders, occlusion, and backgrounds that exercise crop merging.

The repository currently contains no training pipeline, dataset YAML, training
hyperparameters, or accuracy-evaluation pipeline. Existing inference tests are
runtime tests, not a ground-truth benchmark. The small set of previously viewed
internal images is development material and cannot become a blind test set.

## 4. Approved public sources

Only the following sources are approved for iteration 1.

### 4.1 PhenoCam Network v3

Purpose: deployment-like fixed-camera backgrounds, agriculture/nature context,
seasonal appearance, ordinary negatives, and manually annotated positives.

- Dataset page: <https://daac.ornl.gov/VEGETATION/guides/Phenocam_Images_V3.html>
- Dataset-wide license: CC BY 4.0.
- Fair-use statement: <https://zenodo.org/records/14854980/files/Fair-use_Statement.pdf?download=1>
- Annotation status: no suitable person/vehicle boxes; selected positives must
  be manually and completely annotated.
- Sampling unit: original source frame, grouped by site/camera and event/time.

Record the dataset version, landing page, original object URL, site, camera,
timestamp, attribution, license URL, retrieval date, and checksums.

### 4.2 Open Images V7

Purpose: diverse, already boxed people and vehicle subclasses, rare classes,
occlusion/truncation attributes, and verified negatives.

- Facts and annotation license: <https://storage.googleapis.com/openimages/web/factsfigures_v7.html>
- Download and metadata: <https://storage.googleapis.com/openimages/web/download_v7.html>
- Annotation license: CC BY 4.0.
- Image license listing: CC BY 2.0, subject to per-image verification.

An Open Images candidate is eligible only if its individual image-license and
attribution record can be resolved and stored. Missing or contradictory image
provenance is an exclusion, not a warning.

### 4.3 Explicitly excluded sources

Do not substitute another dataset merely to fill a quota. The following are
excluded from iteration 1:

| Source | Exclusion reason |
|---|---|
| COCO | Redundant with COCO-pretrained baseline; per-image Flickr provenance burden |
| BDD100K | Commercial data rights not authoritatively verified for this use |
| VisDrone | No sufficiently clear dataset-level commercial license verified |
| AU-AIR | Repository software license does not clearly license images/annotations |
| CrowdHuman | Official terms restrict use to non-commercial research |
| KITTI | Non-commercial/share-alike dataset terms |
| Cityscapes | Non-commercial use and redistribution restrictions |
| Mapillary object data | Research-only/no-commercial and derivative restrictions |

If an approved source becomes unavailable or its terms change, stop the affected
source intake, preserve the rejection/audit records, and request a revised
composition. Never silently replace it.

## 5. Dataset size and source mixture

Percentages are measured over unique original source frames before crop
materialization. Crops never count as new source frames.

### 5.1 Recommended build

| Source | Positive | Negative | Total | Share |
|---|---:|---:|---:|---:|
| PhenoCam Network v3 | 15 | 706 | 721 | 36.05% |
| Verified Open Images V7 | 1,229 | 50 | 1,279 | 63.95% |
| **Total** | **1,244** | **756** | **2,000** | **100%** |

This makes the source-frame dataset 62.2% positive and 37.8% negative. A positive
contains at least one completely annotated target object. A negative has no live
target objects and has been independently verified as such.

This composition supersedes the provisional 350/750 PhenoCam and 850/50 Open
Images contract after exhaustive human annotation found only 15 genuine
PhenoCam positives among the 350 positive candidates. The remaining qualified
PhenoCam material is retained as the fixed-camera negative domain bridge. The
379-frame Open Images supplement is selected deterministically from the already
licensed, downloaded, and globally deduplicated pool.

### 5.2 Build checkpoints

| Checkpoint | PhenoCam | Open Images | Minimum target instances | Use |
|---|---:|---:|---:|---|
| Recommended | 721 | 1,279 | 2,500 | first complete experiment |

Expand from 2,000 to 3,000 frames only if the 1,200-to-2,000 learning curve
improves image-level recall by at least one percentage point at a fixed false
alarm rate of at most 5% on operational validation. If operational validation is
still pending, do not authorize the optional expansion on training metrics alone.

The class-instance floors apply to the combined 1,244 positive frames. The
supplemental selector enforces those floors against the audited Open Images base
and the 15 confirmed PhenoCam positives. It also preserves the global limit of
five frames per Open Images provenance group.

## 6. Semantic composition

Assign each source frame exactly one primary sampling stratum:

| Primary stratum | Frames | Share | Definition |
|---|---:|---:|---|
| Ordinary fixed-camera negatives | 456 | 22.8% | target-free deployment-like scenes |
| Hard negatives/confusers | 300 | 15% | target-free scenes likely to trigger the detector |
| Common/easier positives | 500 | 25% | clear person/car-family examples |
| Difficult positives | 500 | 25% | small, distant, occluded, or border-truncated targets |
| Rare/environment positives | 244 | 12.2% | bicycles, motorcycles, buses, machinery, unusual conditions |
| **Total** | **2,000** | **100%** | |

Secondary tags may overlap. Across the 1,244 positive frames, require:

- at least 30% containing a small or very small target in its best production view;
- at least 20% containing an occluded or border-truncated target;
- at least 10% containing multiple or overlapping targets;
- no more than 50% classified as easy.

Measure the target short side after projection into the best repository-style
`640 × 640` full/crop view:

| Size tag | Short side |
|---|---:|
| `very_small` | `< 16 px` |
| `small` | `16–32 px` |
| `medium` | `33–96 px` |
| `large` | `> 96 px` |

The complete recommended build must satisfy these instance floors:

| Compiled class | Minimum instances |
|---|---:|
| `person` | 900 |
| `car` family | 750 |
| `truck` family | 270 |
| `bicycle` | 200 |
| `motorcycle` | 200 |
| `bus` | 180 |

The total must be at least 2,500 instances. Floors may overlap within frames and
are not a request to duplicate images. Prefer additional qualified frames over
oversampling files.

## 7. Domain and diversity quotas

For PhenoCam selection:

- use at least 100 distinct sites where the archive permits;
- select no more than 12 source frames per site;
- for ordinary negatives, select at most one frame per site/camera/day;
- include at least seven distinct days for a repeated site/time negative pattern;
- target at least 15% from each season where source coverage permits;
- target approximately 70% normal daylight, at least 15% low light, at least
  10% adverse weather/visibility, and no more than 5% night imagery;
- keep night frames only when night operation is an explicit deployment need.

For Open Images, allow no more than five selected images from the same
author/sequence/provenance group. Prefer geographic, viewpoint, background, and
object-scale diversity over visually redundant quota filling.

Quota tolerances at acceptance are ±1 percentage point for source proportions
and ±2 percentage points for positive/negative and semantic strata. Instance
floors and provenance requirements have no negative tolerance.

## 8. Deterministic candidate selection

Process candidates in this order:

1. Resolve source version, image license, attribution, and original URL.
2. Decode the image and apply basic eligibility checks.
3. Normalize metadata and assign source groups and semantic tags.
4. Remove exact and near duplicates as specified below.
5. Stable-sort by `source_dataset`, `source_id`, `timestamp`, `original_url`.
6. Fill the rarest unsatisfied class/stratum quota first.
7. Within a stratum, use greedy farthest-point selection over fixed image
   embeddings to favor visual diversity.
8. Break equal-distance ties with pseudorandom order from seed `20260826`.
9. Record every rejection and replacement; never silently skip a candidate.

Use one documented, version-pinned embedding model for the whole build. Store
its identifier and preprocessing settings in the manifest. The embedding is a
sampling aid, not an annotation or ground-truth source.

The current detector may prioritize review candidates at confidence `0.05`:

- false detections on verified negatives;
- misses or low-confidence detections on true targets;
- disagreement between full-frame and crop views;
- likely class confusion among car, bus, and truck.

Model output never establishes that an object exists or that an image is
negative. A human/source annotation decision remains authoritative.

## 9. Annotation normalization

Maintain two distinct representations:

1. Source annotation: original class, fine subtype, original coordinates,
   attributes, provenance, and review history.
2. Compiled annotation: COCO-compatible class ID and YOLO normalized box used
   by training.

YOLO rows have exactly:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]` against the compiled image dimensions.
A negative image has no label file. Do not use an empty file as an ambiguous
placeholder for unreviewed annotation.

### 9.1 Open Images conversion

- Convert normalized `XMin`, `XMax`, `YMin`, `YMax` to normalized YOLO center,
  width, and height without a lossy pixel round trip.
- Preserve the Open Images label ID and readable source class.
- Map only exact approved types and reviewed subclasses to the six classes.
- Reject a generic or ambiguous `Vehicle` superclass unless its fine type can
  be established by source metadata or human review.
- Exclude `IsGroupOf=1` frames unless each visible target individual is boxed.
- Treat `IsDepiction=1` as a confuser/negative object, not a live target.
- Preserve occlusion and truncation flags.
- Reject a selected frame if any visible target object lacks a usable box.

### 9.2 PhenoCam annotation

- Manually box every visible live target belonging to the six classes.
- Use the tight visible extent; do not hallucinate the hidden full extent.
- Mark occlusion and truncation separately in source metadata.
- Review person and vehicle completeness independently.
- Exclude any frame with unresolved target presence or class ambiguity because
  standard YOLO text cannot represent ignore regions safely.

Do not label an unpowered trailer without its powered vehicle, a static
implement, wheelchair, boat, train, aircraft, printed image, screen image, or
ambiguous vehicle superclass as one of the six live-target classes.

## 10. Crop materialization

Training must preserve original source-frame accounting while exposing the
runtime's scale and border conditions.

For each accepted source frame:

- materialize the full frame;
- materialize up to two positive crops using the repository's exact adaptive
  crop geometry when they materially improve target scale or border coverage;
- materialize up to one hard-negative crop when the baseline detector emits a
  false detection at confidence at least `0.05`;
- clip and transform boxes deterministically;
- discard a crop if clipping makes its target annotation ambiguous;
- inherit source group, provenance, license, and eventual split from the parent.

No crop may cross a split independently of its parent. Compiled crops must have
unique derivative IDs and record parent ID, crop rectangle, geometry version,
and transformation parameters.

## 11. Deduplication and leakage prevention

Apply all of these signals before sampling acceptance:

1. Original-file SHA-256.
2. SHA-256 of decoded, EXIF-normalized RGB pixels plus dimensions.
3. 64-bit perceptual hash; Hamming distance `≤ 6` creates a duplicate candidate.
4. SSCD-style embedding similarity:
   - cosine `≥ 0.98`: automatic candidate edge;
   - cosine `0.95–0.98`: manual review;
   - calibrate on at least 200 labeled pairs and tighten by `0.01` until false
     merges are at most 1%.
5. Metadata linkage: URL, author, timestamp, burst/sequence, site, and camera.

Build connected duplicate/sequence components and assign each component one
`group_id`. A group is indivisible for selection and split assignment. Split
conflict resolution priority is `test > validation > training`; the lower-priority
copy is removed or moved with the whole group.

Static-background similarity alone does not prove duplication. Distinct target
events or hard confusers from the same fixed camera may remain when temporal
metadata establishes independence and the site quota is respected.

## 12. Internal operational validation and test protocol

Internal PhenoCam/Agrocam imagery is a separate restricted-domain dataset. It is
not part of the 2,000-frame public training mixture or its percentages.

When it arrives:

1. Inventory size, sites, cameras, timestamps, seasons, classes, target scale,
   and failure-mode coverage before deciding any split.
2. Normalize provenance and annotation with the same semantic rules.
3. Group by `site_id + camera_id + sequence/event`; when exact event metadata is
   absent, derive a conservative temporal/visual group and record the method.
4. Assign whole groups, never frames, to operational validation or blind test.
5. Prefer chronologically later independent groups for the blind test.

Operational validation may guide confidence thresholds, hyperparameters,
dataset-composition decisions, and model selection. The blind test must never
guide training, threshold tuning, hyperparameters, composition, or iterative
selection.

Create a blind test only when it can contain independent coverage of at least:

- 60–100 verified negative images;
- 50 person-positive images;
- 75 vehicle-positive images;
- more than one site/camera wherever the archive permits.

These are minimum coverage targets, not a guarantee of statistical power. Also
report per-site uncertainty and group counts. If the archive cannot support both
meaningful validation and test sets, assign it to operational validation/development,
state that no final production-performance claim can be made, and create a blind
test only after additional independent data arrive.

Any internal images already inspected during repository exploration remain
development data and are ineligible for a blind test.

After training the public-only model, evaluate operational failure modes. Record
at least missed people/vehicles, small or distant objects, occlusion, border
truncation, agricultural machinery, recurrent false positives, difficult
backgrounds, and site/camera-specific patterns.

Only a later approved iteration may place selected internal examples into
training. Do not predefine an internal percentage. First report archive size,
diversity, site distribution, temporal coverage, class distribution, and
failure-mode coverage; then submit a revised composition that preserves
independent internal validation/test groups.

## 13. Automated and human quality control

### 13.1 Automatic checks

- every image decodes after EXIF orientation and is compiled as RGB;
- width is at least 640 and height at least 480;
- aspect ratio is between `0.5` and `2.0`, unless explicitly approved;
- every box is finite, non-zero, ordered, and within image bounds;
- clipping corrects no more than one source pixel of numerical overflow;
- class IDs are restricted to `0, 1, 2, 3, 5, 7`;
- exact duplicate label rows are rejected;
- provenance and license fields are present;
- checksums and derivative transformations reproduce;
- no duplicate/source group crosses a dataset boundary.

Warnings requiring review include a target short side below four original
pixels, severe blur/compression/darkness/lens obstruction, same-class box IoU
`≥ 0.95`, conflicting labels, and unusual box aspect ratio.

### 13.2 Human review

- verify every PhenoCam target box and frame-level completeness;
- independently verify every negative, preferably with a second reviewer;
- inspect all automatic duplicate candidates and ambiguous vehicle mappings;
- sample at least 10% of accepted Open Images boxes for class and box quality;
- review 100% of hard negatives and rare-class frames;
- record annotator, reviewer, timestamps, decision, and reason.

### 13.3 Mandatory exclusions

Exclude corrupt files, inconsistent dimensions, missing license/provenance,
invalid boxes, image/annotation mismatches, incomplete target annotations,
unresolved ambiguity, duplicate leakage, unverified negatives, or source terms
that do not permit the intended use.

## 14. Manifests and output contract

The builder must produce this logical output; physical storage may be external,
but paths and schemas must remain equivalent:

```text
mixed-dataset/
├── data.yaml
├── images/
│   └── train/
├── labels/
│   └── train/
├── source-annotations/
└── manifests/
    ├── train.txt
    ├── sources.csv
    ├── licenses.csv
    ├── groups.csv
    ├── rejections.csv
    ├── statistics.json
    └── checksums.sha256
```

The internal restricted dataset, when available, must use separate storage and
manifests. It must not be redistributed with the public mixed dataset.

At minimum, source/annotation records contain:

```text
image_id, derivative_id, parent_id, source_dataset, source_version, source_id,
original_url, landing_url, author, license_url, license_version,
license_verification_date, attribution, retrieval_date, site_id, camera_id,
sequence_id, event_id, timestamp, group_id, source_class, compiled_class,
vehicle_subtype, source_bbox, compiled_bbox, occlusion, truncation, size_tag,
lighting, weather, confuser, primary_stratum, review_status, annotator, reviewer,
source_sha256, decoded_sha256, phash, embedding_model, split, rejection_reason
```

`statistics.json` must report source, class, subtype, positive/negative, primary
stratum, size, occlusion, truncation, site, season, lighting, weather, and review
counts over both source frames and compiled derivatives.

`rejections.csv` must include candidate identity, stage, deterministic reason
code, explanatory note, and replacement identity where applicable.

## 15. Acceptance gates

The public mixed dataset is complete only when all gates pass:

- exactly 2,000 accepted unique source frames for the recommended build;
- source mix within ±1 percentage point;
- positive/negative and semantic strata within ±2 percentage points;
- all instance floors and the 2,500 total-instance floor are met;
- every image decodes and every compiled box passes validation;
- license/provenance completeness is 100%;
- every negative has an independent human/source verification;
- no duplicate, parent, site-event, or sequence group crosses a boundary;
- manifests recreate the selected order with seed `20260826`;
- source and compiled checksums verify;
- manual-review obligations are complete;
- internal status is explicitly `available`, `insufficient`, or `pending`.

A dataset can pass these construction gates while operational validation remains
pending. In that case, describe it as a completed public training dataset, not a
production-validated detector dataset.

## 16. Evaluation and iteration decision

Train the first model only after this dataset is accepted. Compare it with the
unchanged baseline using identical runtime preprocessing and report per-class and
image-level precision/recall, false alarms per image, object-size buckets,
occlusion/truncation, full-frame versus crop behavior, and per-site results.

Tune thresholds only on operational validation. Freeze the model and thresholds
before blind-test evaluation. Never promote training or public validation
metrics as evidence of internal deployment performance.

The second-iteration proposal must connect each requested data change to a
measured internal failure mode. Candidate internal training examples include
misses, small/distant targets, occlusion/border truncation, agricultural
machinery, recurrent false positives, difficult backgrounds, and camera/site
domain characteristics. Approval must precede any such mixing.

## 17. Reproducibility and security

- Pin downloader, annotation converter, embedding model, and hashing versions.
- Treat archives, filenames, metadata, and images as untrusted input.
- Reject path traversal, archive links, decompression bombs, malformed images,
  oversized dimensions, and unexpected MIME/content mismatches.
- Download to a staging area; do not execute source-provided files.
- Store no credentials, signed URLs, or internal filesystem paths in manifests.
- Keep internal site/camera identifiers access-controlled and pseudonymize
  exported analysis when exact identity is unnecessary.
- Record commands, tool versions, seed, configuration hash, source versions,
  timestamps, and operator/reviewer identities needed for an audit.

The current model artifact advertises an AGPL-3.0 Ultralytics license. Resolve
model/runtime licensing separately before commercial deployment; dataset
eligibility does not settle model-license obligations.

## 18. Evidence informing the design

The composition and controls follow established findings on dataset bias,
domain shift, hard-example mining, long-tail sampling, core-set diversity, and
copy detection:

- [Unbiased Look at Dataset Bias](https://www.ri.cmu.edu/publications/unbiased-look-at-dataset-bias/)
- [Domain Adaptive Faster R-CNN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf)
- [Online Hard Example Mining](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
- [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
- [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf)
- [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://arxiv.org/abs/1708.00489)
- [Revisiting Unreasonable Effectiveness of Data](https://openaccess.thecvf.com/content_ICCV_2017/papers/Sun_Revisiting_Unreasonable_Effectiveness_ICCV_2017_paper.pdf)
- [A Self-Supervised Descriptor for Image Copy Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.pdf)
- [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/)
