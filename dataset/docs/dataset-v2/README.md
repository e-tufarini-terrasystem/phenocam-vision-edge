# Phenocam Vision public dataset (v2)

## Overview

Version 2 is the public training artifact used as the reproducible base of
dataset v3. It supports YOLO object detection for people and privacy-relevant
vehicles while preserving the COCO class IDs expected by the runtime. The
materialized artifact is named `training-dataset/` for compatibility with the
builder contract and `notebooks/training-v2.ipynb`.

This artifact contains training data only. It is not an operational validation
or test set and does not prove performance on deployment cameras.

## Content

| Source | Positive images | Negative images | Total |
| --- | ---: | ---: | ---: |
| Open Images V7 | 1,229 | 50 | 1,279 |
| PhenoCam Network v3 | 15 | 706 | 721 |
| **Total** | **1,244** | **756** | **2,000** |

The 1,244 positive images have matching YOLO label files; the 756 intentional
negatives have none. The labels contain 4,677 boxes across six classes.

| ID | Class | Meaning | Boxes |
| ---: | --- | --- | ---: |
| 0 | `person` | person | 3,051 |
| 1 | `bicycle` | bicycle | 200 |
| 2 | `car` | car, van, pickup | 772 |
| 3 | `motorcycle` | motorcycle or scooter | 204 |
| 5 | `bus` | bus or coach | 180 |
| 7 | `truck` | truck, tractor, combine, other self-propelled farm vehicle | 270 |

IDs 4 and 6 remain unused to preserve runtime-compatible COCO identifiers.

## Structure

```text
training-dataset/
├── images/train/
├── labels/train/
├── metadata/
│   ├── acceptance-audit.json
│   ├── checksums.sha256
│   ├── dataset-statistics.json
│   ├── deduplication-audit.json
│   ├── duplicate-groups.csv
│   ├── rejected-candidates.csv
│   ├── source-annotations.jsonl
│   ├── source-images.csv
│   ├── source-licenses.csv
│   └── training-images.txt
├── README.md
└── yolo-dataset.yaml
```

`metadata/source-images.csv` is the canonical one-row-per-image manifest.
`source-annotations.jsonl` preserves source and compiled annotations.
`source-licenses.csv` records attribution and the verified license URL.

## Split and annotation format

All 2,000 images belong to `train`; v2 intentionally has no `val` or `test`
split. Dataset v3 replaces this limitation with group-safe Train, Validation,
TEST-ID, and TEST-OOD partitions.

Each non-empty label line uses normalized YOLO detection format:

```text
class_id x_center y_center width height
```

Coordinates must be finite and in `[0, 1]`, with positive width and height.
An absent label is valid only when the manifest records the image as negative.

## Provenance and validation

Open Images provides diverse boxed objects and per-image provenance. PhenoCam
provides fixed-camera, seasonal, and natural-background imagery from 157
cameras/sites between 2001 and 2023. Model detections only prioritized human
review; they never established ground truth or verified negatives.

Materialization checks source integrity, image decoding, minimum dimensions,
box geometry, class floors, provenance, licenses, exact and perceptual
duplicates, review completeness, and checksums. The acceptance status is
`completed_with_single_reviewer_waiver`: all automatic gates passed, but a
second independent negative reviewer was unavailable.

## Use

The training notebook reads `dataset/training-dataset`. To rebuild or validate
the public artifact, follow [CREATION.md](CREATION.md). Full historical build
evidence remains in [../history/training-dataset.md](../history/training-dataset.md).

Known limitations are the missing validation/test split, the single-reviewer
waiver, strong source imbalance by class, and the absence of operational-domain
performance evidence.
