# Phenocam Vision dataset v3

## Overview

Dataset v3 is the canonical leakage-aware YOLO detection dataset for training,
validation, and separate in-domain and out-of-domain evaluation. It detects
people and privacy-relevant vehicles while preserving the COCO IDs required by
the edge runtime.

The materialized artifact is `dataset/dataset-v3/`. It contains public Open
Images and PhenoCam data plus reviewed private operational images. Private
images must not be redistributed.

## Content

| Source | Images |
| --- | ---: |
| Open Images V7 | 1,279 |
| PhenoCam Network v3 | 721 |
| Private operational cameras | 240 |
| **Total** | **2,240** |

There are 1,475 positive images with label files, 765 intentional negatives
without label files, and 10,562 object boxes. Public PhenoCam contributes 721
images from 157 cameras/sites covering 2001–2023. The private TEST-OOD images
come from `raspberrypi2.local` (2025) and `sitets02` (2026).

## Structure

```text
dataset-v3/
├── images/
│   ├── train/
│   ├── val/
│   └── test/{id,ood}/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/{id,ood}/
├── manifests/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── test-id.csv
│   └── test-ood.csv
├── metadata/
├── reports/
├── dataset.yaml
├── dataset-test-id.yaml
└── dataset-test-ood.yaml
```

`metadata/source-images.csv` is the canonical full manifest. The split
manifests are derived views. `reports/verification.json` is the machine-readable
integrity result; `reports/dataset-analysis.md` records measured composition and
split rationale.

## Splits

| Split | Images | Positive | Negative | Boxes | Dataset share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,600 | 995 | 605 | 3,750 | 71.43% |
| Validation (`val`) | 200 | 125 | 75 | 464 | 8.93% |
| TEST-ID | 200 | 124 | 76 | 463 | 8.93% |
| TEST-OOD | 240 | 231 | 9 | 5,885 | 10.71% |
| **Total** | **2,240** | **1,475** | **765** | **10,562** | **100%** |

The original 2,000 public images are partitioned 80/10/10. Train is used for
weights and augmentation; Validation for tuning, early stopping, and checkpoint
selection. TEST-ID and TEST-OOD are frozen and must not guide thresholds,
mining, or model selection. TEST-ID uses public sources with PhenoCam cameras
disjoint from Train/Validation. TEST-OOD contains only the two later private
operational cameras.

## Classes

| ID | Class | Meaning | Train | Val | TEST-ID | TEST-OOD |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | `person` | person | 2,442 | 306 | 303 | 180 |
| 1 | `bicycle` | bicycle | 160 | 20 | 20 | 0 |
| 2 | `car` | car, van, pickup | 624 | 74 | 74 | 5,558 |
| 3 | `motorcycle` | motorcycle or scooter | 163 | 20 | 21 | 37 |
| 5 | `bus` | bus or coach | 144 | 18 | 18 | 0 |
| 7 | `truck` | truck, tractor, combine, farm vehicle | 217 | 26 | 27 | 110 |

IDs 4 and 6 are intentionally unused. TEST-OOD has no bicycle and is strongly
car-heavy because it represents the available operational cameras rather than
an artificially balanced benchmark.

## Annotation format

Each line in a positive image's `.txt` file is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`; width and height are positive. A
negative image intentionally has no label file and must be marked negative in
the manifest.

## Provenance and leakage prevention

Open Images duplicate groups are indivisible. Each public PhenoCam camera/site
is indivisible and is joined with any connected pHash-near component at Hamming
distance at most 6. Private cameras are indivisible and restricted to TEST-OOD.
The deterministic allocation uses seed `42`, exact split sizes, and only then
minimizes class, season, and luminance imbalance.

Model detections were review suggestions, never ground truth. Public source
licenses and attribution are recorded in v2 metadata. Private operational rows
are explicitly marked non-redistributable.

## Use and validation

Train with `dataset/dataset-v3/dataset.yaml`. Use
`dataset-test-id.yaml` and `dataset-test-ood.yaml` for separate final
evaluations. The local viewer is `dataset/viewer.html`; select the whole
`dataset-v3` directory and choose Train, Validation, Test, Test ID, or Test OOD.

Verify the artifact from the repository root:

```sh
dataset/.venv/bin/python -m dataset.builder.partition verify dataset/dataset-v3
```

The verifier checks images and labels, class IDs and geometry, manifest
membership, exact split sizes, unique identities and hashes, camera/group
isolation, cross-split pHash distance, YAML portability, and checksums.

## Limitations

The public negatives retain the v2 single-reviewer waiver. TEST-OOD contains
only two private cameras, is class-imbalanced, and is not a general PhenoCam
benchmark. Weather and scene-condition labels are unavailable. Dataset v3
supports controlled evaluation but does not itself establish production recall,
false-positive rate, or generalization to unseen operational sites.

Construction and reproducibility are documented in [CREATION.md](CREATION.md).
