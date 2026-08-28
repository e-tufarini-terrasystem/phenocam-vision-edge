<!--
This document is the versioned description of the generated training dataset.
The builder copies it into training-dataset/README.md during materialization.
-->

# Phenocam Vision training dataset

This directory contains the public object-detection dataset assembled for
Phenocam Vision Edge. It combines Open Images V7 and PhenoCam v3 material in
YOLO detection format. It contains training data only: it is not an operational
validation or test set for the deployed camera domain.

## Composition

The accepted composition contains exactly 2,000 unique source frames.

| Source | Positive | Negative | Total |
|---|---:|---:|---:|
| Open Images V7 | 1,229 | 50 | 1,279 |
| PhenoCam v3 | 15 | 706 | 721 |
| **Total** | **1,244** | **756** | **2,000** |

Every positive image has one matching YOLO label file. Negative images have no
label file; their absence is intentional and is recorded in
`metadata/source-images.csv`.

The primary semantic strata are:

| Stratum | Frames |
|---|---:|
| Common positive | 500 |
| Difficult positive | 500 |
| Rare-environment positive | 244 |
| Hard negative | 300 |
| Ordinary fixed-camera negative | 456 |

The final labels contain these instances:

| YOLO ID | Class | Instances |
|---:|---|---:|
| 0 | person | 3,051 |
| 1 | bicycle | 200 |
| 2 | car | 772 |
| 3 | motorcycle | 204 |
| 5 | bus | 180 |
| 7 | truck | 270 |

IDs 4 and 6 are intentionally unused so that the compiled classes retain their
COCO-compatible runtime identifiers.

## Directory layout

```text
training-dataset/
├── README.md
├── yolo-dataset.yaml
├── images/
│   └── train/
├── labels/
│   └── train/
└── metadata/
    ├── acceptance-audit.json
    ├── checksums.sha256
    ├── dataset-statistics.json
    ├── deduplication-audit.json
    ├── duplicate-groups.csv
    ├── rejected-candidates.csv
    ├── source-annotations.jsonl
    ├── source-images.csv
    ├── source-licenses.csv
    └── training-images.txt
```

`images/train/` contains the 2,000 compiled JPEG images. File names expose the
source family and stable source identifier, for example
`open-images-test-001eb42ffe74be81.jpg` or
`phenocam-neon.d02.serc.dp1.00033_2017_06_29_170005.jpg`.

`labels/train/` contains normalized YOLO boxes for positive frames. Each label
has the same stem as its image. `yolo-dataset.yaml` is the training entry point.

## Metadata

- `source-images.csv` is the canonical one-row-per-frame manifest. It records
  source identity, source and compiled checksums, provenance, review status,
  split, and relative image and label paths.
- `source-annotations.jsonl` preserves both source annotations and compiled
  annotations, including imported human-review data when applicable.
- `source-licenses.csv` records the landing page, license URL, attribution, and
  license-verification date for every accepted source.
- `duplicate-groups.csv` records the duplicate group and split assigned to each
  source frame.
- `rejected-candidates.csv` records candidates rejected during the definitive
  negative review and their deterministic reason code.
- `dataset-statistics.json` contains the accepted source, polarity, semantic
  stratum, and class-instance counts.
- `deduplication-audit.json` records the final exact/perceptual duplicate audit
  and pinned SSCD model identity.
- `acceptance-audit.json` records the final gates, review protocol, limitations,
  and operational-validation status.
- `training-images.txt` lists every training image relative to `metadata/`.
- `checksums.sha256` covers every distributed file except itself.

## Selection and annotation

Open Images V7 supplied 1,229 positives and 50 human-checked negatives. The
positive pool was screened for image validity, licensing, class eligibility,
difficulty, diversity, and duplicate groups. A supplemental deterministic
selection repaired the class-instance floors after CVAT review.

PhenoCam v3 supplied 721 frames from a seasonally distributed, multi-site pool.
The initial positive candidates were reviewed in CVAT; 15 contained accepted
targets and 335 were confirmed empty. The remaining negative pool was selected
with SSCD diversity controls and reviewed through local blind-review packets.

Model detections were used only to prioritize human review. They were never
treated as ground truth. Positive annotations came from eligible source boxes
or reviewed CVAT exports. Negative acceptance came from explicit human/source
verification.

The final composition was checked for source integrity, image decoding, box
validity, class floors, provenance, licenses, exact duplicates, perceptual
duplicates, and source-group leakage before materialization.

## Review limitation

The dataset status is `completed_with_single_reviewer_waiver`. A second
independent reviewer was unavailable, and the owner explicitly accepted the
resolved first pass. Therefore `independent_negative_verification` is false and
the negatives must not be described as independently verified.

Operational validation on internal deployment imagery is separate and remains
`pending`. Results obtained by training on this dataset do not establish
production recall or false-positive rates.

## Sources and licenses

Open Images source images retain their per-image attribution and accepted
Creative Commons license recorded in `metadata/source-licenses.csv`. Open Images
annotation metadata is licensed under CC BY 4.0.

PhenoCam frames come from PhenoCam Dataset v3 through ORNL DAAC and use the
license and attribution recorded for each row. Consumers must preserve the
provided attribution and verify that their intended redistribution and use
comply with every recorded source license.

## Integrity check

From `dataset/training-dataset/`, verify the distributed files with:

```sh
shasum -a 256 -c metadata/checksums.sha256
```

A valid copy contains 2,000 image paths, 1,244 label paths, no missing or
unreferenced image/label files, and all checksums report `OK`.
