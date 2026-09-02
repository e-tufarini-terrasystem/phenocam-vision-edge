# Phenocam Vision training data

This directory owns both the completed public training dataset and the local,
reproducible construction pipeline. The distributable `training-dataset/` never
contains internal operational imagery. The private v3 artifact adds reviewed
operational splits without adding them to training. Its verified expanded build
also adds 18 human-reviewed public PhenoCam positives to the public training
split.
Unreviewed V3 mining data and local paths remain under the ignored `workspace/`
boundary.

The canonical training artifact is [`training-dataset/`](training-dataset/).
Its versioned composition and limitations are documented in
[`DATASET.md`](DATASET.md), which is copied to
`training-dataset/README.md` during materialization.

## Current status

The 2,000-frame YOLO dataset is complete with status
`completed_with_single_reviewer_waiver`:

- 1,244 positive and 756 negative frames;
- 1,279 Open Images V7 and 721 PhenoCam v3 frames;
- all automatic acceptance gates passed;
- independent negative verification is false because a second reviewer was
  unavailable;
- operational validation on internal imagery remains `pending`.

The adjacent `training-dataset-v3-expanded/` build contains 2,258 images:
2,018 public training images and 240 reviewed operational images. The public
expansion contributes 18 images and 104 human annotations from CVAT Task 11;
22 additional reviewed images remain reserved for a later cycle.

The waiver must remain visible in reports and training records. This dataset is
not evidence of production accuracy on internal camera imagery.

## Directory layout

- `training-dataset/`: ignored, materialized YOLO dataset ready for training;
- `training-dataset-v3/`: ignored, preserved pre-expansion v3 artifact;
- `training-dataset-v3-expanded/`: ignored, viewer-compatible expanded v3 with
  the reviewed public addition and separate operational splits;
- `DATASET.md`: versioned description copied into the materialized dataset;
- `builder/`: deterministic dataset-construction package;
- `commands/`: shell entry points for building, review, CVAT, and finalization;
- `config/`: versioned dataset contract and fixed invariants;
- `docs/`: normative specifications, operating guidance, and historical status;
- `tests/`: builder-specific automated verification;
- `workspace/`: ignored downloads, source-specific state, reviews, and tools;
- `.venv/` and `.env`: ignored local Python environment and credentials.

The ignored workspace is arranged by responsibility:

```text
workspace/
├── annotation/
├── deduplication/
├── history/
├── models/
├── quarantine/
├── reviews/
├── sources/
│   ├── open-images/
│   └── phenocam/
└── tools/
```

Credentials stay outside the training artifact. In particular,
`workspace/annotation/cvat/credentials.json`, `.env`, and CVAT state must never
be copied into `training-dataset/` or committed.

## Builder commands

Run commands from the repository root:

```sh
dataset/commands/dataset-builder.sh setup
dataset/commands/dataset-builder.sh status
dataset/commands/dataset-builder.sh preflight
dataset/commands/dataset-builder.sh test
```

The completed workflow can be reproduced with the staged builder commands and
then materialized with:

```sh
dataset/commands/dataset-finalization.sh build
dataset/commands/dataset-finalization.sh build-v3
dataset/commands/dataset-finalization.sh build-v3 dataset/training-dataset-v3-expanded
```

The builder refuses to overwrite an existing `training-dataset/`. Move or
archive an existing artifact deliberately before rebuilding it.
The optional v3 destination supports an adjacent, reversible verification build.
Reviewed public PhenoCam additions enter training only through the audited
`included/reserved/rejected` selection described in `docs/development.md`.

Run the tests directly with:

```sh
dataset/.venv/bin/python -m unittest discover -s dataset/tests -v
```

## Dataset viewer

Open [`viewer.html`](viewer.html) in a browser and select either the complete
`dataset/training-dataset/` or `dataset/training-dataset-v3/` directory. The
local-only viewer associates each YOLO label with its image, draws the bounding
boxes, and supports filename, positive/negative, and object-class filters with
image counts without uploading dataset files. V3 filenames can be filtered by
`raspberrypi2.local` or `sitets02`. Click the selected image name to select it
for copying; use the left and right arrow keys to move between images. Zoom with
the mouse wheel, the controls below the image, or the `+`, `-`, and `0` keys.
Drag an enlarged image with the primary mouse button to pan it.

## Annotation commands

CVAT Community `v2.71.0` is pinned in the ignored workspace:

```sh
dataset/commands/cvat-server.sh setup
dataset/commands/cvat-server.sh start
dataset/commands/cvat-server.sh cli-setup
dataset/commands/cvat-server.sh create-superuser
dataset/commands/cvat-server.sh open
```

The positive task wrapper is:

```sh
dataset/commands/cvat-tasks.sh profile
dataset/commands/cvat-tasks.sh list
dataset/commands/cvat-tasks.sh upload-open-images
dataset/commands/cvat-tasks.sh upload-phenocam
dataset/commands/cvat-tasks.sh upload-open-images-supplement
dataset/commands/cvat-tasks.sh upload-v3-public-teacher
```

The complete historical annotation procedure remains in
[`docs/annotation-guide-it.md`](docs/annotation-guide-it.md). Human decisions
remain authoritative; baseline detections may prioritize review but cannot
establish ground truth or verify a negative.

## Data sources and security boundary

NASA Earthdata credentials may be configured in ignored `dataset/.env` as
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or in a mode-`0600` `.netrc`.
Copy `.env.example` for a new workstation and keep `.env` at mode `0600`.
Credentials, signed URLs, local paths, and stack traces are not written to the
training artifact.

External images, archives, metadata, CVAT exports, and review CSVs are treated
as untrusted input. The builder validates size limits, archive membership,
image decoding, dimensions, checksums, required columns, source identity, box
geometry, licenses, and review completeness before accepting data.

## Documentation

- [`DATASET.md`](DATASET.md): final composition, files, licenses, and limits;
- [`DATASET_V3.md`](DATASET_V3.md): private v3 composition, provenance, and
  training boundary;
- [`docs/dataset-plan-en.md`](docs/dataset-plan-en.md): normative build contract;
- [`docs/dataset-plan-it.md`](docs/dataset-plan-it.md): concise Italian plan;
- [`docs/annotation-guide-it.md`](docs/annotation-guide-it.md): manual review;
- [`docs/next-steps-it.md`](docs/next-steps-it.md): completed state and boundary;
- `docs/status/` and `docs/history/`: preserved development history.
- [`../docs/status/training-v3-expansion-review-2026-09-01.md`](../docs/status/training-v3-expansion-review-2026-09-01.md):
  measured pre-expansion audit and quantitative admission policy.
