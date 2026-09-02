# Phenocam Vision training data

This directory owns both the completed public training dataset and the local,
reproducible construction pipeline. The distributable `training-dataset/` never
contains internal operational imagery. The private `dataset-v3/` artifact
partitions public data into train, validation, and TEST-ID, and reserves
reviewed operational imagery as TEST-OOD.
Unreviewed V3 mining data and local paths remain under the ignored `workspace/`
boundary.

The canonical training artifact is [`training-dataset/`](training-dataset/).
Its versioned composition and limitations are documented in
[`docs/dataset-v2/README.md`](docs/dataset-v2/README.md), which is copied to
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

The leakage-aware v3 artifact contains 2,240 images and 10,562 annotations:
1,600 train images, 200 validation images, 200 TEST-ID images, and 240 TEST-OOD
images. Public splits keep PhenoCam cameras and pHash-near components disjoint;
TEST-OOD contains the two later private operational cameras.

The waiver must remain visible in reports and training records. This dataset is
not evidence of production accuracy on internal camera imagery.

## Directory layout

- `training-dataset/`: ignored, materialized YOLO dataset ready for training;
- `dataset-v3/`: ignored, final leakage-aware YOLO v3 artifact;
- `dataset-v3-source/`: optional ignored unsplit v3 build input;
- `dataset-v3-expanded/`: optional, viewer-compatible expanded v3 with
  the reviewed public addition and separate operational splits;
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
cd dataset && .venv/bin/python -m builder.partition build dataset-v3-source dataset-v3
```

The builder refuses to overwrite an existing `training-dataset/` or partition
destination. Move or archive an existing artifact deliberately before rebuilding
it. `build-v3` defaults to the intermediate `dataset-v3-source/`; the partition
command creates the final artifact atomically.
Reviewed public PhenoCam additions enter training only through the audited
`included/reserved/rejected` selection described in `docs/development.md` and
must be requested explicitly with `--include-public-expansion`; they do not
silently alter the canonical 2,240-image v3 dataset.

Run the tests directly with:

```sh
dataset/.venv/bin/python -m unittest discover -s dataset/tests -v
```

## Dataset viewer

Open [`viewer.html`](viewer.html) in a browser and select
`dataset/dataset-v3/`. The Train, Validation, Test, Test ID, and Test OOD
selector reloads only the chosen subset. The
local-only viewer associates each YOLO label with its image, draws the bounding
boxes, shows image/annotation counts and available provenance, and supports
filename, positive/negative, and object-class filters without uploading files.
V3 filenames can be filtered by
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

- [`docs/dataset-v2/README.md`](docs/dataset-v2/README.md) and
  [`docs/dataset-v2/CREATION.md`](docs/dataset-v2/CREATION.md): public v2
  composition and construction;
- [`docs/dataset-v3/README.md`](docs/dataset-v3/README.md) and
  [`docs/dataset-v3/CREATION.md`](docs/dataset-v3/CREATION.md): canonical v3
  composition, split methodology, and reproducibility;
- [`../docs/status/dataset-v3-split-2026-09-02.md`](../docs/status/dataset-v3-split-2026-09-02.md):
  measured split report, evidence, and limitations;
- [`docs/dataset-plan-en.md`](docs/dataset-plan-en.md): normative build contract;
- [`docs/dataset-plan-it.md`](docs/dataset-plan-it.md): concise Italian plan;
- [`docs/annotation-guide-it.md`](docs/annotation-guide-it.md): manual review;
- [`docs/next-steps-it.md`](docs/next-steps-it.md): completed state and boundary;
- `docs/status/` and `docs/history/`: preserved development history;
- [`../docs/status/training-v3-expansion-review-2026-09-01.md`](../docs/status/training-v3-expansion-review-2026-09-01.md):
  measured pre-expansion audit and quantitative admission policy.
