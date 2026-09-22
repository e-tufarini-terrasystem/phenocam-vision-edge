# Phenocam Vision training data

Run commands from the repository root after the
[runtime source setup](../docs/cli.md#manual-setup-from-source). Restoring the
catalog uses the runtime dependencies only; acquisition and CVAT use their
separate workstation environment.

## Current canonical dataset

Use `dataset/data/`. It contains the reviewed former v6 training/validation
artifact and the two historical test splits, with source identities and human
labels preserved. There is no dependency on materializing older dataset versions.

| Split | Images | Purpose |
|---|---:|---|
| train | 2,020 | Training |
| val | 206 | Model and threshold selection |
| pklot_holdout | 6 | Historical parking benchmark |
| test_id | 200 | Historical public benchmark |
| test_ood | 240 | Historical operational benchmark |

Total: 2,672 images and 13,467 annotations. Images include private operational
data. Labels and metadata are tracked; image bytes, source pools, CVAT credentials
and generated workstation YAML stay outside Git. Do not treat this catalog as an
entirely public image distribution.

`config/dataset.json` fixes the manifest and checksum-inventory digests, counts
and 80-class mapping. The inventory pins human labels as well as image bytes;
changing labels and recomputing their checksums requires an explicit catalog
configuration revision.
`data/metadata/source-images.csv` is the authoritative catalog;
`data/labels/` holds each reviewed label file once. `provenance.json` maps every
migrated image and label to its original path and unchanged SHA-256.
Source `decoded_sha256` remains historical provenance; `compiled_decoded_sha256`
identifies the actual compiled image pixels, which may differ after JPEG encoding.

Ultralytics creates local `labels/<split>.cache` files even with image caching
disabled. Verification permits these regular files only for configured splits;
it still rejects cache symlinks, unexpected files, and changed images or labels.
Caches are ignored by Git and are not copied into rebuilt artifacts.

## Image access and reconstruction

The approved local pool stores each compiled JPEG as `<sha256>.jpg`, including
frozen derived crops. Preserve this pool or restore it from an authorized backup.
The acquisition tools can download public originals and prepare review tasks;
they cannot recreate human decisions or provide private images on a fresh clone.

The approved pool is hosted in the private Hugging Face dataset
[`etufarini-terrasystem/phenocam`](https://huggingface.co/datasets/etufarini-terrasystem/phenocam).
Downloading requires a work account with read access. On a fresh workstation,
authenticate with `hf auth login`, then download the pool using the separately
installed Hugging Face CLI:

```sh
hf download etufarini-terrasystem/phenocam --repo-type dataset \
  --revision main --include 'images/*.jpg' \
  --local-dir dataset/workspace/sources/approved
```

For a frozen remote snapshot, replace `main` with its full Hugging Face commit
SHA. In either case, the artifact commands below validate the downloaded bytes
against the catalog pinned in this repository.

From the repository root:

```sh
# Fresh clone: restore all image bytes alongside the tracked labels.
.venv/bin/python -m dataset.builder.artifact hydrate \
  --sources dataset/workspace/sources/approved/images

# Existing complete dataset: verify bytes, labels, inventory and split isolation.
.venv/bin/python -m dataset.builder.artifact verify

# Independent rebuild at a new destination.
.venv/bin/python -m dataset.builder.artifact build \
  --sources dataset/workspace/sources/approved/images --destination output/dataset-check

# CI or a clone without private images: verify the tracked catalog only.
.venv/bin/python -m dataset.builder.artifact verify --metadata-only
```

Construction refuses existing outputs and publishes only a fully verified
artifact. Image and label copies do not share writable hard links with sources.
Training/validation/test membership, parent image identity and site-day groups
remain separate. New annotations require an explicit catalog revision and audit;
there is no automatic admission of downloaded or model-annotated images.

## Acquisition and annotation

The maintained acquisition tools have a separate environment:

```sh
dataset/commands/dataset-builder.sh setup
```

Use the [annotation guide](docs/annotation-guide-it.md) for CVAT, the current
five-target policy and the older six-class campaigns. The original public
negatives retain `single_reviewer_waiver`; they were not independently verified.
New review decisions do not modify the frozen catalog without an explicit revision.

Source-specific download commands remain under `python -m dataset.builder`.
Operational mining uses `config/mining.json` and the configured `specialized`
model. Paired diagnostic commands use `--baseline-index` and `--candidate-index`;
these are explicit prediction inputs, not dependencies on older shipped models.
Existing CVAT task names and local workspace paths preserve annotation history.

Mining screening refuses `sealed_test` rows before reading images unless the
configuration declares `test_status: opened` and `--split sealed_test` is
explicitly selected. For development, pass `--split operational_dev` and/or
`--split operational_mining`. In both ONNX and teacher screening, `--resume`
reuses completed records and retries failed records after checking their input
identities. Operational CVAT imports reject non-integer image/category IDs and
references. SSCD calibration rejects a pool with fewer distinct image pairs
than the configured minimum before generating review outputs.

## Dataset viewer

Open [viewer.html](viewer.html), select `dataset/data/`, then choose a split. It stays
local and does not upload images. Training instructions are in
[the development guide](../docs/development.md#current-workflow).


## Historical acquisition and annotation context

The following records earlier datasets, not the current `dataset/data` contract.
The English/Italian dataset plans describe iteration 1; versioned dataset guides
and status reports preserve the decisions and waivers of their respective cycles.

<details>
<summary>Public v2/v3 acquisition, annotation and retired build commands (da14b7d)</summary>

The following describes the pre-consolidation pipeline at `da14b7d`. Intermediate
artifacts, model versions and historical commands are preserved as context;
use the canonical commands above for current builds. The old partition command
and `build-v3` materializer have been retired. The unversioned public-source
review/finalization tools remain available for acquisition work.


This directory owns both the completed public training dataset and the local,
reproducible construction pipeline. The distributable `training-dataset/` never
contains internal operational imagery. The private `dataset-v3/` artifact
partitions public data into train, validation, and TEST-ID, and reserves
reviewed operational imagery as TEST-OOD.
Unreviewed V3 mining data and local paths remain under the ignored `workspace/`
boundary.

The historical public training artifact was `training-dataset/` (local, ignored).
Its versioned composition and limitations are documented in
[`docs/dataset-v2/README.md`](docs/dataset-v2/README.md), which is copied to
`training-dataset/README.md` during materialization.

### Historical dataset status

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

The local `dataset-v5/` experiment artifact preserves v4, adds the completed
27-image PKLot review, and aligns selected training crops with the runtime
geometry. Its measured composition, leakage boundary, and training protocol are
recorded in
[`../docs/status/training-v5-design-2026-09-03.md`](../docs/status/training-v5-design-2026-09-03.md).

The waiver must remain visible in reports and training records. This dataset is
not evidence of production accuracy on internal camera imagery.

### Directory layout

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

### Builder commands

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
`included/reserved/rejected` selection described in `../docs/development.md` and
must be requested explicitly with `--include-public-expansion`; they do not
silently alter the canonical 2,240-image v3 dataset.

Run the tests directly with:

```sh
dataset/.venv/bin/python -m unittest discover -s dataset/tests -v
```

### Historical viewer

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

### Annotation commands

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

The first parking-lot source evaluation is recorded in
[`../docs/status/parking-lot-source-evaluation-2026-09-03.md`](../docs/status/parking-lot-source-evaluation-2026-09-03.md).
Only the completed official-PKLot task 17 is admitted to `dataset-v5`; the open
task 13 and its automatic suggestions remain outside every training artifact.

### Data sources and security boundary

NASA Earthdata credentials may be configured in ignored `dataset/.env` as
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or in a mode-`0600` `.netrc`.
Copy `.env.example` for a new workstation and keep `.env` at mode `0600`.
Credentials, signed URLs, local paths, and stack traces are not written to the
training artifact.

External images, archives, metadata, CVAT exports, and review CSVs are treated
as untrusted input. The builder validates size limits, archive membership,
image decoding, dimensions, checksums, required columns, source identity, unique
integer COCO image IDs, box geometry, licenses, and review completeness before
accepting data.

### Documentation

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

</details>
