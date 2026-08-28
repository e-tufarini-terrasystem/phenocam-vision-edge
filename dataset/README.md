# Public mixed-dataset builder

This workstation-only pipeline implements the approved public dataset contract
in [`docs/dataset-plan-en.md`](docs/dataset-plan-en.md). It never reads or writes
internal imagery.
The versioned configuration fixes `operational_validation` to `pending` until a
separately approved second phase.

The recommended entry point is the resumable workflow wrapper. Run commands
from the repository root:

```sh
dataset/workflow.sh setup
dataset/workflow.sh status
dataset/workflow.sh preflight
dataset/workflow.sh test
```

After preflight passes, the next automated stage is:

```sh
dataset/workflow.sh prepare-openimages-review
dataset/workflow.sh prepare-openimages-supplement
```

This screens the 900-image Open Images selection, checkpoints progress every 25
images, resumes a compatible checkpoint after interruption, and regenerates the
review packet only when it contains no human decisions. It refuses to overwrite
review work. The resulting `baseline_*` columns are prioritization hints, never
ground truth.

Generated metadata, downloads, and review packets live under ignored
`dataset/work/` paths. Final materialized datasets live under ignored
`dataset/artifacts/` paths. Builder source, configuration, documentation, and
tests remain tracked inside this directory. A command completing successfully
means only that its own stage passed; the dataset is complete only after the
final acceptance audit.

## Workspace layout

- `builder/`: deterministic CLI and dataset-construction modules;
- `config/`: versioned dataset contract and invariants;
- `docs/`: normative specifications, progress reports, and historical plans;
- `tests/`: builder-specific automated verification;
- `work/`: ignored downloads, caches, manifests, and human-review packets;
- `artifacts/`: ignored final materializations ready for downstream training.

Run the builder tests separately from the edge-runtime suite:

```sh
dataset/.venv/bin/python -m unittest discover -s dataset/tests -v
```

Human decisions remain authoritative. In particular, model output may prioritize
review but cannot establish ground truth, approve an ambiguous class, or verify a
negative image.

See [`docs/next-steps-it.md`](docs/next-steps-it.md) for the current automation
boundary and the remaining staged plan.

## Local annotation

CVAT Community is pinned and managed separately from the builder:

```sh
dataset/cvat.sh setup
dataset/cvat.sh start
dataset/cvat.sh cli-setup
dataset/cvat.sh create-superuser
dataset/cvat.sh open
```

After creating a personal access token in the local CVAT UI, configure the CLI
without placing credentials in the repository and create the positive tasks:

```sh
dataset/cvat-tasks.sh profile
dataset/cvat-tasks.sh upload-openimages
dataset/cvat-tasks.sh upload-phenocam
dataset/cvat-tasks.sh upload-openimages-supplement
```

Positive export, audit, and import are combined in the `finish` command. The
final negative pages and their import are managed by `dataset/finalize.sh`.
The complete human procedure is documented in
[`docs/annotation-guide-it.md`](docs/annotation-guide-it.md).

## Staged workflow

The public build uses these gates in order:

1. `openimages-index`, `openimages-shortlist`, and `openimages-download` resolve
   metadata, licenses, an oversized candidate pool, and official mirrored bytes.
2. `phenocam-index`, `phenocam-fetch-sites`, `phenocam-plan`,
   `phenocam-download`, and `phenocam-sample` retrieve a bounded seasonal pool.
   NASA Earthdata credentials may be configured in the ignored `dataset/.env`
   as `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or outside the repository
   in a mode-`0600` `.netrc`. Copy `dataset/.env.example` when configuring a new
   workstation and keep the resulting `.env` at mode `0600`; credentials and
   signed URLs are never written to output.
3. `baseline-screen` uses the unchanged model at confidence `0.05` only to order
   human work.
4. `embeddings` and `duplicate-pairs` compute the pinned SSCD descriptors and
   produce the mandatory duplicate and 200-pair calibration queues.
5. `openimages-select` creates a provisional 850-positive/50-negative selection
   with class, difficulty, provenance-group, and diversity controls. Global
   class-instance floors remain a reported joint gate until the PhenoCam
   positives are annotated.
6. `phenocam-select` creates a conservative provisional 350-positive/750-negative
   selection with distinct duplicate groups, at least 100 sites, a 12-frame site
   ceiling, independent negative day groups, and seasonal coverage.
7. `sscd-calibration-packet`, `phenocam-review-packet`, and
   `openimages-review-packet` create the local HTML, COCO, and CSV human-review
   artifacts. Human decisions must be imported before final materialization.
8. After the completed PhenoCam audit, `openimages-supplement-select` enforces
   the unchanged class floors with 379 additional positives. Baseline screening
   sends only conflicts plus a deterministic 10% box sample to the supplemental
   CVAT task.
9. `dataset/finalize.sh prepare` locks every reviewed positive, performs the
   smallest deterministic floor-repair swap, reuses 335 CVAT-confirmed empty
   PhenoCam frames, and produces reduced blind negative-review queues.

If a second PhenoCam reviewer is unavailable, the dataset owner may explicitly
run `dataset/finalize.sh accept-single-review` after resolving every rejected
negative. This weaker protocol accepts the completed first pass and writes
`negative-reviews-audit.json` with `single_reviewer_waiver`; it must not be
reported as independent negative verification.

After either review protocol has produced `negative-reviews.csv`, run
`dataset/finalize.sh build`. The command assembles the fixed 2,000-frame
composition, reruns exact and SSCD duplicate checks, validates every source and
box, and atomically writes the YOLO artifact, manifests, licenses, statistics,
checksums, and acceptance receipt under `dataset/artifacts/mixed-dataset/`.

The standard final contract requires PhenoCam annotation, independent negative
verification, duplicate review, and SSCD calibration. An owner-approved
single-review build is materialized with a visible waiver status instead of
being reported as a full independent-verification pass. Provisional command
output is never a completed dataset.

Internal imagery is intentionally absent from every command in this package.
Its future validation/test workflow remains a separately approved iteration,
with current status `operational_validation: pending`.
