# Creation of dataset v3

> Historical record for this iteration; use the [current guide](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md).
> Commands refer to the original workflow; ignored artifacts require local evidence.

## Origin and acquisition

Dataset v3 starts from the verified 2,000-image public v2 artifact and adds 240
reviewed private operational images for out-of-domain evaluation. The public
base combines Open Images V7 and PhenoCam Network v3; its acquisition,
licensing, review, and deduplication are described in
[`../dataset-v2/CREATION.md`](../dataset-v2/CREATION.md).

Public PhenoCam frames were acquired through NASA Earthdata from fixed cameras,
then selected across sites, cameras, seasons, and time. The public set covers
157 cameras/sites from 2001 through 2023. Private operational candidates were
inventoried from a read-only root and kept under the ignored `workspace/`
security boundary.

## PhenoCam and operational selection

Candidate images are validated before decoding for safe path, type, and pixel
limits. Inventory records stable identity, site, timestamp, group, source and
decoded hashes, and perceptual hash. Deterministic site-day groups prevent
adjacent operational frames from being treated as independent samples.

Baseline and v2 YOLO models screen candidates only to prioritize review and
identify representative or informative frames. Suggestions are never accepted
as labels without human review. The selected operational cohorts contain 120
representative development frames and 120 informative mining frames from the
two available private cameras.

An optional, separately named `dataset-v3-expanded/` artifact can include 18
additional human-reviewed public PhenoCam positives. It is not part of the
canonical 2,240-image v3 composition and is never included implicitly.

## Annotation

CVAT bundles preserve stable image identity and may contain model suggestions.
Annotators correct classes and boxes, add missed objects, and complete the task.
Import validates the task ID, completion state, source hashes, image dimensions,
class mapping, geometry, annotator, reviewer, and manifest membership.

The compiler stores normalized YOLO boxes and a JSONL audit of source and
compiled annotations. Private images are marked `human_cvat`; their filenames
retain site, timestamp, and a short identity hash. Intentional negative images
remain label-free.

## Cleaning and integrity controls

The pipeline excludes unsafe roots and links, unsupported files, oversized or
corrupt images, identity/hash mismatches, incomplete reviews, unknown classes,
invalid boxes, duplicate artifact names, and manifest paths escaping the data
root. External metadata, archives, CVAT exports, CSV, JSON, and image files are
treated as untrusted input.

Exact source, decoded-pixel, and compiled hashes must be unique. For splitting,
Open Images uses reviewed duplicate `group_id`; PhenoCam uses camera/site plus
connected pHash components at Hamming distance at most 6. pHash groups images
but does not delete them. All 2,240 source images remain in the final artifact.

## Construction pipeline

```text
public v2 artifact + reviewed private images
→ strict manifest and checksum validation
→ canonical 2,240-image unsplit v3 source artifact
→ source/label inventory and measured statistics
→ camera, duplicate, and pHash connected components
→ deterministic constrained allocation (seed 42)
→ hard-linked Train/Val/TEST-ID/TEST-OOD materialization
→ manifests, YAML files, reports, and checksums
→ independent full-artifact verification
```

The final partition is atomic: the builder writes a temporary sibling, verifies
it, and only then promotes it. It refuses to overwrite an existing destination.

## Split method

The 2,000 public images use exact 80/10/10 counts: 1,600 Train, 200 Validation,
and 200 TEST-ID. The 240 private images are sealed into TEST-OOD. Group
integrity has priority over class balance.

- Open Images: reviewed duplicate/provenance groups stay together.
- Public PhenoCam: whole cameras/sites and pHash≤6 components stay together.
- Private operational data: whole cameras/sites stay in TEST-OOD.
- Seed: `42`.
- Objective after grouping: minimize class, season, and luminance deviation
  while preserving exact image counts.

Augmentation happens only after partitioning. TEST-ID and TEST-OOD never guide
selection, thresholds, mining, checkpoint choice, or future additions.

## Reproducibility

From the repository root, prepare the environment and verify the builder:

```sh
dataset/commands/dataset-builder.sh setup
dataset/commands/dataset-builder.sh test
```

Starting from a clean destination and complete reviewed workspace, materialize
the canonical source and partition it:

```sh
dataset/commands/dataset-finalization.sh build-v3
cd dataset
.venv/bin/python -m builder.partition build dataset-v3-source dataset-v3
.venv/bin/python -m builder.partition verify dataset-v3
```

Recalculate an existing artifact's inventory and statistics without writing:

```sh
cd dataset
.venv/bin/python -m builder.partition analyze dataset-v3
```

Regenerate the split beside a valid artifact and verify it:

```sh
cd dataset
.venv/bin/python -m builder.partition build dataset-v3 dataset-v3-rebuilt
.venv/bin/python -m builder.partition verify dataset-v3-rebuilt
```

Build the optional reviewed public expansion explicitly:

```sh
dataset/.venv/bin/python -m dataset.builder.finalization build-v3 \
  --destination dataset/dataset-v3-expanded \
  --include-public-expansion
```

The canonical source build omits the optional expansion by default, so its
2,240-image contract remains accepted by the partition verifier.

## Validation gates

The final verifier requires:

- 1,600/200/200/240 images in Train/Val/TEST-ID/TEST-OOD;
- matching image/label paths and accurate annotation summaries;
- no empty or duplicate stable identity or source/decoded/compiled hash;
- no partition group or public PhenoCam camera crossing splits;
- no pHash≤6 pair crossing splits;
- private rows only in TEST-OOD;
- all six classes in each public split;
- portable YAML paths and complete checksum validation.

The repository test command covers the builder, finalization, mining, split,
viewer controls, and security boundaries. The viewer additionally supports a
manual local check of images, boxes, classes, filters, and navigation.

## Design decisions and known limits

The final artifact keeps `metadata/` because provenance and reproducibility are
dataset data, not incidental reports. Human-readable and machine-readable
validation outputs live under `reports/`. Generated workspace state, models,
downloads, CVAT tools, credentials, and caches remain ignored.

The public expansion is explicit because silently changing 2,240 to 2,258
breaks the approved split contract. It remains available for a later versioned
decision rather than being deleted. The two-camera TEST-OOD set measures the
available operational shift but cannot support broad claims about unseen sites.
