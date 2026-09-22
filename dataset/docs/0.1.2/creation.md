# Creation of the public dataset (0.1.2)

> Historical record for this iteration; use the [current guide](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md).
> Commands refer to the original workflow; ignored artifacts require local evidence.

## Data origin and acquisition

The public dataset combines Open Images V7 and PhenoCam Network v3. Open Images
supplies diverse people and vehicle boxes; PhenoCam supplies deployment-like
fixed-camera scenes and seasonal negative imagery. Only images with resolvable
provenance, attribution, and an accepted license enter the dataset.

The builder downloads source metadata before images, validates every external
field, and records source URL, landing page, source ID, site/camera, timestamp,
license, attribution, and checksums. PhenoCam archives are obtained through
NASA Earthdata. Credentials stay in ignored local configuration and never enter
an artifact or report.

## Selection and annotation

Open Images candidates are selected deterministically under source, class,
difficulty, and provenance-group quotas. Eligible source boxes are mapped to
the six runtime classes. A supplemental selection repairs measured class floors.

PhenoCam candidates are distributed across sites, cameras, seasons, and time.
Baseline YOLO screening at low confidence only orders the review queue. Human
review in CVAT establishes positives and complete boxes. Blind local packets
establish negatives. The completed composition contains 15 positive and 706
negative PhenoCam frames.

The annotation pipeline preserves both the original fine-grained annotation
and the compiled YOLO box. It rejects unknown classes, incomplete reviews,
invalid geometry, empty positive annotations, and unverified negatives.

## Cleaning and deduplication

Before acceptance, the pipeline excludes unreadable or undersized images,
unsafe archive members, missing provenance, invalid licenses, malformed rows,
invalid boxes, unresolved review decisions, and source-identity conflicts.

Deduplication uses source-file SHA-256, EXIF-normalized decoded-pixel SHA-256,
64-bit perceptual hashes, SSCD embeddings, and metadata grouping. Candidate
near-duplicates receive review; accepted duplicate components retain one
indivisible `group_id`. Rejected candidates and reasons remain in generated
metadata instead of disappearing silently.

## Pipeline

```text
source metadata and archives
→ strict download and image validation
→ deterministic candidate selection
→ model-assisted review prioritization
→ human annotation and negative review
→ annotation normalization
→ deduplication and quota reconciliation
→ acceptance audit
→ immutable materialization and checksums
```

The exact approved specification is preserved in
[`../plan-en.md`](../plan-en.md), with an Italian summary in
[`../plan-it.md`](../plan-it.md). Manual annotation guidance is
in [`../annotation-guide-it.md`](../annotation-guide-it.md).

## Reproducibility

From the repository root, create the isolated environment and run the checks:

```sh
dataset/commands/dataset-builder.sh setup
dataset/commands/dataset-builder.sh status
dataset/commands/dataset-builder.sh preflight
dataset/commands/dataset-builder.sh test
```

The full workflow uses the staged builder and review commands documented in the
normative plan. Once all reviewed inputs are present, materialize the artifact:

```sh
dataset/commands/dataset-finalization.sh build
```

The command refuses to overwrite `dataset/training-dataset/`. A successful
artifact can be checked without network access:

```sh
cd dataset/training-dataset
shasum -a 256 -c metadata/checksums.sha256
```

Expected invariants are 2,000 manifest/image rows, 1,244 label files, 4,677
boxes, no unlisted files, and valid checksums. The automated builder suite runs
with:

```sh
dataset/.venv/bin/python -m unittest discover -s dataset/tests -v
```

Principal Python dependencies are pinned in `dataset/requirements.txt`; CVAT
CLI support is separately pinned in `dataset/requirements-cvat.txt`.

## Split decision and leakage boundary

0.1.2 deliberately materializes only `train`. Duplicate and provenance groups are
still recorded so later splitting can keep related items together. Internal
operational images are excluded entirely. This preserves the first experiment's
public-data boundary but makes 0.1.2 unsuitable for final validation by itself.

Dataset 0.1.3 performs the actual group-safe partition. It keeps Open Images
duplicate groups indivisible, keeps each public PhenoCam camera together, joins
PhenoCam pHash-near components, and reserves private operational cameras for
TEST-OOD.

## Design decisions and limitations

- COCO-compatible IDs are preserved instead of collapsing vehicles into one class.
- Negative images omit label files; the manifest makes that state explicit.
- Model output may prioritize review but cannot create ground truth.
- Source and compiled annotations are both retained for traceability.
- External input is strictly validated and path traversal is rejected.
- The single-reviewer waiver remains visible and cannot be described as
  independent negative verification.
- Operational validation is outside 0.1.2 and remained pending until 0.1.3.
