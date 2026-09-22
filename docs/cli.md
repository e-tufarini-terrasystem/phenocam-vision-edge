<!--
This document owns the operational instructions for using Phenocam Vision Edge,
including CLI commands, arguments, output behavior, metadata, timing, and status.
-->

# Software operation and CLI reference

Use this guide to operate Phenocam Vision Edge after installation. For source
setup and class configuration, see the [manual guide](manual.md).

## Usage

Request an annotated output:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --model models/yolo26n-phenocam.onnx
```

Request privacy output only, or both products from the same inference:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --privacy-output output/privacy.jpg \
  --model models/yolo26n-phenocam.onnx

.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --privacy-output output/privacy.jpg \
  --meta input/example.meta \
  --model models/yolo26n-phenocam.onnx
```

Request conditional input and metadata deletion, optionally specifying image
outputs (deletion takes precedence), or use deletion-only mode:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --privacy-output output/privacy.jpg \
  --meta input/example.meta \
  --model models/yolo26n-phenocam.onnx \
  --delete-input-on-detection

.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --model models/yolo26n-phenocam.onnx \
  --delete-input-on-detection
```

To replace the input with the privacy image, use the same path for both:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --privacy-output input/example.jpg \
  --model models/yolo26n-phenocam.onnx
```

The same applies to `--annotated-output`. Replacement occurs only when at least
one enabled final detection remains and `--delete-input-on-detection` is absent.
With that flag, deletion takes precedence even when an output names the input.

To update only detection metadata, request `--meta` without image outputs:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --model models/yolo26n-phenocam.onnx \
  --meta input/example.meta
```

The `.meta` file must already exist. This mode performs the complete inference,
updates detection counts, and leaves images untouched. Both image paths in
metadata are empty, including when enabled objects are detected.

At least one image output, `--meta`, or `--delete-input-on-detection` is required.
Supplying only `--input` and `--model` fails with
`error: at least one output path, metadata path, or input deletion is required`.
Deletion-only mode also performs the complete inference.
`--output` has been removed without an alias. Output directories must already exist.

The default uses all four Pi 3 CPU cores. To reduce CPU load or temperature at
the cost of latency, set the thread count to a value from 1 to 4:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/example.jpg \
  --model models/yolo26n-phenocam.onnx
```

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Existing local image file. |
| `--annotated-output` | Optional annotated image path; parent must exist; may replace the input. |
| `--privacy-output` | Optional privacy image path; parent must exist; may replace the input. |
| `--model` | Existing local file with a case-insensitive `.onnx` extension. |
| `--meta` | Optional existing regular non-symlink `.meta` file, distinct from input, model, and outputs; usable without image outputs or deletion. |
| `--delete-input-on-detection` | On an enabled final detection, delete the input, then the supplied `--meta`, instead of writing products; symbolic-link inputs are rejected. |

Images are written only when at least one enabled final detection remains
after confidence filtering and global suppression and deletion is disabled.
With no such detection,
the input and any existing outputs remain byte-for-byte unchanged, missing
outputs are not created, and the command succeeds. Existing outputs from an
earlier execution are not removed automatically.

Requested outputs must be distinct from each other after path, symbolic-link,
and existing hard-link resolution. One output may use the input path, including
equivalent spellings such as `./image.jpg`. Symbolic-link inputs or outputs
cannot be used for input replacement, and distinct hard-link aliases of the
input remain invalid. An output may name the input with deletion enabled:
detection deletes the input, while no detection leaves it untouched. All
explicit paths are still validated, including output directories.

Each image is encoded to a temporary file in its destination directory,
verified and fully decoded, then atomically replaces its destination. Encoding,
verification, decoding, or replacement failures preserve that destination's
previous contents. Full decoding also rejects truncated JPEG pixel data that
the format verification alone may accept.
Existing permission bits are retained; new image files have private permissions
(`0600`). Replacement requires write permission on the destination directory
and changes the file identity: other hard links retain the old contents.
Ordinary output symlinks to files other than the input continue to target those
files. Images use the existing RGB/EXIF normalization and encoding behavior;
replacement does not preserve the original file's embedded metadata.

Both products start from the original image loaded in memory. Annotated saving
finishes first; if privacy saving then fails, the completed annotated file
remains, including when it replaced the input. Atomicity is per image, not
across both images and metadata. The command is headless and does not open a
graphical window.

Conditional deletion is opt-in and takes precedence over all product writes.
After inference and global suppression, an enabled final detection triggers
input deletion followed by deletion of the file explicitly supplied via
`--meta`, when present. No image is generated and metadata is not updated first.
Existing output images remain untouched. Without `--meta`, no metadata file is
searched for or deleted, including a file with the same stem as the input.

If input deletion fails, metadata remains untouched; the command returns status
`1` and prints `error: input image could not be deleted`. If metadata deletion
fails, the input has already been removed; the command returns status `1` and
prints `error: metadata file could not be deleted`. These two deletions are not
atomic and there is no rollback of the input deletion. Neither failure writes
image outputs or updates metadata.

When no enabled final detection remains, the input and existing output images
are retained, no new images are generated, and any supplied metadata is updated
with `detected=false`, zero count, and empty image paths. Without the deletion
flag, the normal image/metadata behavior applies.

Immediately before unlinking, the command verifies that the path is still a
regular file with the device/inode identity captured during argument validation.
The same check applies to supplied metadata before its deletion. This does not
claim atomic protection against a hostile replacement between verification and
unlink. Hard links are allowed, and only the exact directory entry passed to
`--input` or `--meta` is removed.

## Detection filtering

Inference uses the full image and fifteen overlapping crops, with a uniform
confidence threshold of `0.47`. Duplicate boxes compete within each class;
`car`, `bus`, and `truck` also compete with one another. Suppression requires
IoU of at least `0.50`, or at least `0.50` coverage of the smaller box when the
boxes come from different views. Smaller-box coverage alone does not suppress
two detections from the same view, preserving adjacent partially occluded
objects. These thresholds are fixed in the runtime, not CLI options.

This rule can recover objects that were predicted but suppressed. Objects with
confidence below `0.47`, or with no model prediction, can still be missed behind
foliage or railings. Calibration results and limitations are recorded in the
[development guide](development.md#occlusion-calibration).

## Detection metadata

When conditional deletion is triggered, the supplied `.meta` file is deleted
after the input and is not updated. Otherwise, when `--meta` is supplied, the
existing file is updated after every generated
image has been written and verified, or directly after inference when no
images are requested or no enabled final detection remains. All previous exact lowercase
`[detection]` sections are removed and one current section is appended
atomically; other bytes are retained and no history is created. Counts include
only enabled final detections after global suppression, in canonical COCO
order:

```text
[detection]
detected=true|false
software_name=phenocam-detection
software_version=0.2.3
model_id=yolo26n-phenocam
model_version=0.1.6
annotated_image=<CLI path of image produced this execution, or empty>
privacy_image=<CLI path of image produced this execution, or empty>
classes=<comma-separated detected classes or empty>
<class-key>_count=<positive integer, detected classes only>
total_count=<sum, or 0>
```

`software_version` comes from `phenocam.__version__`, defined once in
`phenocam/__init__.py`. Keep its declaration in the literal form
`__version__ = "MAJOR.MINOR.PATCH"` and update it before creating a release.
The packager checks it against the requested archive version using the selected
commit, without executing that commit's code. Historical commits without this
declaration remain packageable. `model_version` is independent of the software
version. Previously distributed archives are not changed by this correction.

With `--meta`, `model_id` and `model_version` come from the selected ONNX file's
sibling `.json` receipt (the same filename with its extension replaced).
The runtime validates the identity and checks `onnx_sha256` against the selected
model before inference, image writes, or conditional deletion. It never uses
the receipt's path fields to select a model. The receipt is limited to 64 KiB;
`model_id` is 1–64 ASCII letters, digits, dots, underscores or hyphens, beginning
with a letter or digit. `model_version` uses `MAJOR.MINOR.PATCH`, without leading
zeros, within 32 characters. The digest is 64 lowercase hexadecimal characters.
An absent receipt yields `model_id=unknown` and `model_version=0.1.0`.
Without a receipt, `0.1.0` is a conventional initial value, not a verified
revision; no model hash is checked in this case.
A present but invalid or mismatched receipt fails with `error: inference failed`
and status `1`. Without `--meta`, the receipt is not read.

The bundled model's historical revision v6 is version `0.1.6` in the project's
experimental `0.1.N` model series. Its stable filename remains
`yolo26n-phenocam.onnx`. This corrects identity only: the weights, ONNX bytes,
historical provenance and experimental acceptance status are unchanged.

The optional base model `models/yolo26n.onnx` uses `models/yolo26n.json`,
which declares `model_id=yolo26n` and `model_version=0.1.0`. This is the
project's initial version of that base artifact, not an upstream Ultralytics
release number. Selecting another model with `--model` selects its sibling
receipt automatically. Keep the ONNX and its receipt together when copying
or renaming them.

Each `(model_id, model_version)` pair must identify a single ONNX SHA-256.
Different ONNX bytes, including a new export, require a new model version
and matching hash. Keep filenames stable and software versions independent.
The Ultralytics `version` metadata describes the exporter, not the weights.
The SHA-256 check verifies the pairing of receipt and model; it does not
authenticate their provenance.

With no enabled final detection, metadata contains `detected=false`,
`total_count=0`, and empty `annotated_image` and `privacy_image` values, even
when those paths were requested or older output files exist.

Output paths retain their CLI spelling. A metadata failure keeps completed
images and the previous metadata file available, returns status `1`, and
prints `error: metadata file could not be updated`. If a completed image
replaced the input, the original is not restored. Invalid explicit metadata
paths fail before inference with `error: metadata file does not exist or is not
a file`, `error: metadata file must use the .meta extension`,
`error: output path cannot be stored in metadata`, or
`error: metadata path must differ from input, model, and output paths`.

## Timing and exit statuses

A successful command prints `Execution time: N.NNN s`, the sum of the sixteen
ONNX `session.run()` intervals. It excludes startup, session creation, image
preparation, rendering, and writing. Raspberry Pi multi-view timing is not
verified; the 15-second complete-command limit remains an unverified target.

| Status | Meaning |
|---|---|
| `0` | Inference and applicable image writes, metadata update, and conditional deletion succeeded; no enabled detection is also success. |
| `1` | Validation, inference, output writing, metadata update, or conditional deletion failed. |
| `2` | Command-line syntax is invalid. |
