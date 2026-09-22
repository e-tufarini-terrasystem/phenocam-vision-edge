<!-- Own installation, configuration and the single-image operating contract. -->

# Installation and CLI reference

For the versioned Raspberry Pi installer, start with the [README](../README.md).
Commands below run from the installation or source repository root.

## Release and source differences

This guide describes the current source. The frozen `v0.2.3` release uses the
same CLI and conditional output behavior, but its missing-receipt fallback differs:

| Selected model receipt | Published `v0.2.3` | Current source |
|---|---|---|
| Absent, with `--meta` | `model_id=unknown`, `model_version=unknown` | `model_id=unknown`, `model_version=0.1.0` |
| Valid bundled receipt | `yolo26n-phenocam`, `0.1.6` | `yolo26n-phenocam`, `0.1.6` |
| Invalid or mismatched, with `--meta` | Fail before inference or writes/deletion | Same |

Without a receipt, no model hash is checked; `0.1.0` is a conventional initial
value, not a verified revision. Published archives are unchanged. The optional
base ONNX and its receipt are available in the current source repository, not
in the specialized-model runtime archive.

## Manual setup from source

Clone the repository and enter it before running setup commands:

```sh
git clone https://github.com/e-tufarini-terrasystem/phenocam-vision-edge.git
cd phenocam-vision-edge
```

On Raspberry Pi OS 64-bit, provision Python 3.13 and its `venv` support first.
Check `python3 --version`; the OS default must be 3.13 for the commands below.
Installing `python3-venv` alone does not upgrade an older Python interpreter.

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt
mkdir -p input output
```

On macOS with Homebrew and Apple Silicon:

```sh
brew install python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
mkdir -p input output
```

Activation is optional: the commands use the environment's interpreter directly.
Keep `models/yolo26n-phenocam.onnx` and its sibling `.json` together.
Ultralytics, PyTorch, OpenCV and the training/export tools are unnecessary on the
Pi. A minimal deployment can omit PT checkpoints, `training/`, `dataset/` and
workstation scripts/requirements; the runtime archive contains deployment files.

Historical storage measurements on an 8 GB microSD (about 6.9 GiB usable) were
154 MiB for the environment and 187 MiB for the working copy, using Python
3.13.5 on a four-core Cortex-A53 Pi. These are not measurements of the current
model deployment. `--no-cache-dir` avoids retaining pip downloads.

## Usage

Copy an image into `input/`, then request either image output or both:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg --model models/yolo26n-phenocam.onnx \
  --annotated-output output/example_annotated.jpg \
  --privacy-output output/example_privacy.jpg
```

For metadata only, supply an existing `.meta` file:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg --model models/yolo26n-phenocam.onnx \
  --meta input/example.meta
```

To delete the source when a target is detected:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg --model models/yolo26n-phenocam.onnx \
  --delete-input-on-detection
```

Adding `--meta input/example.meta` to that command also deletes the supplied
metadata when detection triggers deletion. No same-stem metadata is discovered
automatically by the direct CLI.

To replace the input with a privacy image, set `--privacy-output` to the input
path. The same works for `--annotated-output`. Replacement requires a detection;
with the deletion flag, deletion takes precedence even for an in-place output.

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Required existing local image file. |
| `--model` | Required existing local file with a case-insensitive `.onnx` extension. |
| `--annotated-output` | Annotated image path; parent must exist; may replace the input. |
| `--privacy-output` | Privacy image path; parent must exist; may replace the input. |
| `--meta` | Existing regular non-symlink `.meta` file, distinct from input, model and outputs. |
| `--delete-input-on-detection` | Delete input, then supplied metadata, instead of writing products; rejects symlink inputs. |

At least one output, `--meta`, or deletion is required. All modes perform full
inference. The old `--output` option has no alias. Use `--help` for syntax.

Outputs must be distinct after path, symlink and existing hard-link resolution.
One output may name the input directly, including `./image.jpg`; symlink
replacement and distinct hard-link aliases of the input are rejected. Explicit
paths are validated even when deletion will take precedence.

## Configuration and filtering

Edit only the existing booleans in `phenocam/classes/configuration.py`; keep at
least one enabled. The defaults are `person`, `car`, `motorcycle`, `bus`, `truck`.
A bicycle alone is not a privacy target; a cyclist is covered by the person box.
Filtering controls image products, metadata counts and deletion. All 80 model
classes still participate in inference and suppression, so disabling classes
does not reduce compute or memory.

Inference uses sixteen views and confidence `0.47`. Boxes compete within their
class; `car`, `bus` and `truck` also compete across labels. Suppression requires
IoU ≥ `0.50`, or smaller-box coverage ≥ `0.50` between different views.
Coverage alone does not suppress two boxes from the same view. These are runtime
constants, not CLI options. Objects below threshold or never predicted can be
missed; see the [historical calibration](development.md#occlusion-calibration).

Privacy expands each box by 10% on every side, clips it to the image and applies
Gaussian blur with radius `max(8 px, 10% of the region's shorter side)`.
It adds no labels or outlines.

The runtime defaults to at most four available CPU cores. Set 1–4 threads to
trade latency for CPU load; invalid environment values fall back to the default:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input input/example.jpg --model models/yolo26n-phenocam.onnx \
  --annotated-output output/example.jpg
```

## Outputs and conditional deletion

| Final enabled detection | Deletion requested | Result |
|---|---|---|
| No | Either | Preserve input and existing outputs; create no images; update supplied metadata with zero detections and empty image paths. |
| Yes | No | Write requested images, then update supplied metadata. Metadata-only mode writes no images. |
| Yes | Yes | Delete input, then supplied metadata; do not write images or update metadata. Existing outputs remain untouched. |

Each image is encoded to a temporary file beside its destination, verified and
fully decoded, then atomically replaces the destination. Failed encoding,
decoding or replacement preserves that destination. Existing permission bits
are retained; new files have mode `0600`. Replacement requires directory write
permission and changes file identity: other hard links retain the old contents.
Ordinary output symlinks to files other than the input still target those files.
EXIF orientation and RGB normalization apply; original embedded image metadata
is not preserved.

Both products use the original image in memory. Annotated saving finishes before
privacy saving. Atomicity is per file: a later image or metadata failure leaves
completed writes in place, including an input already replaced. The command
is headless. Stale outputs are never removed automatically after no detection.

Deletion is ordered and has no rollback. If input deletion fails, metadata is
untouched. If metadata deletion fails, the input is already gone. Immediately
before each unlink, the runtime checks that the entry is still a regular file
with its validated device/inode identity; this is not atomic protection against
hostile replacement between check and unlink. Only the supplied directory entry
is removed; other hard links remain.

## Detection metadata

Except when conditional deletion triggers, `--meta` updates the existing UTF-8
file after all requested image writes, or directly after inference when no images
are produced. Exact lowercase `[detection]` sections are replaced atomically by
one current section; other bytes are retained and no history is created.
Counts include enabled final detections in canonical COCO order:

```text
[detection]
detected=true|false
software_name=phenocam-detection
software_version=0.2.3
model_id=yolo26n-phenocam
model_version=0.1.6
annotated_image=<produced image path, or empty>
privacy_image=<produced image path, or empty>
classes=<comma-separated detected classes, or empty>
<class-key>_count=<positive integer, detected classes only>
total_count=<sum, or 0>
```

`software_version` comes from `phenocam.__version__`; it is independent of the
model version. Image paths describe products of this execution, not older files.
They retain relative or absolute CLI form, subject to Python `Path` normalization
(such as removing a leading `./`). Metadata-only mode always records empty paths.
A metadata failure preserves its previous contents and completed images.

With `--meta`, the runtime reads the selected model's sibling `.json` (same stem),
validates its identity and checks `onnx_sha256` against the ONNX before inference,
writes or deletion. It never selects model paths from receipt fields. Without
`--meta`, the receipt is not read. A receipt is limited to 64 KiB and requires:

| Field | Format |
|---|---|
| `model_id` | 1–64 ASCII letters, digits, dots, underscores or hyphens; starts with a letter or digit. |
| `model_version` | `MAJOR.MINOR.PATCH`, no leading zeros, at most 32 characters. |
| `onnx_sha256` | 64 lowercase hexadecimal characters matching the model. |

The specialized model is `yolo26n-phenocam`, version `0.1.6`, and remains
experimental. The optional base uses `models/yolo26n.json` with
identity `yolo26n`, version `0.1.0`. These are project artifact versions, not
Ultralytics exporter versions. Copy or rename the ONNX and receipt together.
The hash binds the pair but does not authenticate its provenance.
For missing receipts, see [release and source differences](#release-and-source-differences).

## Batch usage

```sh
./scripts/batch.sh
```

The script processes supported images directly inside `input/`, one process per
image, using `models/yolo26n-phenocam.onnx`. It creates `output/` and requests
`<stem>_annotated.<ext>`, preserving extension spelling. A regular non-symlink
`input/<stem>.meta` is passed when present; missing metadata is not created.
Privacy and deletion remain direct CLI operations; the batch never enables deletion.
The same conditional output and metadata rules apply to each image.

The settings at the top of the script select paths, confidence (default `0.47`)
and threads. `YOLO_NUM_THREADS` overrides threads; unlike the direct runtime,
the script rejects values outside 1–4. Confidence customization affects only
that batch's child processes. Compare the same inputs in separate output
directories when trying thresholds. Failures do not stop subsequent images;
exit status is `1` if any image fails or no supported image is found.

## Timing and errors

`Execution time: N.NNN s` sums the sixteen ONNX `session.run()` intervals.
It excludes startup, session creation, preprocessing, rendering and writes.
The 15-second complete-command target and current Pi multi-view timing remain
unverified; this timer alone cannot establish either.

| Status | Meaning |
|---|---|
| `0` | Requested actions succeeded; no enabled detection is also success. |
| `1` | Validation, inference, output, metadata or deletion failed. |
| `2` | Invalid command-line syntax. |

Runtime failures report `error: inference failed`, `error: output image could
not be written`, `error: metadata file could not be updated`, `error: input
image could not be deleted`, or `error: metadata file could not be deleted`.
Invalid explicit paths fail before inference, including nonexistent parents,
non-regular metadata, wrong extensions and colliding paths.

The versioned installation chain stops on download, checksum or extraction
failure before proceeding. Installer failure leaves the extracted package and
any partial `.venv` for inspection. Diagnostics identify architecture (`aarch64`),
Python (`3.13`), missing `venv` support, an existing environment, missing runtime
requirements/model, or failed environment creation, dependency installation or
input/output directory creation. The installer does not repair an existing `.venv`.
