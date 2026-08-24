<!--
This document owns manual installation, runtime configuration, command usage,
output behavior, metadata, batch processing, and operator-facing diagnostics.
-->

# Operation and manual configuration

## Manual setup from source

On a 64-bit Raspberry Pi OS installation, create an isolated environment and
disable pip's download cache to save space on a small microSD card:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt
```

The Python 3.13 dependency set in `requirements/runtime.txt` was installed and
tested with Python 3.13.5 on a four-core Cortex-A53 Raspberry Pi. Python 3.11
uses its compatible NumPy pin.

### 8 GB microSD deployment

The tested 8 GB microSD exposes about 6.9 GiB. The deployment stays small by
installing only NumPy, ONNX Runtime, and Pillow with `--no-cache-dir`;
Ultralytics, PyTorch, OpenCV, and the export toolchain are not required on the
Pi. The virtual environment occupies about 154 MiB and the complete tested
working copy about 187 MiB.

A minimal source copy may omit `models/yolo26n.pt`, the export scripts, and
`requirements/export.txt`, but must retain `models/yolo26n.onnx`.

### macOS

Install Python 3.13 with Homebrew, then create the virtual environment from the
project directory. The runtime requirements also support macOS on Apple
Silicon.

```sh
brew install python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
```

The commands below work without activation. To use the environment's `python`,
run `source .venv/bin/activate`; run `deactivate` when finished.

## Usage

Request an annotated output:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --model models/yolo26n.onnx
```

Request privacy output only, or both products from the same inference:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --privacy-output output/privacy.jpg \
  --model models/yolo26n.onnx

.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --privacy-output output/privacy.jpg \
  --meta input/example.meta \
  --model models/yolo26n.onnx
```

Request products and conditional input deletion together, or deletion without
creating an image product:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --privacy-output output/privacy.jpg \
  --model models/yolo26n.onnx \
  --delete-input-on-detection

.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --model models/yolo26n.onnx \
  --delete-input-on-detection
```

At least one image output or `--delete-input-on-detection` is required;
metadata alone is not a final action. Deletion-only mode still performs the
complete inference. `--output` has been removed without an alias. Output
directories must already exist.

The default uses all four Pi 3 CPU cores. To reduce CPU load or temperature at
the cost of latency, set the thread count to a value from 1 to 4:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/example.jpg \
  --model models/yolo26n.onnx
```

## Class configuration

The fixed `phenocam/classes/configuration.py` file is loaded automatically.
Change only its existing `True` and `False` values and keep at least one class
enabled. The committed configuration enables `person`, `bicycle`, `car`,
`motorcycle`, `bus`, and `truck`.

Class filtering controls final annotations, privacy regions, and the conditional
deletion trigger. All model classes still participate in inference, merging,
and global suppression, so disabling classes does not reduce compute or memory.

Privacy uses rectangular regions only. Each selected box expands by 10% on
every side, clips to the image, and receives Gaussian blur with radius
`max(8 px, 10% of the region's shorter side)`. It adds no boxes, names, or
confidence text.

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Existing local image file. |
| `--annotated-output` | Optional annotated image path; parent must exist. |
| `--privacy-output` | Optional privacy image path; parent must exist. |
| `--model` | Existing local file with a case-insensitive `.onnx` extension. |
| `--meta` | Optional existing regular non-symlink `.meta` file, distinct from input, model, and outputs. |
| `--delete-input-on-detection` | Delete the validated input only after an enabled final detection and every requested product succeeds; symbolic-link inputs are rejected. |

Requested outputs must be distinct from the input and from each other after
path, symbolic-link, and existing hard-link resolution. Existing outputs are
overwritten. With both products, annotated saving finishes first; if privacy
saving then fails, the completed annotated file remains. Zero selected
detections still writes each requested normalized source image. The command is
headless and does not open a graphical window.

Conditional deletion is opt-in and is always the final transaction step:
inference completes first, followed by requested images, metadata, and finally
deletion. Any earlier failure retains the input. A deletion failure does not
roll back completed products; it returns status `1` and prints exactly
`error: input image could not be deleted`.

When no enabled final detection remains, the input is retained and the command
still succeeds.

Immediately before unlinking, the command verifies that the path is still a
regular file with the validated device/inode identity. This check does not
claim atomic protection against a hostile replacement between verification and
unlink. Hard links are allowed, and only the exact directory entry passed to
`--input` is removed.

## Detection metadata

When `--meta` is supplied, the existing file is updated only after every
requested image has been written and verified. All previous exact lowercase
`[detection]` sections are removed and one current section is appended
atomically; other bytes are retained and no history is created. Counts include
only enabled final detections after global suppression, in canonical COCO
order:

```text
[detection]
detected=true|false
software_name=phenocam-detection
software_version=0.1.0
model_id=yolo26n
model_version=0.1.0
annotated_image=<CLI path or empty>
privacy_image=<CLI path or empty>
classes=<comma-separated detected classes or empty>
<class-key>_count=<positive integer, detected classes only>
total_count=<sum, or 0>
```

Output paths retain their CLI spelling. A metadata failure keeps completed
images and the previous metadata file available, returns status `1`, and
prints `error: metadata file could not be updated`. Invalid explicit metadata
paths fail before inference with `error: metadata file does not exist or is not
a file`, `error: metadata file must use the .meta extension`,
`error: output path cannot be stored in metadata`, or
`error: metadata path must differ from input, model, and output paths`.

## Batch usage

Run inference on every supported image directly inside `input/`:

```sh
./scripts/batch.sh
```

The script creates `output/` when needed and writes
`<stem>_annotated.<ext>` and `<stem>_privacy.<ext>` in one process per input,
preserving extension spelling. If a regular non-symlink
`input/<stem>.meta` exists, it is passed through `--meta`; a missing match is
not created and does not fail that image.

Existing files are overwritten. Processing continues after individual
failures, but the script exits with status `1` if any image fails or none is
supported. The batch script never enables `--delete-input-on-detection`.

## Timing and exit statuses

A successful command prints `Execution time: N.NNN s`, measured as the sum of
the sixteen ONNX `session.run()` intervals. It excludes Python startup,
model/session creation, image decoding, view preparation, merging, rendering,
and image writing. Raspberry Pi multi-view timing is not verified; the
15-second complete-command limit remains an unverified acceptance target.

| Status | Meaning |
|---|---|
| `0` | Inference and every requested output, metadata update, and conditional deletion succeeded. |
| `1` | Validation, inference, output writing, metadata update, or conditional deletion failed. |
| `2` | Command-line syntax is invalid. |

## Installer diagnostics

A failed release download stops before checksum validation. A checksum failure
stops before extraction, and an extraction failure stops before installer
execution. An installer failure returns non-zero and retains the extracted
package and any partial `.venv` for inspection. The installer reports one of
these terminal errors without additional host details:

- `error: Raspberry Pi aarch64 is required`
- `error: Python 3.11 or newer is required`
- `error: python3-venv is required`
- `error: virtual environment already exists`
- `error: runtime requirements do not exist`
- `error: virtual environment could not be created`
- `error: runtime dependencies could not be installed`
- `error: input directory could not be created`
- `error: output directory could not be created`
