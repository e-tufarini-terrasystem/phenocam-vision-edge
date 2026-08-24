<!--
This document owns supported-runtime, installation, usage, metadata, batch, and
verification guidance. Source files remain the authority for implementation
details.
-->

<p align="center">
  <img src="assets/logo.svg" alt="Phenocam Vision Edge logo" width="220">
</p>

# YOLO single-image inference on Raspberry Pi

This project analyzes one local image with the included YOLO26n ONNX model. It
can create an annotated image, a privacy-blurred image, or both, and can
optionally delete the input when an enabled class is detected.
The deployment runtime is designed and tested for a Raspberry Pi 3 with 1 GB
RAM and a 64-bit Raspberry Pi OS installation.

Inference still uses the original `models/yolo26n.onnx` model and its end-to-end ONNX
graph. The Raspberry Pi runtime calls it directly through ONNX Runtime; it does
not install Ultralytics, PyTorch, or OpenCV.

Each image is processed sequentially in one ONNX session using one full-image
view plus fifteen adaptive overlapping crops. Horizontal and square images use a
`5×3` crop grid, while vertical images use `3×5`; both use 20% nominal overlap.
The EXIF-normalized RGB source supplies all sixteen views and remains the final
rendering background.
Crop detections are converted to global image coordinates, all model classes
are merged, and all rows require confidence greater than or equal to 0.30.
Duplicates are suppressed when IoU or smaller-box coverage
reaches 0,50; `car`, `bus`, and `truck` compete across labels, while other
classes compete only with themselves. `phenocam/classes/configuration.py` then
selects the final annotations and privacy regions.

## Runtime requirements

- Raspberry Pi OS 64-bit (`aarch64`)
- Python 3.11 or newer and `python3-venv` (tested with Python 3.13.5)
- `curl`, `tar`, and `sha256sum`
- `numpy`, `onnxruntime`, and `Pillow` from `requirements/runtime.txt`

The Python 3.13 dependency set in the requirements file was installed and
tested on a four-core Cortex-A53 Raspberry Pi. Python 3.11 uses its compatible
NumPy pin. The optional export workflow is separate and is not intended to run
on the Pi.

## Raspberry Pi installation

Install the operating-system prerequisites separately before installing the
application. The versioned command below never invokes `sudo`, `apt`, or another
system package manager.

### Versioned installation (v0.1.0)

Run this command from the directory that should contain the installation. It
stops before downloading when `phenocam-vision-edge-0.1.0/` already exists,
downloads both release assets with one `curl` process, verifies their SHA-256
checksum, extracts the archive, and runs the bundled installer:

```sh
[ ! -e phenocam-vision-edge-0.1.0 ] && \
curl --fail --fail-early --location --silent --show-error \
  --output phenocam-vision-edge-0.1.0.tar.gz \
  https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/releases/download/v0.1.0/phenocam-vision-edge-0.1.0.tar.gz \
  --output phenocam-vision-edge-0.1.0.tar.gz.sha256 \
  https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/releases/download/v0.1.0/phenocam-vision-edge-0.1.0.tar.gz.sha256 && \
sha256sum -c phenocam-vision-edge-0.1.0.tar.gz.sha256 && \
tar -xzf phenocam-vision-edge-0.1.0.tar.gz && \
./phenocam-vision-edge-0.1.0/scripts/installer.sh
```

Success leaves the configured application at
`./phenocam-vision-edge-0.1.0/`, including the ONNX model, `.venv/`, `input/`,
and `output/`. The downloaded archive and checksum remain in the current
directory. The checksum detects corruption or a mismatched download; it is not
a publisher signature.

A failed download stops before checksum validation. A checksum failure stops
before extraction, and an extraction failure stops before installer execution.
An installer failure returns non-zero and retains the extracted package and any
partial `.venv` for inspection. The installer reports one of these terminal
errors without additional host details:

- `error: Raspberry Pi aarch64 is required`
- `error: Python 3.11 or newer is required`
- `error: python3-venv is required`
- `error: virtual environment already exists`
- `error: runtime requirements do not exist`
- `error: virtual environment could not be created`
- `error: runtime dependencies could not be installed`
- `error: input directory could not be created`
- `error: output directory could not be created`

### Manual setup from source

From a source checkout, create an isolated environment and disable pip's
download cache to save space on a small microSD card:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt
```

### 8 GB microSD deployment

The tested 8 GB microSD exposes about 6.9 GiB. The deployment stays small by
installing only NumPy, ONNX Runtime, and Pillow with `--no-cache-dir`;
Ultralytics, PyTorch, OpenCV, and the export toolchain are not required on the
Pi. The virtual environment occupies about 154 MiB and the complete tested
working copy about 187 MiB. A minimal copy may also omit `models/yolo26n.pt`,
the export scripts, and `requirements/export.txt`, but must retain
`models/yolo26n.onnx`.

## macOS installation

Install Python 3.13 with Homebrew, then create the virtual environment from the
project directory. The runtime packages in `requirements/runtime.txt` also support
macOS on Apple Silicon.

```sh
brew install python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
```

The commands in this README work without activating the environment. To use
its `python` command directly in the current shell, run `source
.venv/bin/activate`; run `deactivate` when finished. The batch can then be
started with `./scripts/batch.sh`.

## Usage

Request an annotated output:

```sh
.venv/bin/python -m phenocam \
  --input input/raspberrypi2.local_2025-12-18_141905.jpg \
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
directories must already exist. The default uses all four Pi 3 CPU cores. To
reduce CPU load or temperature at the cost of latency, set the thread count to
a value from 1 to 4:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python -m phenocam \
  --input input/example.jpg --annotated-output output/example.jpg \
  --model models/yolo26n.onnx
```

## Class selection

The fixed `phenocam/classes/configuration.py` file is loaded automatically.
Change only its existing
`True`/`False` values and keep at least one class enabled. The committed
configuration enables `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`.

The ONNX model must expose exactly the standard 80 COCO classes and the
end-to-end six-column detection output used by the included YOLO26n model.
Incompatible metadata or tensor shapes are rejected before inference.

Class filtering controls final annotations, privacy regions, and the conditional
deletion trigger. The input is eligible for deletion only when at least one
enabled detection remains after normalization and global suppression. Privacy
uses rectangular regions only: each selected box expands by 10% on every side,
clips to the image, and receives Gaussian blur with radius
`max(8 px, 10% of the region's shorter side)`. It adds no boxes, names, or
confidence text. All model classes still participate in the sixteen inference
calls, merge, and global suppression, so disabling classes does not reduce
compute or memory.

## Timing

A successful command prints `Execution time: N.NNN s`. This is the measured
sum of the sixteen ONNX `session.run()` intervals. It excludes Python startup,
model/session creation, image decoding, view preparation, coordinate merging,
global suppression, final rendering, and image writing.

Raspberry Pi multi-view timing is not verified — external verification:
reference Raspberry Pi 3 is unavailable. The 15-second complete-command limit
remains an unverified acceptance target and is not inferred from workstation
measurements.

When the six ignored reference images are available, real integration tests
verify valid annotated outputs and the final suppression-domain overlap
contract. Person and car counts are not asserted because they are not ground
truth or a measurement of accuracy, precision, recall, or mAP.

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Existing local image file. |
| `--annotated-output` | Optional annotated image path; parent must exist. |
| `--privacy-output` | Optional privacy image path; parent must exist. |
| `--model` | Existing local file with a case-insensitive `.onnx` extension. |
| `--meta` | Optional existing regular non-symlink `.meta` file, distinct from input, model, and outputs. |
| `--delete-input-on-detection` | Optional boolean flag. Delete the validated input only after an enabled final detection and all requested products succeed; symbolic-link inputs are rejected. |

Requested outputs must be distinct from the input and from each other after
path, symbolic-link, and existing hard-link resolution. Existing outputs are
overwritten. With both products, annotated saving finishes first; if privacy
saving then fails, the completed annotated file remains. Zero selected
detections still writes each requested normalized source image. With conditional
deletion enabled, zero enabled final detections retain the input and the command
still succeeds. The command is headless and does not open a graphical window.

Conditional deletion is opt-in and is always the final transaction step:
inference completes first, then requested images, then requested metadata, and
only then deletion. Any earlier failure retains the input. A deletion failure
does not roll back completed products; it returns status `1` and prints exactly
`error: input image could not be deleted`. Immediately before unlinking, the
command verifies that the path is still a regular file with the validated
device/inode identity. This check does not claim atomic protection against a
hostile replacement between verification and unlink. Symbolic-link inputs are
rejected when deletion is enabled. Hard links are allowed, and only the exact
directory entry passed to `--input` is removed.

### Detection metadata

When `--meta` is supplied, the existing file is updated only after every
requested image has been written and verified. All previous exact lowercase
`[detection]` sections are removed and one current section is appended
atomically; other bytes are retained and no history is created. Counts include
only enabled final detections after global suppression, in canonical COCO order:

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
images and the previous metadata file available, returns status `1`, and prints
`error: metadata file could not be updated`. Invalid explicit metadata paths
fail before inference with `error: metadata file does not exist or is not a file` or `error: metadata file must use the .meta extension`,
and with `error: output path cannot be stored in metadata` or `error: metadata path must differ from input, model, and output paths`.

## Batch usage

Run inference on every supported image directly inside `input/`:

```sh
./scripts/batch.sh
```

The script creates `output/` when needed and writes `<stem>_annotated.<ext>` and
`<stem>_privacy.<ext>` in one process per input, preserving extension spelling.
If regular non-symlink `input/<stem>.meta` exists, it is passed through
`--meta`; a missing match is not created and does not fail that image. Existing
files are overwritten. Processing continues after individual failures, but the
script exits with status `1` if any image fails or none is supported. The batch
script never enables `--delete-input-on-detection`; input deletion remains an
explicit choice for direct CLI calls.

## Exit statuses

| Status | Meaning |
|---|---|
| `0` | Inference and every requested output, metadata update, and conditional deletion succeeded. |
| `1` | Validation, inference, output writing, metadata update, or conditional deletion failed. |
| `2` | Command-line syntax is invalid. |

## Verification

Run the complete test suite from the project directory with the runtime
environment. The six real-image integration cases and their inventory check are
skipped when the named, untracked reference images are not present in `input/`.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Optional model export

Export is a workstation task because Ultralytics brings a much larger Python
stack. Install it in a separate environment only when regeneration is needed:

```sh
python3 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements/export.txt
.venv-export/bin/python scripts/export/fp32.py
```

The included and tested `models/yolo26n.onnx` does not need to be exported on the Pi.
