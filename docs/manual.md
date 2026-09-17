<!--
This document owns manual installation, class configuration, batch operation,
storage guidance, and installer diagnostics for Phenocam Vision Edge.
-->

# Manual installation and configuration

For instructions on using the installed software, see the
[software operation and CLI guide](cli.md).

## Manual setup from source

On a 64-bit Raspberry Pi OS installation, create an isolated environment and
disable pip's download cache to save space on a small microSD card:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements/runtime.txt
```

The runtime requires Python 3.13. Its dependency set was installed and tested
with Python 3.13.5 on a four-core Cortex-A53 Raspberry Pi.

### 8 GB microSD deployment

The tested 8 GB microSD exposes about 6.9 GiB. The deployment stays small by
installing only NumPy, ONNX Runtime, and Pillow with `--no-cache-dir`;
Ultralytics, PyTorch, OpenCV, and the export toolchain are not required on the
Pi. The virtual environment occupies about 154 MiB and the complete tested
working copy about 187 MiB.

A minimal source copy may omit PT checkpoints, `training/`, `dataset/`,
workstation scripts and workstation requirements. It must retain
`models/yolo26n-phenocam.onnx`. The figures above describe the historical
runtime environment, not a new qualification of the specialized model.

### macOS

Install Python 3.13 with Homebrew, then create the virtual environment from the
project directory. The runtime requirements also support macOS on Apple
Silicon.

```sh
brew install python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements/runtime.txt
```

The commands work without activation. To use the environment's `python`, run
`source .venv/bin/activate`; run `deactivate` when finished.

## Class configuration

The fixed `phenocam/classes/configuration.py` file is loaded automatically.
Change only its existing `True` and `False` values and keep at least one class
enabled. The committed configuration enables `person`, `car`, `motorcycle`,
`bus`, and `truck`. A bicycle without a person is not a privacy target; a
cyclist is protected by the `person` box.

Class filtering controls final annotations, privacy regions, and the conditional
deletion trigger. All model classes still participate in inference, merging,
and global suppression, so disabling classes does not reduce compute or memory.

Privacy uses rectangular regions only. Each selected box expands by 10% on
every side, clips to the image, and receives Gaussian blur with radius
`max(8 px, 10% of the region's shorter side)`. It adds no boxes, names, or
confidence text.

## Batch usage

Run inference on every supported image directly inside `input/`:

```sh
./scripts/batch.sh
```

The script creates `output/` when needed and writes
`<stem>_annotated.<ext>` only when an enabled final detection remains, in one
process per input, preserving extension spelling. Privacy output remains
available through the direct CLI. If a regular
non-symlink
`input/<stem>.meta` exists, it is passed through `--meta`; a missing match is
not created and does not fail that image.

The batch script, installer and runtime package use the same
`models/yolo26n-phenocam.onnx`. Its bytes match the former v6 model, whose
experimental status is unchanged. The current comparison is recorded in
`docs/status/model-comparison-2026-09-15.md`; specialized-model Raspberry Pi
timing still requires a target-device run.

Existing outputs are atomically replaced when an enabled final detection
remains; otherwise they are left untouched and no new image is created.
Metadata still records the current detection result, with empty image paths
when no image was produced. Processing continues after individual
failures, but the script exits with status `1` if any image fails or none is
supported. The batch script never enables `--delete-input-on-detection`.

## Installer diagnostics

A failed release download stops before checksum validation. A checksum failure
stops before extraction, and an extraction failure stops before installer
execution. An installer failure returns non-zero and retains the extracted
package and any partial `.venv` for inspection. The installer reports one of
these terminal errors without additional host details:

- `error: Raspberry Pi aarch64 is required`
- `error: Python 3.13 is required`
- `error: python3-venv is required`
- `error: virtual environment already exists`
- `error: runtime requirements do not exist`
- `error: runtime model does not exist`
- `error: virtual environment could not be created`
- `error: runtime dependencies could not be installed`
- `error: input directory could not be created`
- `error: output directory could not be created`
