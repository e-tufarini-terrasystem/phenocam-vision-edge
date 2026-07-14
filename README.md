# YOLO single-image inference on Raspberry Pi

This project annotates one local image with the included YOLO26n ONNX model.
The deployment runtime is designed and tested for a Raspberry Pi 3 with 1 GB
RAM and a 64-bit Raspberry Pi OS installation.

Inference still uses the original `yolo26n.onnx` model and its end-to-end ONNX
graph. The Raspberry Pi runtime calls it directly through ONNX Runtime; it does
not install Ultralytics, PyTorch, or OpenCV.

Each image is processed sequentially in one ONNX session using one full-image
view plus eight adaptive overlapping crops. Horizontal and square images use a
`4×2` crop grid, while vertical images use `2×4`; both use 20% nominal overlap.
Crop detections are converted to global image coordinates, all model classes
are merged, and same-class boxes are deduplicated with IoU-0.50 NMS before
`classes.py` selects the final annotations.

## Runtime requirements

- Raspberry Pi OS 64-bit (`aarch64`)
- Python 3.11 or newer and `python3-venv` (tested with Python 3.13.5)
- `numpy`, `onnxruntime`, and `Pillow` from `requirements-rpi.txt`

The Python 3.13 dependency set in the requirements file was installed and
tested on a four-core Cortex-A53 Raspberry Pi. Python 3.11 uses its compatible
NumPy pin. The optional export workflow is separate and is not intended to run
on the Pi.

## Raspberry Pi installation

Create an isolated environment and disable pip's download cache to save space
on a small microSD card:

```sh
sudo apt update
sudo apt install --no-install-recommends python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-rpi.txt
```

The tested virtual environment occupies about 154 MiB. The application,
environment, included models, source images, and generated test outputs occupy
about 187 MiB in total.

For the smallest deployment copy, `yolo26n.pt`, `export_onnx.py`,
`export_onnx_int8.py`, and `requirements-export.txt` may be omitted. They are
export-time assets and are not read by `run.py`. Keep `yolo26n.onnx`.

## macOS installation

Install Python 3.13 with Homebrew, then create the virtual environment from the
project directory. The runtime packages in `requirements-rpi.txt` also support
macOS on Apple Silicon.

```sh
brew install python@3.13
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-rpi.txt
```

The commands in this README work without activating the environment. To use
its `python` command directly in the current shell, run `source
.venv/bin/activate`; run `deactivate` when finished. The batch can then be
started with `./batch.sh`.

## Usage

Run inference with all three required options:

```sh
.venv/bin/python run.py \
  --input images/raspberrypi2.local_2025-12-18_141905.jpg \
  --output output/annotated.jpg \
  --model yolo26n.onnx
```

The output directory must already exist. The default uses all four Pi 3 CPU
cores. To reduce CPU load or temperature at the cost of latency, set the
thread count to a value from 1 to 4:

```sh
YOLO_NUM_THREADS=2 .venv/bin/python run.py \
  --input images/example.jpg --output output/example.jpg --model yolo26n.onnx
```

## Class selection

The fixed `classes.py` file is loaded automatically. Change only its existing
`True`/`False` values and keep at least one class enabled. The committed
configuration enables `person` and `car`.

The ONNX model must expose exactly the standard 80 COCO classes and the
end-to-end six-column detection output used by the included YOLO26n model.
Incompatible metadata or tensor shapes are rejected before inference.

Class filtering controls which final detections are annotated. All model
classes still participate in the nine inference calls, merge, and NMS, so
disabling classes does not reduce neural-network compute or memory requirements.

## Timing

A successful command prints `Execution time: N.NNN s`. This is the measured
sum of the nine ONNX `session.run()` intervals. It excludes Python startup,
model/session creation, image decoding, view preparation, coordinate merging,
NMS, drawing, and JPEG writing.

Raspberry Pi multi-view timing is not verified — external verification:
reference Raspberry Pi 3 is unavailable. The 15-second complete-command limit
remains an unverified acceptance target and is not inferred from workstation
measurements.

The six included images have fixed detection-count snapshots used to detect
behavioral regressions in the committed model and dependencies.
Reference counts are a regression snapshot, not ground truth or a measurement of accuracy, precision, recall, or mAP.
They do not prove that individual boxes are correct.

See `MODIFICHE_RASPBERRY_PI.md` for the full measurements and
`LIMITAZIONI_RASPBERRY_PI.md` for deployment constraints.

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Existing local image file. |
| `--output` | Output path distinct from the input; parent must exist. |
| `--model` | Existing local file with a case-insensitive `.onnx` extension. |

An existing output file is overwritten. The command is headless and does not
open a graphical window.

## Batch usage

Run inference on every supported image directly inside `images/`:

```sh
./batch.sh
```

The script creates `output/` when needed and writes each result with the input
filename. Existing output files with the same name are overwritten. Processing
continues after individual failures, but the script exits with a non-zero status
if any image fails.

## Exit statuses

| Status | Meaning |
|---|---|
| `0` | Inference and output writing succeeded. |
| `1` | Validation, inference, or output writing failed. |
| `2` | Command-line syntax is invalid. |

## Optional model export

Export is a workstation task because Ultralytics brings a much larger Python
stack. Install it in a separate environment only when regeneration is needed:

```sh
python3 -m venv .venv-export
.venv-export/bin/python -m pip install -r requirements-export.txt
.venv-export/bin/python export_onnx.py
```

The included and tested `yolo26n.onnx` does not need to be exported on the Pi.
