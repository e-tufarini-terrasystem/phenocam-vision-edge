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

Request products and conditional input deletion together, or use deletion-only mode:

```sh
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/annotated.jpg \
  --privacy-output output/privacy.jpg \
  --model models/yolo26n-phenocam.onnx \
  --delete-input-on-detection

.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --model models/yolo26n-phenocam.onnx \
  --delete-input-on-detection
```

At least one image output or `--delete-input-on-detection` is required; metadata
alone is not a final action. Deletion-only mode still performs the complete inference.
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
`error: input image could not be deleted`. When no enabled final detection
remains, the input is retained and the command still succeeds.

Immediately before unlinking, the command verifies that the path is still a
regular file with the validated device/inode identity. This check does not
claim atomic protection against a hostile replacement between verification and
unlink. Hard links are allowed, and only the exact directory entry passed to
`--input` is removed.

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
model_id=yolo26n-phenocam
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

## Timing and exit statuses

A successful command prints `Execution time: N.NNN s`, the sum of the sixteen
ONNX `session.run()` intervals. It excludes startup, session creation, image
preparation, rendering, and writing. Raspberry Pi multi-view timing is not
verified; the 15-second complete-command limit remains an unverified target.

| Status | Meaning |
|---|---|
| `0` | Inference and every requested output, metadata update, and conditional deletion succeeded. |
| `1` | Validation, inference, output writing, metadata update, or conditional deletion failed. |
| `2` | Command-line syntax is invalid. |
