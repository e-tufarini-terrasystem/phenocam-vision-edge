# YOLO Single-Image Inference

This project annotates one local image using a local YOLO model in ONNX format
and writes the annotated image to a chosen local output path.

## Requirements

- Python 3
- `ultralytics`
- `onnxruntime`

## Installation

Using a virtual environment is recommended. Install the runtime packages with:

```sh
python3 -m pip install ultralytics onnxruntime
```

## Export the ONNX model

The repository includes `yolo26n.pt`. Export it with the existing script:

```sh
python3 export_onnx.py
```

This creates `yolo26n.onnx` from `yolo26n.pt` in the current directory.
Generated ONNX models are runtime artifacts rather than source files.

## Class selection

The fixed `classes.py` file in the repository root is loaded automatically;
there is no CLI option for selecting another configuration. It contains all 80
COCO classes grouped into the 12 standard categories. Change only the existing
`True`/`False` values, and keep at least one class enabled. The committed
configuration enables only `person`.

The selected ONNX model must expose exactly the standard 80 COCO classes. A
custom-class or otherwise incompatible model is rejected. Invalid configuration
stops the command before model inference.

Class filtering limits the detections returned and annotated. It does not
guarantee reduced neural-network inference time, CPU or GPU use, or memory use.
A valid run with no matching detections still succeeds and saves the output
image without annotations.

## Usage

Run inference with all three required options:

```sh
python3 run.py --input /path/to/image.jpg --output ./annotated.jpg --model ./yolo26n.onnx
```

## Arguments

| Option | Constraint |
|---|---|
| `--input` | Existing local image file. |
| `--output` | Output image path, distinct from the input; its parent directory must already exist. |
| `--model` | Existing local file with a case-insensitive `.onnx` extension. |

## Behavior

- An existing output file is overwritten.
- The output parent directory is not created automatically.
- The command does not open a graphical window or print detections.
- Successful execution is quiet and writes only the annotated image.

Generated models and annotated images are not repository source artifacts.

## Exit statuses

| Status | Meaning |
|---|---|
| `0` | Inference and output writing succeeded. |
| `1` | Path validation, inference, or output writing failed. |
| `2` | The command-line syntax is invalid. |

Operational errors are written to standard error without exposing local paths
or third-party exception details.

Class-selection failures use these status-`1` diagnostics:

- `error: class configuration is invalid`
- `error: model classes are incompatible`

## Limitations

- Input is limited to one local image per execution.
- Models must be local ONNX files.
- Batch input, directories, URLs, webcams, videos, and standard input are not
  supported.
