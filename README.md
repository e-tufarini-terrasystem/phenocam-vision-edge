<!--
This document introduces Phenocam Vision Edge and owns the supported release
installation path. Detailed operation and development guidance live in docs/.
-->

<p align="center">
  <img src="assets/logo.svg" alt="Phenocam Vision Edge logo" width="220">
</p>

# Phenocam Vision Edge

Phenocam Vision Edge analyzes local images with the included YOLO26n ONNX
model. It runs inference directly on a Raspberry Pi, without sending images to
an external service, and can produce annotated images, privacy-blurred images,
or both. It can also delete an input after an enabled class is detected.

The deployment runtime is designed and tested for a Raspberry Pi 3 with 1 GB
RAM and a 64-bit Raspberry Pi OS installation. It processes images sequentially
through ONNX Runtime and does not install Ultralytics, PyTorch, or OpenCV.

Each image is evaluated as one full-image view plus fifteen adaptive overlapping
crops. Detections are merged into the source coordinate space, filtered, and
used to produce the requested outputs according to the fixed class
configuration.

## Runtime requirements

- Raspberry Pi OS 64-bit (`aarch64`)
- Python 3.11 or newer and `python3-venv`
- `curl`, `tar`, and `sha256sum`

## Raspberry Pi installation

Install the operating-system prerequisites separately before installing the
application. The versioned command below never invokes `sudo`, `apt`, or another
system package manager.

### Versioned installation (v0.1.0)

Run this command from the directory that should contain the installation. It
stops before downloading when `phenocam-vision-edge-0.1.0/` already exists,
downloads the release archive and its checksum, verifies the archive, extracts
it, and runs the bundled installer:

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
and `output/`. The archive and checksum remain in the current directory. The
checksum detects corruption or a mismatched download; it is not a publisher
signature.

## Documentation

- [Software operation instructions and CLI reference](docs/cli.md)
- [Manual installation and configuration](docs/manual.md)
- [Development, verification, and model export](docs/development.md)
