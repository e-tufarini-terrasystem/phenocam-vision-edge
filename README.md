<!-- Own the project introduction and versioned release installation. -->

<p align="center">
  <img src="assets/logo.svg" alt="Phenocam Vision Edge logo" width="220">
</p>

# Phenocam Vision Edge

[![CI](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/actions/workflows/ci.yml)

Detect people and vehicles in local PhenoCam images on a Raspberry Pi, using
NumPy, ONNX Runtime and Pillow. Images stay on the device. Each input uses one
full-image view and fifteen overlapping crops, processed sequentially.

The command can write annotated images, blur detected regions, update an
existing metadata file, or conditionally delete the input and supplied metadata.
Image outputs are produced only when an enabled final detection remains;
otherwise existing images stay unchanged. Deletion is opt-in and takes precedence
over output writes. See the [operating guide](docs/cli.md) for the complete behavior.

## Stable software release v0.2.3

Software release **v0.2.3** is considered stable by the maintainer following
manual testing on Raspberry Pi. This status applies to the software runtime.

The bundled model is `models/yolo26n-phenocam.onnx`, model version `0.1.6`.
Its sibling JSON identifies the model
and its ONNX SHA-256. The model remains **experimental**: training acceptance
criteria were not all met, and overall improvement was not demonstrated.
Software stability does not establish detection accuracy.

The [training report](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/docs/status/training-0.1.6.md)
and [model comparison](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/docs/status/model-comparison-2026-09-15.md)
preserve the evidence and limitations. These repository reports are not included
in the runtime archive.

Published releases are frozen. The current source tree has later changes,
including the optional base ONNX model and its receipt; see
[release and source differences](docs/cli.md#release-and-source-differences).

## Supported environments

This section applies only to the latest stable software release, **v0.2.3**.

| Platform | Required Python |
|---|---|
| Raspberry Pi OS 64-bit (`aarch64`) | 3.13 |

The target device is a Raspberry Pi 3 with 1 GB RAM.
The maintainer reports successful manual testing on Raspberry Pi.
The exact Pi model, OS/Python versions and individual checks were not
recorded; the requirements above are not a recorded test configuration.

For installation from the current source tree, see
[manual setup from source](docs/cli.md#manual-setup-from-source).
For development and CI, see [development](docs/development.md).

## Quick install: v0.2.3

Prerequisites: Raspberry Pi OS 64-bit, Python 3.13 with `python3-venv`,
`curl`, `tar`, and `sha256sum`. Install OS prerequisites separately; this
command never invokes `sudo` or a system package manager. Downloads are public
and need no GitHub account or token.

Run from the parent directory of the new installation. The directory
`phenocam-vision-edge-0.2.3/` must not already exist.

### Versioned installation (v0.2.3)

```sh
[ ! -e phenocam-vision-edge-0.2.3 ] && \
curl --fail --fail-early --location --silent --show-error \
  --output phenocam-vision-edge-0.2.3.tar.gz \
  https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/releases/download/v0.2.3/phenocam-vision-edge-0.2.3.tar.gz \
  --output phenocam-vision-edge-0.2.3.tar.gz.sha256 \
  https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/releases/download/v0.2.3/phenocam-vision-edge-0.2.3.tar.gz.sha256 && \
sha256sum -c phenocam-vision-edge-0.2.3.tar.gz.sha256 && \
tar -xzf phenocam-vision-edge-0.2.3.tar.gz && \
./phenocam-vision-edge-0.2.3/scripts/installer.sh
```

Success leaves the configured application at
`./phenocam-vision-edge-0.2.3/`, including the ONNX model, `.venv/`, `input/`,
and `output/`. The archive and checksum remain in the current directory. The
checksum detects corruption or a mismatched download; it is not a publisher
signature.

## First inference

Copy an image into the installation and request both output products:

```sh
cd phenocam-vision-edge-0.2.3
cp /path/to/image.jpg input/example.jpg
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/example_annotated.jpg \
  --privacy-output output/example_privacy.jpg \
  --model models/yolo26n-phenocam.onnx
```

The command leaves the source image unchanged. See the
[CLI guide](docs/cli.md) for metadata, batch processing, thread control, and
conditional input deletion.

## Documentation

- [Installation, configuration and CLI](docs/cli.md): operating the software.
- [Development](docs/development.md): architecture, tests, training and export.
- [Dataset](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/README.md): catalog, access, reconstruction and verification.
- [Annotation](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/blob/main/dataset/docs/annotation-guide-it.md): human review and CVAT.

Dataset tools and annotation documents belong to the source repository. Historical
reports and AI development instructions remain there separately from these guides.

## Support and license

Report reproducible problems through [GitHub Issues](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/issues).
Repository code is distributed under the [Apache License 2.0](LICENSE).
Model and dataset terms are separate from the repository code license.
