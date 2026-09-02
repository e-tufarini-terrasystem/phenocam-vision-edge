<!--
This document introduces Phenocam Vision Edge and owns the supported release
installation path. Detailed operation and development guidance live in docs/.
-->

<p align="center">
  <img src="assets/logo.svg" alt="Phenocam Vision Edge logo" width="220">
</p>

# Phenocam Vision Edge

[![CI](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/actions/workflows/ci.yml)

Phenocam Vision Edge detects people and vehicles in local PhenoCam images with
the included YOLO26n ONNX model. It runs directly on a Raspberry Pi without
sending images to an external service and produces annotated images,
privacy-blurred images, or both. It can also delete an input after an enabled
class is detected.

## Why Phenocam Vision Edge?

- Designed and tested for a Raspberry Pi 3 with 1 GB RAM.
- Uses only NumPy, ONNX Runtime, and Pillow on the deployment device.
- Processes images sequentially to keep resource ownership predictable.
- Evaluates one full image and fifteen adaptive overlapping crops for each input.
- Keeps output classes explicit through one fixed configuration file.

Phenocam Vision Edge is a focused single-image runtime, not a general-purpose
object-detection or model-training framework.

## Supported environments

| Use | Platform | Python | Scope |
|---|---|---|---|
| Stable `v0.1.0` | Raspberry Pi OS 64-bit (`aarch64`) | 3.11 or newer | Frozen release contract |
| Current `main` | Raspberry Pi OS 64-bit (`aarch64`) | 3.13 | Deployment target |
| Current `main` | macOS on Apple Silicon | 3.13 | Manual source setup |
| CI | Ubuntu x86-64 | 3.13 | Tests and shell syntax only |

The CI badge does not qualify Raspberry Pi hardware. See the
[development guide](docs/development.md) for the complete verification scope.

## Quick install: stable v0.1.0

The versioned installer requires Raspberry Pi OS 64-bit, Python 3.11 or newer
with `python3-venv`, and `curl`, `tar`, and `sha256sum`.

Install the operating-system prerequisites separately before installing the
application. The versioned command below never invokes `sudo`, `apt`, or another
system package manager. Run it from the directory that should contain the
installation:

### Versioned installation (v0.1.0)

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

## First inference

Copy an image into the installation and request both output products:

```sh
cd phenocam-vision-edge-0.1.0
cp /path/to/image.jpg input/example.jpg
.venv/bin/python -m phenocam \
  --input input/example.jpg \
  --annotated-output output/example_annotated.jpg \
  --privacy-output output/example_privacy.jpg \
  --model models/yolo26n-v3-extended.onnx
```

The command leaves the source image unchanged. See the
[CLI guide](docs/cli.md) for metadata, batch processing, thread control, and
conditional input deletion.

## Outputs and deletion safety

| Result | Option | Behavior |
|---|---|---|
| Annotated image | `--annotated-output` | Draws enabled detections and labels. |
| Privacy image | `--privacy-output` | Blurs expanded regions without labels. |
| Detection metadata | `--meta` | Atomically updates an existing `.meta` file. |
| Conditional deletion | `--delete-input-on-detection` | Removes the input only after an enabled final detection and all requested products succeed. |

Deletion is disabled by default. At least one image output or conditional
deletion is required; metadata alone is not a final action.

## Stable release and main

The installation command above deliberately downloads the published `v0.1.0`
release, a frozen snapshot of the software associated with the annotated
`v0.1.0` tag. It does not download the current contents of `main`. The `main`
branch contains ongoing development and may advance freely beyond this release.
Release claims and verification apply only to the exact tagged commit.

## Documentation

- [Software operation instructions and CLI reference](docs/cli.md)
- [Manual installation and configuration](docs/manual.md)
- [Development, verification, and model export](docs/development.md)
- [Training dataset composition and reproducible builder](dataset/README.md)
- [V2, v3, and v3-expanded training notebooks](notebooks/)

## Support and license

Use [GitHub Issues](https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/issues)
to report a reproducible problem or request clarification. Phenocam Vision Edge
is available under the [Apache License 2.0](LICENSE).
