# Public reference image fixtures

These six original PKLot JPEGs are the required inputs for
`tests/test_reference_images.py`. They cover the parking2 camera in cloudy,
rainy and sunny weather, with medium and high occupancy. Each image is 1280×720;
together they occupy 2,418,121 bytes. They are copied without resizing,
recompression or metadata changes from the canonical `pklot_holdout` split.
Only filenames differ from the original archive. No private operational images
are included. Runtime release archives exclude `tests/`.

## License and provenance

The [official PKLot page](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/)
licenses the database under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
(verified on 2026-09-22). This license applies to these six photographs,
independently of the repository's software license. Retain this attribution and
license link when redistributing them, and identify any subsequent changes.

Attribution: Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A.,
*PKLot – A robust dataset for parking lot classification*, Expert Systems with
Applications, 42(11):4937–4949, 2015.

Source: [official PKLot archive](https://www.inf.ufpr.br/lesoliveira/download/PKLot.tar.gz).
The local acquisition receipt records archive SHA-256
`df182f46184fc29fb7ef3cd293b49cae52599365d9fe9415c4e7e7ea7ded9f79`.
Each fixture matches both the source and compiled SHA-256 in
`dataset/data/metadata/source-images.csv`. No dataset catalog, label or split
was changed for these runtime tests.

| Fixture filename | SHA-256 |
| --- | --- |
| `pklot-parking2-cloudy-high-2012-11-08_10_50_37.jpg` | `66f0b43dbde220e96328a5914a12327c6f6415dac7ddeffc3ec94ac736612f1e` |
| `pklot-parking2-cloudy-mid-2012-09-16_12_33_39.jpg` | `a1496e0d824e0145c073676ecd78e870e914550af98e60725e9307a76991aa0e` |
| `pklot-parking2-rainy-high-2012-11-10_09_42_47.jpg` | `cee269889479d9a1774e8866e19d62fd30a5042c7212928c360bcaad37d143a0` |
| `pklot-parking2-rainy-mid-2012-11-09_18_07_05.jpg` | `9d9ae7d649de725ddae3b71fd01887bcc79dc2ad5a13bd914062bbd9c8e5dcb6` |
| `pklot-parking2-sunny-high-2012-10-17_10_59_45.jpg` | `231eb194caa0a7e2925425dc475fc073cc714410735efccb8024c61c08563420` |
| `pklot-parking2-sunny-mid-2012-10-29_07_42_56.jpg` | `7eb1fd3accaa20349cc3715690f688e183ccba45533517e5214b7753fc5ab2b1` |

Original archive paths, in the same order:

- `PKLot/parking2/cloudy/2012-11-08/2012-11-08_10_50_37.jpg`
- `PKLot/parking2/cloudy/2012-09-16/2012-10-16_12_33_39.jpg`
- `PKLot/parking2/rainy/2012-11-10/2012-11-10_09_42_47.jpg`
- `PKLot/parking2/rainy/2012-11-09/2012-11-09_18_07_05.jpg`
- `PKLot/parking2/sunny/2012-10-17/2012-10-17_10_59_45.jpg`
- `PKLot/parking2/sunny/2012-10-29/2012-10-29_07_42_56.jpg`

The cloudy-mid source intentionally retains the original archive's directory /
filename date mismatch; fixture naming follows the existing catalog.

## Tests and benchmark

Run from the repository root with the runtime environment:

```sh
YOLO_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -p test_reference_images.py -v
```

Tests verify image and model hashes, dimensions, annotated output, suppression
invariants and one reviewed car snapshot per image. The two-decimal confidence
and two-pixel box tolerances are unchanged. Snapshots describe model `0.1.6`
behavior, not accuracy or coverage of all people and vehicles. These images
belong to an already-used diagnostic split, not an untouched accuracy holdout.
Missing or altered fixtures fail; there is no fallback to `input/` and no skip.

The fixed input for future complete-command timing comparisons is
`pklot-parking2-cloudy-high-2012-11-08_10_50_37.jpg`:

```sh
reference_output=$(mktemp -d)
time env YOLO_NUM_THREADS=4 .venv/bin/python -m phenocam \
  --input tests/fixtures/reference/pklot-parking2-cloudy-high-2012-11-08_10_50_37.jpg \
  --model models/yolo26n-phenocam.onnx \
  --annotated-output "$reference_output/annotated.jpg"
```

Keep model, input, output mode, threads, device and thermal conditions fixed
between comparisons. This 1280×720 benchmark replaces the earlier operational
4608×2592 input; timings across that change do not measure a software speedup.
Use disposable copies for deletion or in-place replacement tests. See the
[hardware report](../../../docs/status/raspberry-pi-0.2.4.md) for measured results.
