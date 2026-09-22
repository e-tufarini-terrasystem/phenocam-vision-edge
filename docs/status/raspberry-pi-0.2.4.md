<!-- Record target-device evidence for one immutable runtime package. -->

# Raspberry Pi checks — software 0.2.4

Measured on 2026-09-22 over SSH, in a separate test directory on
`pheno3bplus.local`. Existing deployment files and source images were preserved.
The software remains a release candidate; model `0.1.6` remains experimental.
The initial operational-image checks and their four resolved failures are
historical evidence. Maintained fixtures now use public PKLot images (CC BY 4.0);
the operational JPEGs are excluded from the branch history to be published.

## Identity and environment

| Item | Observed value |
| --- | --- |
| Packaged commit | `f1e2bc4578862331cc5f47426a7b4618e43c76ae` |
| Hardware | Raspberry Pi 3 Model B Plus Rev 1.4, 1 GB RAM |
| OS | Debian 13.5 (trixie), `aarch64` |
| Kernel | `6.18.34+rpt-rpi-v8` |
| Python | 3.13.5 |
| Runtime dependencies | NumPy 2.5.1, ONNX Runtime 1.27.0, Pillow 12.3.0 |
| Inference | CPU, `YOLO_NUM_THREADS=4`, sixteen 640×640 views |
| Model identity | `yolo26n-phenocam`, `0.1.6` |
| ONNX SHA-256 | `72521182fa0c90fec60fb1cd9f3b2ac113d16de476aaeea3a328508b7b2d0b30` |
| Archive SHA-256 | `3b16718db1c2d97570a80759fff205b14873aeddc50e540134214582cedd0dea` |

The archive was built using `scripts/package.py 0.2.4` and the commit above,
then transferred, checksum-verified and extracted. Its packaged installer
created a new virtual environment successfully. All 25 packaged files were
subsequently checked against their archive hashes and remained unchanged.
Test support from the same commit was added separately.

At the user's request, the unavailable reference
`raspberrypi2.local_2025-12-18_141905.jpg` was replaced by the available
`raspberrypi2.local_2025-12-19_141905.jpg`. The initial filename substitution
preserved all assertions, including the four existing expected detection checks.
All six reference images were recovered from the local dataset
catalog, verified by SHA-256 and confirmed as 4608×2592 pixels. This is an
explicit fixture revision, not a claim to have tested the missing image.
The initial fixture-only revision of `tests/test_reference_images.py` had SHA-256
`f0668374da5005635f5841c330a109c3da236721b027a5b6f232b3313b893905`.

## Initial functional checks

| Check | Result |
| --- | --- |
| Packaged installer, checksum, model receipt, unchanged packaged files | PASS |
| Pi test suite | FAIL: 277 passed, 4 failed, 0 skipped; 281 tests in 154.574 s |
| Six reference images plus inventory | 3 tests passed, 4 exact-detection expectations failed |
| Annotated and privacy JPEGs, dimensions and decoding | PASS; both outputs also visually inspected |
| Metadata-only; software/model identity; unrelated metadata preserved | PASS |
| Missing receipt gives `unknown` identity and version | PASS |
| Invalid/mismatched receipt fails before output writes or deletion | PASS |
| Positive conditional deletion; negative preserves inputs and existing output | PASS |
| Corrupt image and missing input return failure without replacing output | PASS |
| Real batch: two positive inputs, one negative, spaces, optional metadata | PASS; 59.919 s total |
| Runtime dependency consistency and runtime shell syntax on Pi | PASS |
| Dataset suite on Mac | PASS: 67 tests; not a Pi dataset-suite result |

The reference failures are `2025-11-19_121905`, `2025-11-19_141905`,
`2025-11-19_151905` and `2025-12-19_124905`. Each reached the final
`EXPECTED_WINNERS` check after passing image-output, confidence and suppression
invariants, then found no exact expected detection. The same four failures
reproduce on the Mac with the packaged specialized model. A separate diagnostic
using base `yolo26n.onnx` passes all seven reference tests. Git commit `545b212`
introduced those expected values with that base model; the current test selects
the specialized model. This identifies a model/expectation mismatch, not a
Pi-specific numerical failure. Expectations were left unchanged in that initial
run; the subsequent review and test update are recorded below. These expected
detections are runtime snapshots, not human accuracy labels.

The base diagnostic used ONNX SHA-256
`09fa4b119751eafe13962607f5a9742aff9f8e3e3e3a950c057ecb8d40cd82ec`.
It is diagnostic evidence only; the deployed model remains `0.1.6`.

The disposable functional harness passed 15 command checks, including the five
benchmark repetitions. An initial probe used a recent image with no detections;
its CLI succeeded and wrote correct negative metadata, but the harness had
incorrectly assumed a positive image. The completed checks use a reference image
with detections. The first isolated batch fixture lacked the package import
path; after linking the unchanged packaged runtime, the batch was rerun.
These setup errors did not require application changes.

## Reviewed reference expectations

After the user approved aligning the tests, the four annotated outputs were
visually reviewed and matched to the same objects as the original cases. The
selected boxes overlap the old boxes by IoU 0.840–0.970. Independent runs on
Mac and Pi produced the same final detection identities on all four images;
the maximum coordinate difference was 0.000440 pixels and the maximum confidence
difference was 0.00000710. Model and image hashes matched across both machines.

| Image timestamp | Snapshot class | Confidence | Source-image box (pixels) |
| --- | --- | ---: | --- |
| 2025-11-19 12:19:05 | car | 0.85 | (2203, 2304, 2704, 2541) |
| 2025-11-19 14:19:05 | car | 0.74 | (1831, 2386, 2378, 2589) |
| 2025-11-19 15:19:05 | car | 0.76 | (1754, 2431, 2230, 2590) |
| 2025-12-19 12:49:05 | bus | 0.89 | (1794, 1556, 2531, 2055) |

The fourth object is a **truck**, confirmed visually and by its human annotation
(class 7) in
`dataset/data/labels/test_ood/raspberrypi2.local--2025-12-19T124905--c8c7d046603f.txt`.
Model `0.1.6` predicts `bus` for that object. The new expectation records this
known classification error as a reproducible runtime output; it does not
approve the label or establish model accuracy. The test includes that limitation
as an inline comment. The model remains experimental.

The existing two-pixel coordinate tolerance, two-decimal confidence comparison,
class check, image-output checks and suppression invariants are unchanged.
A `setUp` check now binds every reference test to the exact specialized ONNX
SHA-256 above, including when images are unavailable. A substitution probe
confirmed that changed model bytes fail before inference or an image skip.
The updated test file is 188 lines and has SHA-256
`3b0f7f3ba8d15df9e43e5a10e30dd78bfcf2aa5b03959cf0b4c3629e19ae2fe8`.
Runtime, model, dependency and packaged-file bytes are unchanged.

Validation used `YOLO_NUM_THREADS=4` in isolated copies with all six images:

```sh
.venv/bin/python -m unittest discover -s tests -p test_reference_images.py -v
.venv/bin/python -m unittest discover -s tests -v
```

The focused Mac run passed all seven tests. Full suites passed **281 tests with
zero failures and zero skips** on both Mac (21.112 s) and Pi (153.679 s).
Runtime shell syntax and `git diff --check` also passed. On the Pi, all 25
packaged files still matched their archive hashes after the test update.
The updated tests were supplied separately to the original packaged commit;
this run does not qualify a new release archive or later commit.

Follow-up evidence is under
`output/qualification/2026-09-22-pi-0.2.4/reference-update/`, including both
platforms' detections, their comparison, the model-substitution probe, test logs
and provenance. Original failing logs remain available in the parent directory.

## Time, memory and thermal conditions

Each repetition launched a new CLI process on
`raspberrypi2.local_2025-11-19_121905.jpg`, SHA-256
`d45590988493dcaa68499ef7ceb065e478b9e4883645bcbe5cec8e312e5d4607`,
with annotated JPEG output, the same model, four threads and default thresholds.
Wall time covers process startup through exit. Peak RSS comes from Linux
`wait4`; temperature and firmware flags were sampled about once per second.
Runs were consecutive after functional tests, with no cooldown or cache reset.

| Repetition | Complete command (s) | ONNX calls only (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| 1 | 18.459 | 13.037 | 227.39 |
| 2 | 18.500 | 13.118 | 227.49 |
| 3 | 19.624 | 14.057 | 225.95 |
| 4 | 23.155 | 17.097 | 227.20 |
| 5 | 19.706 | 14.010 | 226.68 |

Median complete-command time was **19.624 s**. The **then-current 15-second
target failed on all five repetitions**. This is one image and output mode,
not a general latency guarantee. The separate both-images-and-metadata case took 19.773 s
with 273.25 MiB peak RSS. All five benchmark commands completed successfully;
these short runs do not establish long-duration reliability.

Functional/benchmark samples ranged from 60.1°C to 78.4°C. Initial inspection
already showed historical firmware flags `0xa0000`; samples during the checks
included `0xa0008`, indicating an active soft temperature limit in addition to
historical flags. A spot measurement under load showed 1.2 GHz with the
`ondemand` governor. Flag meanings follow the
[official Raspberry Pi documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/os/graphics-utilities.adoc#get_throttled).
Thermal conditions were not held constant, so these results do not isolate the
cause of the timing variation or predict performance with different cooling.
No firmware, clock, cooling or OS settings were changed.

The maintainer subsequently approved retiring the legacy 15-second requirement.
The initial baseline was **about 20 seconds per image** on this Pi 3 B+, using
the operational reference image, four threads and annotated output (18–23 seconds
observed). It is historical evidence, with no fixed latency requirement;
timings vary with image, output mode and device conditions. Optimization is
optional future work and does not block release. Original measurements remain
unchanged; final commit qualification and model acceptance remain separate.

## Withdrawn operational fixtures

The maintainer initially requested that all six operational images be tracked.
That local revision placed 36,265,858 bytes in `tests/fixtures/reference/`, with
provenance, hashes and a benchmark command. Before any push, the maintainer
withdrew that choice and authorized recreating the two local commits with public
dataset fixtures. The historical validation below describes the withdrawn
revision; those JPEGs are not part of the maintained fixture set.

Reference tests were changed to read only that fixture directory and verify image hashes
before decoding or inference. Missing or changed fixtures fail, with no fallback
or skip. Both failure paths were exercised in disposable directories. The CI
workflow now identifies the reference-image checks as part of the required suite.
At validation time, these changes were uncommitted and had no hosted CI run.

The focused local run passed seven tests. An isolated source copy containing
the intended Git files, with neither `input/` nor `dataset/data/images/`, passed
all 281 tests without skips in 20.286 s; its 67 dataset tests also passed in
2.483 s. The Pi rerun using the new fixture directory passed all 281 tests
without skips in 149.341 s. All six image blobs staged in Git were verified
against the source hashes. The relocated test file is 190 lines, with
SHA-256 `47cbc3c1542628966eb13c9f38fef5ed9e3bbc1e3930e8918f80391f29f36372`.
Evidence is retained under
`output/qualification/2026-09-22-pi-0.2.4/versioned-fixtures/`.

## Public PKLot fixtures

The maintained suite uses six original 1280×720 PKLot frames under CC BY 4.0,
selected from the existing `pklot_holdout` split: cloudy, rainy and sunny weather,
each at medium and high occupancy. The 2,418,121 image bytes match the canonical
source hashes. No training catalog or split changed. The fixture README records
the official license, attribution, original archive paths and per-image hashes.
These are already-used diagnostic images, not a new accuracy holdout.

The two unpushed local commits containing operational images were withdrawn
with the maintainer's approval. Their six image blobs were checked against all
current branch/tag object histories and were absent. They can still exist in
local reflogs or ignored evidence; those are not published branch history.
No remote history was rewritten.

All six annotated outputs were visually reviewed. Independent Mac/Pi runs
produced identical final detection identities on every image (981 detections
in total), with maximum coordinate difference 0.0000611 pixels and confidence
difference 0.0000651. Each fixture pins one reviewed car snapshot to the same
model hash; the original two-pixel and two-decimal tolerances remain unchanged.
Suppression, image-output and hash checks are retained. Disposable missing and
altered fixture probes failed before inference, with no skip or fallback.

The focused Mac suite passed 7 tests in 3.118 s. Full suites passed **281 tests,
zero failures and zero skips**, on Mac (19.438 s) and Pi (145.776 s). The Mac
dataset suite passed 67 tests in 1.956 s. Runtime and dataset shell syntax passed.
The Pi source copy contained the tracked source and public fixtures only, with
no private dataset images or operational `input/` directory.

The maintained reference test has SHA-256
`1303fae4ac80c6b1f94a264331d9c8f5678270151a5843878b76eef98db05041`.
Evidence is retained under
`output/qualification/2026-09-22-pi-0.2.4/public-fixtures/` and in the isolated Pi
directory `~/phenocam-public-qualification-9NFnfR/`. These source checks precede
the final release archive; they do not claim qualification of a later commit.

### Public-input functional checks and timing

The disposable CLI harness passed all 15 checks again with
`pklot-parking2-cloudy-high-2012-11-08_10_50_37.jpg`: annotated/privacy outputs,
metadata identity and preservation, absent/invalid/mismatched receipts,
conditional deletion, negative input, corrupt/missing input, and five timings.
The model and environment are those recorded above; four threads and sixteen
views are unchanged. Each benchmark starts a fresh process and writes annotated
JPEG output. Runs follow functional tests without a cooldown or cache reset.

| Repetition | Complete command (s) | ONNX calls only (s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| 1 | 18.336 | 13.487 | 115.60 |
| 2 | 18.162 | 13.330 | 115.60 |
| 3 | 18.130 | 13.299 | 115.58 |
| 4 | 18.349 | 13.502 | 115.57 |
| 5 | 18.141 | 13.327 | 115.62 |

Median complete-command time is **18.162 s**, summarized as **about 18 seconds**
for this public input. This is an indicative baseline, with no fixed latency
requirement. Functional/benchmark temperatures ranged from 60.7°C to 74.1°C;
all sampled firmware flags were `0xa0008` (active soft temperature limit).
The both-images-and-metadata case took 18.581 s with 115.78 MiB peak RSS.
The real batch rerun passed in 52.782 s with two public positive inputs, one
synthetic negative, spaces in paths and optional metadata; sources were preserved.
The new image is 1280×720 rather than 4608×2592, with a different detection load;
these figures do not measure a software speedup over the earlier benchmark.
No runtime, model, dependency, clock or cooling change was made.

## Remaining work

- Complete release gates on the final chosen commit: passing GitHub CI,
  regenerated and verified archive/checksum, and qualification of that exact
  candidate if it differs from the commit above. No GitHub Actions run was
  returned for the tested commit during this check. Tags and publication were
  not performed by this test task.
- Model acceptance remains separate: these hardware checks do not establish
  detection accuracy or promote model `0.1.6` out of experimental status.

Raw logs, fixture hashes, harnesses and measurements are retained locally under
`output/qualification/2026-09-22-pi-0.2.4/` (ignored by Git). The remote test
root is `~/phenocam-qualification-20260922-CUEISw/`. The runtime archive and
checksum are preserved there. This report records the tested commit and fixture
revision; it does not automatically qualify subsequent changes.
