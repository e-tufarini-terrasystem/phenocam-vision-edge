---
name: release
description: >-
  Prepare, qualify, tag, and optionally publish a Phenocam Vision Edge release.
  Use when the user asks to choose or release a version, build its deterministic
  archive and checksum, create its Git tag, or publish its GitHub Release.
---

<!--
This skill owns release identity, qualification, packaging, and publication;
feature implementation and ordinary CI remain separate responsibilities.
-->

# Release

Communicate with the user in Italian. Keep commands, commit titles, release
notes, and committed technical documentation in English. Follow `AGENTS.md`.

Preserve the release invariant: one qualified source commit, one immutable
annotated tag, one deterministic allowlisted runtime archive, and one adjacent
SHA-256 record.

## Authority boundary

A request to prepare a release authorizes local version edits, validation, a
focused release commit, and local artifact generation. Creating a tag, pushing
commits or tags, and creating a GitHub Release are separate mutations. Perform
each only when the current request explicitly authorizes it.

Immediately before the first remote mutation, show the exact commit, tag, asset
names, release title, and destination remote. Never force-push, move or reuse a
release tag, or replace published assets. Do not inspect or print credentials;
use only existing authenticated Git and GitHub configuration.

Never stage or commit `docs/dataset/**`. Preserve unrelated worktree changes and
keep generated release assets outside the repository.

## Version identity

Accept a strict `MAJOR.MINOR.PATCH` version and derive:

```text
version: MAJOR.MINOR.PATCH
tag: vMAJOR.MINOR.PATCH
archive: phenocam-vision-edge-MAJOR.MINOR.PATCH.tar.gz
root: phenocam-vision-edge-MAJOR.MINOR.PATCH/
checksum: phenocam-vision-edge-MAJOR.MINOR.PATCH.tar.gz.sha256
```

When the version is omitted, inspect reachable version tags and changes since
the latest release. Propose a patch for compatible fixes and a minor version for
compatible features. Never select a major version without explicit approval.
Reject versions that are not newer than all existing local and remote release
tags.

## Preflight

1. Read `AGENTS.md`, `README.md`, `docs/development.md`, `docs/manual.md`,
   `requirements/runtime.txt`, `scripts/installer.sh`, `scripts/package.py`,
   packaging tests, and changes since the previous version tag.
2. Identify the branch, upstream, `HEAD`, previous tag, target remote, and every
   tracked or untracked change. Fetch tags read-only when network access exists.
3. Prove the target tag and assets do not already exist locally or remotely.
4. Classify user-visible, internal, compatibility, packaging, and documentation
   changes. Evidence from an older commit never qualifies a new release.
5. Record the invariant, smallest release diff, principal risk, and unavailable
   external verification before editing.

Require a clean source tree before creating a tag. Do not release a feature
branch as the canonical release branch unless the user explicitly selected it.

## Prepare the release commit

Use `rg` to classify every occurrence of the previous version. Update only
current-version surfaces, including when applicable:

- the versioned installation heading, archive names, URLs, and stable-release
  explanation in `README.md`;
- matching archive and URL fixtures in `tests/test_installer.py`;
- current installer, runtime, and verification documentation;
- release notes derived from the actual diff and verified behavior.

Do not replace historical version references, tags, evidence, or release
records. `scripts/package.py` receives the version as input and must keep its
explicit runtime allowlist unless the release intentionally changes packaged
contents.

Run focused tests after edits. Create one normal release-preparation commit,
usually `chore(release): prepare vMAJOR.MINOR.PATCH`. Stage only reviewed paths,
explicitly inspect the staged file list, and prove `docs/dataset/**` is absent.

## Qualification gate

Run the deterministic local gates with the Python 3.13 runtime environment:

```sh
.venv/bin/python -m unittest discover -s tests -v
sh -n scripts/batch.sh
sh -n scripts/installer.sh
git diff --check
```

Run `tests/test_release_package.py` explicitly when packaging or release
contents changed. Validate every changed skill directory with the installed
skill validator. Inspect the final diff for secrets, personal paths, generated
assets, model changes, dependency drift, weakened tests, and unrelated files.

Require the GitHub CI check to pass on the exact candidate commit before
tagging. CI on Ubuntu x86-64 proves deterministic Python and shell checks only.
Skipped reference-image tests are `external verification`, not inference proof.

Before publication, qualify the same candidate on the target Raspberry Pi with
Python 3.13, the bundled `models/yolo26n.onnx`, and the named reference images.
Verify installation, real inference, output images, and the reference-image
suite without skips. Missing Pi hardware or fixtures may defer tagging and
publication but must never be reported as `PASS`.

Any source change after qualification creates a new candidate and invalidates
earlier CI, Raspberry Pi, and artifact evidence.

## Candidate assets

Generate assets from the exact candidate commit into a new directory outside
the repository:

```sh
.venv/bin/python scripts/package.py "$version" "$commit" "$output_directory"
```

Require exactly the derived archive and checksum names. Verify the checksum,
archive root, sorted member list, required runtime files, regular-file types,
and absence of non-allowlisted source. Extract into a new directory and inspect
the packaged README, installer, model, requirements, and Python modules.

On the Raspberry Pi, run the packaged installer from the extracted archive and
perform one documented CLI smoke inference. A non-aarch64 host may inspect the
archive but cannot qualify installation.

## Tag and publish

When local tag creation is explicitly authorized and all required evidence is
current, create one annotated `vMAJOR.MINOR.PATCH` tag on the qualified commit.
Regenerate both assets from that tag into a new external directory and prove
the tag resolves to the candidate commit and the assets are byte-identical to
the qualified candidate assets.

If a post-tag check fails, stop. Never delete or move the tag; correct the defect
in a later version.

When remote publication is explicitly authorized:

1. require the release commit to be present on the intended remote;
2. push the annotated tag without force;
3. create one GitHub Release from that existing tag;
4. attach only the deterministic archive and its checksum;
5. use release notes derived from the qualified diff;
6. read back the tag, release metadata, and exact asset names;
7. download both assets anonymously and verify the checksum and archive again.

Published tags and assets are immutable. Development on `main` may continue
without changing the identity or evidence of an earlier release.

## Final report

Report the previous and new versions, candidate commit, tag identity, asset
names, staged and committed paths, each gate as `PASS`, `FAIL`, or `external
verification`, mutations actually performed, and remaining publication work.
Do not claim completion for any requested action without current direct
evidence.
