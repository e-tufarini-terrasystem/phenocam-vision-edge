---
name: github-release
description: Prepare and publish one guarded phenocam-vision-edge GitHub release when explicitly invoked through @.skills/github-release/SKILL.md with a stable release version and optional model version. Do not use for general packaging or unattended publication.
---

<!--
This skill owns release preflight, local preparation, explicit authorization,
publication, and final verification. Deterministic runtime packaging belongs to
scripts/package.py.
-->

# GitHub Release

Release only `e-tufarini-terrasystem/phenocam-vision-edge`. Complete every
read-only preflight and show one full mutation plan before requesting
authorization. Never print credentials or read credential files.

## Inputs

- Require `release_version` in stable `MAJOR.MINOR.PATCH` form. Accept each
  numeric component only as `0` or a non-zero digit followed by digits.
- Accept an optional `model_version` with the same syntax. When omitted, read
  and preserve the current `MODEL_VERSION` and its test expectations.
- Do not add `v`, infer a version, or accept whitespace, prerelease identifiers,
  build metadata, or leading zeroes.
- Derive the tag, release title, and annotation as `v<release_version>`,
  `v<release_version>`, and `Release v<release_version>`.

Stop without mutation when either supplied version is invalid.

## Read-only preflight

Run all checks before editing tracked files, committing, tagging, packaging,
pushing, creating a release, or uploading an asset. A failed or inconclusive
check stops the workflow and identifies the invariant that failed.

1. Require an empty result from `git status --porcelain=v1 --untracked-files=all`
   and require `git branch --show-current` to be exactly `main`.
2. Read `remote.origin.url`. Accept the exact HTTPS URL
   `https://github.com/e-tufarini-terrasystem/phenocam-vision-edge.git` or an SSH
   URL that names exactly GitHub owner `e-tufarini-terrasystem` and repository
   `phenocam-vision-edge`; reject other hosts, owners, or repositories.
3. Refresh only the remote main ref with tags disabled, using
   `git fetch --no-tags origin refs/heads/main:refs/remotes/origin/main`. Require
   `git rev-parse main` and `git rev-parse origin/main` to return the same commit.
4. Require `git`, `python3`, and `gh`. Verify GitHub authentication with
   `gh auth status --hostname github.com` and repository readability with
   `gh repo view e-tufarini-terrasystem/phenocam-vision-edge`. Do not display or
   inspect authentication tokens.
5. Require the target tag to be absent from `refs/tags/` locally. Query the
   refreshed remote explicitly and require the tag to be absent there; treat a
   remote query error as a failure, not as absence.
6. Query GitHub for the release by exact tag. Continue only when the API
   conclusively reports it absent; authentication, network, permission, or
   unexpected API failures stop the workflow.
7. Read the current release surfaces in `phenocam/metadata.py`,
   `tests/test_metadata.py`, and `README.md`. Determine whether the requested
   software version and optional model version require changes. The only
   approved changed files are these three paths.

Display a summary containing:

- requested release version and effective model version;
- current `main` commit and whether the tag will target it or a new preparation
  commit;
- exact files that would change and whether a preparation commit is needed;
- tag, archive, and checksum names;
- exact GitHub repository;
- planned local mutations, atomic ref push, release creation, and two asset
  uploads.

If every release surface already matches, explicitly state that no empty commit
will be created and that the current clean `main` will be tagged.

## Authorization boundary

After the summary, request exactly:

`CONFIRM RELEASE v<release_version>`

Proceed only when the operator returns that exact text. Every other response,
including different case, spacing, punctuation, or surrounding text, stops the
workflow without mutation. Do not request earlier or additional confirmations.

## Local preparation

After authorization, update only the approved release surfaces when needed:

- set `SOFTWARE_VERSION` and its metadata test expectations to
  `release_version`;
- set `MODEL_VERSION` and its metadata test expectations only when
  `model_version` was supplied;
- update the README metadata example and its versioned release heading, URLs,
  filenames, directory, and commands to the effective versions.

Reject any changed path outside `phenocam/metadata.py`, `tests/test_metadata.py`,
and `README.md`. Run these pre-tag checks from the repository root:

```sh
sh -n scripts/installer.sh
.venv/bin/python -m unittest discover -s tests -v
python3 /Users/emanuele/.codex/skills/.system/skill-creator/scripts/quick_validate.py .skills/github-release
git diff --check
```

If a check fails, stop and report all retained local changes. If changes were
needed and checks pass, commit exactly those approved paths as
`chore(release): prepare v<release_version>`. If no changes were needed, create
no commit. In either case require a clean tree, then create the annotated tag
with `git tag -a v<release_version> -m "Release v<release_version>"`.

## Package validation

Create two distinct task-specific directories with `mktemp -d`. Build from the
new tag, never from the working tree, with:

```sh
python3 .skills/github-release/scripts/package.py <release_version> v<release_version> <first-directory>
python3 .skills/github-release/scripts/package.py <release_version> v<release_version> <second-directory>
```

Compare the two archives byte-for-byte and compare the two checksum files
byte-for-byte. Inspect one archive member list. Require one versioned root and
exactly `README.md`, `assets/logo.svg`, all tracked regular `phenocam/**/*.py`
files from the tag, `models/yolo26n.onnx`, `requirements/runtime.txt`,
`scripts/batch.sh`, and `scripts/installer.sh`, plus their parent directories.
Reject every other member, absolute path, `..` component, non-regular selected
file, or symlink.

Any failure before push stops the workflow. Retain and report the preparation
commit, annotated tag, local changes, and both package directories that exist;
do not roll them back automatically.

## Publication

Resolve the tag target and require it to equal the intended clean `main` commit.
Then publish in this order:

1. Push `main` and the annotated tag together with one atomic push:
   `git push --atomic origin main v<release_version>`.
2. If and only if the atomic push succeeds, create a normal GitHub release in
   `e-tufarini-terrasystem/phenocam-vision-edge`, titled `v<release_version>`,
   with generated notes and exactly the archive and checksum from the first
   package directory. Do not use draft or prerelease mode.

If the atomic push fails, create no GitHub release. If release creation or asset
upload fails after the push, do not delete, move, replace, force-push, retag,
retry destructively, or attempt rollback. Query and report the observed remote
commit, tag, release, and asset state; repair or resume is a separate task.

## Final verification

Declare success only after read-only checks prove all of the following:

- remote tag `v<release_version>` targets the intended release commit;
- the release exists, is titled `v<release_version>`, and is neither draft nor
  prerelease;
- its asset names are exactly the versioned `.tar.gz` and matching `.sha256`.

Report the release and model versions, commit, tag, release URL, and both asset
names. Only after success, verify that each cleanup target is the non-empty
task-specific directory returned by `mktemp -d`, then remove those two local
temporary directories. Never remove or overwrite remote state during cleanup.
