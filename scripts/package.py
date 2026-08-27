#!/usr/bin/env python3
"""Build one deterministic runtime archive from an allowlisted Git commit.

The resolved commit is the only content source, the runtime allowlist is the
security boundary, and the archive/checksum pair are the only final outputs.
"""

import argparse
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
REQUIRED = (
    "README.md",
    "assets/logo.svg",
    "models/yolo26n.onnx",
    "requirements/runtime.txt",
    "scripts/batch.sh",
    "scripts/installer.sh",
)
DOCUMENTATION = ("docs/cli.md", "docs/development.md", "docs/manual.md")


class PackageError(RuntimeError):
    """Report an expected packaging failure without a traceback."""


def run_git(*arguments):
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(
            ["git", *arguments],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        raise PackageError("git is required") from None


def safe_path(path):
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise PackageError("source commit contains an unsafe path")
    return path


def source_files(reference):
    resolved = run_git(
        "rev-parse", "--verify", "--quiet", "--end-of-options", f"{reference}^{{commit}}"
    )
    if resolved.returncode != 0:
        raise PackageError("source commit does not exist")
    commit_lines = resolved.stdout.splitlines()
    if len(commit_lines) != 1:
        raise PackageError("source commit does not exist")
    commit = commit_lines[0].decode("ascii", errors="strict")

    listing = run_git("ls-tree", "-rz", "--full-tree", commit)
    if listing.returncode != 0:
        raise PackageError("source commit could not be read")
    tree = {}
    try:
        for record in listing.stdout.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
            tree[path] = (mode, kind, object_id)
    except (UnicodeError, ValueError):
        raise PackageError("source commit could not be read") from None

    selected = {}
    for path in REQUIRED:
        if path not in tree:
            raise PackageError(f"required source path does not exist: {path}")
        selected[safe_path(path)] = tree[path]
    # Older source commits remain packageable; current commits carry their versioned docs.
    for path in DOCUMENTATION:
        if path in tree:
            selected[safe_path(path)] = tree[path]
    for path, entry in tree.items():
        if path.startswith("phenocam/") and path.endswith(".py"):
            selected[safe_path(path)] = entry

    contents = {}
    for path in sorted(selected, key=lambda item: item.encode("utf-8")):
        mode, kind, object_id = selected[path]
        if kind != "blob" or mode not in ("100644", "100755"):
            raise PackageError(f"source path is not a regular file: {path}")
        blob = run_git("cat-file", "blob", object_id)
        if blob.returncode != 0:
            raise PackageError("source commit could not be read")
        contents[path] = blob.stdout
    return contents


def archive_members(root, contents):
    directories = {root}
    for path in contents:
        parent = PurePosixPath(root, path).parent
        while str(parent) != ".":
            directories.add(parent.as_posix())
            if parent.as_posix() == root:
                break
            parent = parent.parent
    members = [(path, None) for path in directories]
    members.extend((f"{root}/{path}", data) for path, data in contents.items())
    return sorted(members, key=lambda item: item[0].encode("utf-8"))


def write_archive(path, root, contents):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, data in archive_members(root, contents):
                    information = tarfile.TarInfo(name)
                    information.uid = information.gid = 0
                    information.uname = information.gname = ""
                    information.mtime = 0
                    if data is None:
                        information.type = tarfile.DIRTYPE
                        information.mode = 0o755
                        archive.addfile(information)
                    else:
                        information.size = len(data)
                        information.mode = 0o755 if name.endswith(("/batch.sh", "/installer.sh")) else 0o644
                        archive.addfile(information, fileobj=io.BytesIO(data))


def build(version, reference, output_directory):
    if VERSION.fullmatch(version) is None:
        raise PackageError("version must use MAJOR.MINOR.PATCH")
    output = Path(output_directory)
    if not output.is_dir():
        raise PackageError("output directory does not exist")

    archive_name = f"phenocam-vision-edge-{version}.tar.gz"
    archive_path = output / archive_name
    checksum_path = output / f"{archive_name}.sha256"
    if archive_path.exists() or checksum_path.exists():
        raise PackageError("release asset already exists")

    contents = source_files(reference)
    temporary_paths = []
    completed_paths = []
    try:
        descriptor, temporary_name = tempfile.mkstemp(dir=output, prefix=".release-", suffix=".tar.gz")
        os.close(descriptor)
        temporary_archive = Path(temporary_name)
        temporary_paths.append(temporary_archive)
        write_archive(temporary_archive, f"phenocam-vision-edge-{version}", contents)

        with tarfile.open(temporary_archive, "r:gz") as archive:
            expected = [name for name, _data in archive_members(f"phenocam-vision-edge-{version}", contents)]
            if archive.getnames() != expected:
                raise PackageError("release archive validation failed")
        digest = hashlib.sha256(temporary_archive.read_bytes()).hexdigest()
        descriptor, temporary_name = tempfile.mkstemp(dir=output, prefix=".release-", suffix=".sha256")
        temporary_checksum = Path(temporary_name)
        temporary_paths.append(temporary_checksum)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as checksum:
            checksum.write(f"{digest}  {archive_name}\n")

        # Final names stay absent until both temporary assets pass validation.
        if archive_path.exists() or checksum_path.exists():
            raise PackageError("release asset already exists")
        temporary_archive.rename(archive_path)
        temporary_paths.remove(temporary_archive)
        completed_paths.append(archive_path)
        temporary_checksum.rename(checksum_path)
        temporary_paths.remove(temporary_checksum)
        completed_paths.append(checksum_path)
    except PackageError:
        raise
    except (OSError, tarfile.TarError):
        raise PackageError("release assets could not be created") from None
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except OSError:
                pass
        if len(completed_paths) != 2:
            for path in completed_paths:
                try:
                    path.unlink()
                except OSError:
                    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a deterministic runtime release package.")
    parser.add_argument("version")
    parser.add_argument("commit")
    parser.add_argument("output_directory")
    arguments = parser.parse_args(argv)
    try:
        build(arguments.version, arguments.commit, arguments.output_directory)
    except PackageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
