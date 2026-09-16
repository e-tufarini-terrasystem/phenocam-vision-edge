"""Verify deterministic release packaging in disposable Git repositories.

Synthetic commits exercise allowlisting and path safety without reading the
working repository, accessing a network, or consulting the real remote.
"""

import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts/package.py"


class ReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository with spaces"
        self.output = self.root / "output"
        self.repository.mkdir()
        self.output.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release@example.invalid")
        files = {
            "README.md": b"committed readme\n",
            "assets/logo.svg": b"<svg/>\n",
            "docs/cli.md": b"software operation guide\n",
            "docs/development.md": b"development guide\n",
            "docs/manual.md": b"operator manual\n",
            "phenocam/__init__.py": b'"""Package."""\n',
            "phenocam/classes/naïve.py": b"VALUE = 1\n",
            "models/yolo26n-phenocam.onnx": b"model bytes\x00",
            "requirements/runtime.txt": b"Pillow==12.3.0\n",
            "scripts/batch.sh": b"#!/bin/sh\nexit 0\n",
            "scripts/installer.sh": b"#!/bin/sh\nexit 0\n",
            "models/yolo26n.onnx": b"forbidden baseline model\n",
            "models/yolo26n.pt": b"forbidden checkpoint\n",
            "tests/test_example.py": b"forbidden test\n",
        }
        for name, data in files.items():
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.git("add", ".")
        self.commit("initial")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def git(self, *arguments, check=True, input_bytes=None):
        return subprocess.run(
            ["git", *arguments],
            cwd=self.repository,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def commit(self, message):
        environment = {
            **os.environ,
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=self.repository,
            env=environment,
            check=True,
        )

    def run_packager(self, version="0.1.0", reference="HEAD", output=None, env=None):
        return subprocess.run(
            [sys.executable, str(PACKAGER), version, reference, str(output or self.output)],
            cwd=self.repository,
            env=env,
            capture_output=True,
            text=True,
        )

    def asset_paths(self, output=None):
        destination = output or self.output
        archive = destination / "phenocam-vision-edge-0.1.0.tar.gz"
        return archive, destination / f"{archive.name}.sha256"

    def test_success_uses_only_normalized_allowlisted_commit_files(self):
        (self.repository / "README.md").write_text("working tree change\n", encoding="utf-8")
        (self.repository / "phenocam/untracked.py").write_text("UNTRACKED = True\n", encoding="utf-8")
        result = self.run_packager()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((result.stdout, result.stderr), ("", ""))
        archive_path, checksum_path = self.asset_paths()
        self.assertTrue(archive_path.is_file())
        self.assertTrue(checksum_path.is_file())

        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
            root = "phenocam-vision-edge-0.1.0"
            expected_files = {
                f"{root}/README.md",
                f"{root}/assets/logo.svg",
                f"{root}/docs/cli.md",
                f"{root}/docs/development.md",
                f"{root}/docs/manual.md",
                f"{root}/models/yolo26n-phenocam.onnx",
                f"{root}/phenocam/__init__.py",
                f"{root}/phenocam/classes/naïve.py",
                f"{root}/requirements/runtime.txt",
                f"{root}/scripts/batch.sh",
                f"{root}/scripts/installer.sh",
            }
            self.assertEqual({member.name for member in archive if member.isfile()}, expected_files)
            self.assertTrue(all(name == root or name.startswith(f"{root}/") for name in names))
            self.assertNotIn(f"{root}/models/yolo26n.onnx", names)
            self.assertNotIn(f"{root}/models/yolo26n.pt", names)
            self.assertNotIn(f"{root}/tests/test_example.py", names)
            self.assertNotIn(f"{root}/phenocam/untracked.py", names)
            self.assertEqual(archive.extractfile(f"{root}/README.md").read(), b"committed readme\n")
            for member in archive:
                self.assertEqual((member.uid, member.gid, member.uname, member.gname, member.mtime), (0, 0, "", "", 0))
                self.assertEqual(member.mode, 0o755 if member.isdir() or member.name.endswith(".sh") else 0o644)

        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        self.assertEqual(checksum_path.read_text(encoding="ascii"), f"{digest}  {archive_path.name}\n")

    def test_two_builds_are_byte_identical(self):
        second_output = self.root / "second output"
        second_output.mkdir()
        self.assertEqual(self.run_packager().returncode, 0)
        self.assertEqual(self.run_packager(output=second_output).returncode, 0)
        first = self.asset_paths()
        second = self.asset_paths(second_output)
        self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
        self.assertEqual(first[1].read_bytes(), second[1].read_bytes())

    def test_missing_required_file_identifies_path_and_leaves_no_assets(self):
        (self.repository / "assets/logo.svg").unlink()
        self.git("add", "-u")
        self.commit("remove logo")
        result = self.run_packager()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "error: required source path does not exist: assets/logo.svg\n")
        self.assertEqual(tuple(self.output.iterdir()), ())

    def test_tracked_python_symlink_is_rejected(self):
        link = self.repository / "phenocam/link.py"
        link.symlink_to("__init__.py")
        self.git("add", "phenocam/link.py")
        self.commit("add symlink")
        result = self.run_packager()
        self.assertEqual(result.returncode, 1)
        self.assertIn("source path is not a regular file: phenocam/link.py", result.stderr)
        self.assertEqual(tuple(self.output.iterdir()), ())

    def test_unsafe_archive_paths_are_rejected(self):
        specification = importlib.util.spec_from_file_location("release_package", PACKAGER)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        for path in ("/absolute.py", "phenocam/../escape.py"):
            with self.subTest(path=path), self.assertRaisesRegex(
                module.PackageError,
                "source commit contains an unsafe path",
            ):
                module.safe_path(path)

    def test_invalid_versions_have_exact_error_and_no_output(self):
        for version in ("v0.1.0", "01.2.3", "1.2", "1.2.3-rc1", " 1.2.3"):
            with self.subTest(version=version):
                result = self.run_packager(version=version)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, "error: version must use MAJOR.MINOR.PATCH\n")
                self.assertEqual(tuple(self.output.iterdir()), ())

    def test_invalid_reference_has_exact_error(self):
        result = self.run_packager(reference="missing")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "error: source commit does not exist\n")
        self.assertEqual(tuple(self.output.iterdir()), ())

    def test_existing_asset_is_never_modified(self):
        archive, checksum = self.asset_paths()
        for existing in (archive, checksum):
            with self.subTest(existing=existing.name):
                existing.write_bytes(b"keep")
                result = self.run_packager()
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, "error: release asset already exists\n")
                self.assertEqual(existing.read_bytes(), b"keep")
                for path in self.output.iterdir():
                    path.unlink()

    def test_missing_output_directory_has_exact_error(self):
        missing = self.root / "missing"
        result = self.run_packager(output=missing)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "error: output directory does not exist\n")
        self.assertFalse(missing.exists())

    def test_missing_or_unexecutable_git_has_exact_error(self):
        unavailable = self.root / "unavailable git"
        unavailable.mkdir()
        (unavailable / "git").write_text("not executable\n", encoding="utf-8")
        for path in ("", str(unavailable)):
            with self.subTest(path=path):
                result = self.run_packager(env={"PATH": path})
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, "error: git is required\n")
                self.assertEqual(tuple(self.output.iterdir()), ())

    def test_blob_read_failure_is_sanitized_and_cleaned(self):
        command_directory = self.root / "failing git"
        command_directory.mkdir()
        wrapper = command_directory / "git"
        real_git = shutil.which("git")
        wrapper.write_text(
            "#!/bin/sh\n"
            "[ \"$1\" = \"cat-file\" ] && exit 1\n"
            f'exec "{real_git}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        environment = {**os.environ, "PATH": f"{command_directory}{os.pathsep}{os.environ['PATH']}"}
        result = self.run_packager(env=environment)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "error: source commit could not be read\n")
        self.assertEqual(tuple(self.output.iterdir()), ())

    def test_command_line_syntax_error_returns_two(self):
        result = subprocess.run(
            [sys.executable, str(PACKAGER)],
            cwd=self.repository,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
