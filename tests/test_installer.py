"""Verify package-local and documented release installation paths.

Tests replace ``uname`` and ``python3`` through a temporary ``PATH``, avoid
network access, and assert effects only inside disposable package layouts. The
same command environment exercises the exact README bootstrap sequence.
"""

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts/package.py"
ARCHIVE_NAME = "phenocam-vision-edge-0.2.0.tar.gz"
CHECKSUM_NAME = f"{ARCHIVE_NAME}.sha256"
ARCHIVE_URL = (
    "https://github.com/e-tufarini-terrasystem/phenocam-vision-edge/"
    f"releases/download/v0.2.0/{ARCHIVE_NAME}"
)
CHECKSUM_URL = f"{ARCHIVE_URL}.sha256"


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.package = self.root / "package with spaces"
        self.commands = self.root / "commands"
        (self.package / "scripts").mkdir(parents=True)
        (self.package / "requirements").mkdir()
        (self.package / "models").mkdir()
        self.commands.mkdir()
        shutil.copy2(ROOT / "scripts/installer.sh", self.package / "scripts/installer.sh")
        shutil.copy2(
            ROOT / "requirements/runtime.txt",
            self.package / "requirements/runtime.txt",
        )
        (self.package / "models/yolo26n-phenocam.onnx").write_bytes(b"model")
        self.pip_log = self.root / "pip.log"
        self.venv_python = self.root / "venv-python"
        self._write_executable(
            self.commands / "uname",
            '#!/bin/sh\nprintf "%s\\n" "${TEST_ARCHITECTURE:-aarch64}"\n',
        )
        self._write_executable(
            self.venv_python,
            """#!/bin/sh
: > "$TEST_PIP_LOG"
for argument do
    printf '%s\\n' "$argument" >> "$TEST_PIP_LOG"
done
exit "${TEST_PIP_STATUS:-0}"
""",
        )
        self._write_executable(
            self.commands / "python3",
            """#!/bin/sh
if [ "$1" = "-c" ]; then
    case "$2" in
        *version_info*)
            [ "$2" = 'import sys; raise SystemExit(sys.version_info[:2] != (3, 13))' ] || exit 1
            exit "${TEST_VERSION_STATUS:-0}"
            ;;
        *'import venv'*) exit "${TEST_VENV_IMPORT_STATUS:-0}" ;;
    esac
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    [ "${TEST_VENV_CREATE_STATUS:-0}" -eq 0 ] || exit "$TEST_VENV_CREATE_STATUS"
    /bin/mkdir -p "$3/bin" || exit 1
    /bin/cp "$TEST_VENV_PYTHON" "$3/bin/python" || exit 1
    /bin/chmod +x "$3/bin/python" || exit 1
    exit 0
fi
exit 1
""",
        )
        for name in ("curl", "dirname", "gzip", "mkdir", "sha256sum", "tar"):
            source = shutil.which(name)
            self.assertIsNotNone(source)
            (self.commands / name).symlink_to(source)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_executable(self, path, content):
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_installer(self, **overrides):
        environment = {
            "PATH": str(self.commands),
            "TEST_PIP_LOG": str(self.pip_log),
            "TEST_VENV_PYTHON": str(self.venv_python),
            **overrides,
        }
        return subprocess.run(
            ["/bin/sh", str(self.package / "scripts/installer.sh")],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
        )

    def assert_failure(self, message, **environment):
        result = self.run_installer(**environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, f"{message}\n")
        self.assertNotIn("Traceback", result.stderr)
        return result

    def documented_command(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### Versioned installation (v0.2.0)", 1)[1]
        section = re.split(r"\n### |\n## ", section, maxsplit=1)[0]
        blocks = re.findall(r"```sh\n(.*?)```", section, flags=re.DOTALL)
        self.assertEqual(len(blocks), 1)
        command_tokens = blocks[0].split()
        self.assertEqual(command_tokens.count(ARCHIVE_URL), 1)
        self.assertEqual(command_tokens.count(CHECKSUM_URL), 1)
        return blocks[0]

    def release_assets(self):
        assets = self.root / "release assets"
        source = self.root / "release source"
        assets.mkdir()
        for relative in (
            "README.md",
            "assets/logo.svg",
            "docs/cli.md",
            "docs/development.md",
            "docs/manual.md",
            "models/yolo26n-phenocam.onnx",
            "models/yolo26n-phenocam.json",
            "requirements/runtime.txt",
            "scripts/batch.sh",
            "scripts/installer.sh",
        ):
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        shutil.copytree(ROOT / "phenocam", source / "phenocam")
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(["git", "add", "."], cwd=source, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Release Test",
                "-c", "user.email=release@example.invalid",
                "commit", "-q", "-m", "release source",
            ],
            cwd=source,
            check=True,
        )
        result = subprocess.run(
            ["python3", str(PACKAGER), "0.2.0", "HEAD", str(assets)],
            cwd=source,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return assets / ARCHIVE_NAME, assets / CHECKSUM_NAME

    def run_documented_command(self, archive_url, checksum_url, **overrides):
        command = self.documented_command()
        command = command.replace(CHECKSUM_URL, checksum_url)
        command = command.replace(ARCHIVE_URL, archive_url)
        destination = self.root / "bootstrap"
        destination.mkdir(exist_ok=True)
        environment = {
            "PATH": str(self.commands),
            "TEST_PIP_LOG": str(self.pip_log),
            "TEST_VENV_PYTHON": str(self.venv_python),
            **overrides,
        }
        return subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=destination,
            env=environment,
            capture_output=True,
            text=True,
        ), destination

    def test_success_creates_only_package_local_runtime_state(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "Installation complete\n")
        self.assertEqual(result.stderr, "")
        self.assertTrue(os.access(self.package / ".venv/bin/python", os.X_OK))
        self.assertTrue((self.package / "input").is_dir())
        self.assertTrue((self.package / "output").is_dir())
        self.assertEqual(
            self.pip_log.read_text(encoding="utf-8").splitlines(),
            [
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "-r",
                str(self.package / "requirements/runtime.txt"),
            ],
        )

    def test_wrong_architecture_fails_before_creating_environment(self):
        self.assert_failure(
            "error: Raspberry Pi aarch64 is required",
            TEST_ARCHITECTURE="x86_64",
        )
        self.assertFalse((self.package / ".venv").exists())

    def test_missing_or_unsupported_python_fails_before_creating_environment(self):
        (self.commands / "python3").unlink()
        self.assert_failure("error: Python 3.13 is required")
        self._write_executable(self.commands / "python3", "#!/bin/sh\nexit 1\n")
        self.assert_failure("error: Python 3.13 is required")
        self.assertFalse((self.package / ".venv").exists())

    def test_missing_venv_module_fails_before_creating_environment(self):
        self.assert_failure(
            "error: python3-venv is required",
            TEST_VENV_IMPORT_STATUS="1",
        )
        self.assertFalse((self.package / ".venv").exists())

    def test_missing_requirements_fails_before_creating_environment(self):
        (self.package / "requirements/runtime.txt").unlink()
        self.assert_failure("error: runtime requirements do not exist")
        self.assertFalse((self.package / ".venv").exists())

    def test_symlinked_requirements_are_rejected(self):
        requirements = self.package / "requirements/runtime.txt"
        requirements.unlink()
        requirements.symlink_to(self.root / "outside-requirements.txt")
        self.assert_failure("error: runtime requirements do not exist")
        self.assertFalse((self.package / ".venv").exists())

    def test_missing_or_symlinked_runtime_model_is_rejected(self):
        model = self.package / "models/yolo26n-phenocam.onnx"
        model.unlink()
        self.assert_failure("error: runtime model does not exist")
        model.symlink_to(self.root / "outside-model.onnx")
        self.assert_failure("error: runtime model does not exist")
        self.assertFalse((self.package / ".venv").exists())

    def test_preexisting_environment_is_not_modified(self):
        environment = self.package / ".venv"
        environment.mkdir()
        marker = environment / "marker"
        marker.write_text("keep", encoding="utf-8")
        self.assert_failure("error: virtual environment already exists")
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_environment_creation_failure_is_sanitized(self):
        self.assert_failure(
            "error: virtual environment could not be created",
            TEST_VENV_CREATE_STATUS="1",
        )

    def test_dependency_failure_retains_partial_environment(self):
        self.assert_failure(
            "error: runtime dependencies could not be installed",
            TEST_PIP_STATUS="1",
        )
        self.assertTrue((self.package / ".venv").is_dir())
        self.assertFalse((self.package / "input").exists())
        self.assertFalse((self.package / "output").exists())

    def test_input_directory_failure_is_sanitized(self):
        (self.package / "input").write_text("blocking file", encoding="utf-8")
        self.assert_failure("error: input directory could not be created")
        self.assertFalse((self.package / "output").exists())

    def test_output_directory_failure_retains_completed_input(self):
        (self.package / "output").write_text("blocking file", encoding="utf-8")
        self.assert_failure("error: output directory could not be created")
        self.assertTrue((self.package / "input").is_dir())

    def test_installer_contains_no_privileged_package_manager_calls(self):
        installer = (self.package / "scripts/installer.sh").read_text(encoding="utf-8")
        for command in ("sudo", "apt", "apt-get"):
            self.assertNotIn(command, installer.split())

    def test_documented_command_installs_local_release_assets(self):
        archive, checksum = self.release_assets()
        result, destination = self.run_documented_command(
            archive.as_uri(),
            checksum.as_uri(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = destination / "phenocam-vision-edge-0.2.0"
        self.assertTrue((installed / "README.md").is_file())
        self.assertTrue((installed / "models/yolo26n-phenocam.onnx").is_file())
        self.assertTrue(os.access(installed / ".venv/bin/python", os.X_OK))
        self.assertTrue((installed / "input").is_dir())
        self.assertTrue((installed / "output").is_dir())
        self.assertTrue((destination / ARCHIVE_NAME).is_file())
        self.assertTrue((destination / CHECKSUM_NAME).is_file())

    def test_documented_command_preserves_existing_destination(self):
        archive, checksum = self.release_assets()
        destination = self.root / "bootstrap/phenocam-vision-edge-0.2.0"
        destination.mkdir(parents=True)
        marker = destination / "marker"
        marker.write_text("keep", encoding="utf-8")
        result, bootstrap = self.run_documented_command(
            archive.as_uri(),
            checksum.as_uri(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((bootstrap / ARCHIVE_NAME).exists())
        self.assertFalse(self.pip_log.exists())

    def test_documented_command_stops_after_archive_download_failure(self):
        _archive, checksum = self.release_assets()
        missing = (self.root / "missing archive").as_uri()
        result, destination = self.run_documented_command(missing, checksum.as_uri())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((destination / CHECKSUM_NAME).exists())
        self.assertFalse((destination / "phenocam-vision-edge-0.2.0").exists())
        self.assertFalse(self.pip_log.exists())

    def test_documented_command_stops_after_checksum_download_failure(self):
        archive, _checksum = self.release_assets()
        missing = (self.root / "missing checksum").as_uri()
        result, destination = self.run_documented_command(archive.as_uri(), missing)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((destination / ARCHIVE_NAME).is_file())
        self.assertFalse((destination / "phenocam-vision-edge-0.2.0").exists())
        self.assertFalse(self.pip_log.exists())

    def test_documented_command_stops_after_checksum_mismatch(self):
        archive, checksum = self.release_assets()
        checksum.write_text(f"{'0' * 64}  {ARCHIVE_NAME}\n", encoding="ascii")
        result, destination = self.run_documented_command(
            archive.as_uri(),
            checksum.as_uri(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((destination / "phenocam-vision-edge-0.2.0").exists())
        self.assertFalse(self.pip_log.exists())

    def test_documented_command_stops_after_invalid_archive(self):
        assets = self.root / "invalid assets"
        assets.mkdir()
        archive = assets / ARCHIVE_NAME
        archive.write_bytes(b"not a tar archive")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = assets / CHECKSUM_NAME
        checksum.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="ascii")
        result, _destination = self.run_documented_command(
            archive.as_uri(),
            checksum.as_uri(),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.pip_log.exists())

    def test_documented_command_retains_package_after_installer_failure(self):
        archive, checksum = self.release_assets()
        result, destination = self.run_documented_command(
            archive.as_uri(),
            checksum.as_uri(),
            TEST_PIP_STATUS="1",
        )
        self.assertNotEqual(result.returncode, 0)
        installed = destination / "phenocam-vision-edge-0.2.0"
        self.assertTrue((installed / ".venv").is_dir())
        self.assertFalse((installed / "input").exists())
        self.assertFalse((installed / "output").exists())
        self.assertTrue(self.pip_log.is_file())


if __name__ == "__main__":
    unittest.main()
