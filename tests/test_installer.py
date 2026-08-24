"""Verify package-local installation with controlled Raspberry Pi commands.

Tests replace ``uname`` and ``python3`` through a temporary ``PATH``, avoid
network access, and assert effects only inside a disposable package layout.
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.package = self.root / "package with spaces"
        self.commands = self.root / "commands"
        (self.package / "scripts").mkdir(parents=True)
        (self.package / "requirements").mkdir()
        self.commands.mkdir()
        shutil.copy2(ROOT / "scripts/installer.sh", self.package / "scripts/installer.sh")
        shutil.copy2(
            ROOT / "requirements/runtime.txt",
            self.package / "requirements/runtime.txt",
        )
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
        *version_info*) exit "${TEST_VERSION_STATUS:-0}" ;;
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
        for name, source in (("dirname", "/usr/bin/dirname"), ("mkdir", "/bin/mkdir")):
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

    def test_missing_or_old_python_fails_before_creating_environment(self):
        (self.commands / "python3").unlink()
        self.assert_failure("error: Python 3.11 or newer is required")
        self._write_executable(self.commands / "python3", "#!/bin/sh\nexit 1\n")
        self.assert_failure("error: Python 3.11 or newer is required")
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


if __name__ == "__main__":
    unittest.main()
