"""
Verify two optional output paths at the public CLI boundary.

Temporary entries prove the at-least-one and pairwise identity invariants for
untrusted paths without modifying production files.
"""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phenocam.arguments import ArgumentValidationError, Arguments, parse_arguments


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input = self.root / "input.jpg"
        self.model = self.root / "model.onnx"
        self.output = self.root / "output.jpg"
        self.input.write_bytes(b"image")
        self.model.write_bytes(b"model")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def argv(self, output=None, model=None, output_option="--annotated-output"):
        return [
            "--input",
            str(self.input),
            output_option,
            str(output or self.output),
            "--model",
            str(model or self.model),
        ]

    def assert_validation_error(self, expected, argv):
        with self.assertRaisesRegex(ArgumentValidationError, f"^{expected}$"):
            parse_arguments(argv)

    def test_valid_options_are_returned_as_immutable_paths(self):
        arguments = parse_arguments(
            [
                "--model",
                str(self.model),
                "--input",
                str(self.input),
                "--annotated-output",
                str(self.output),
            ]
        )

        self.assertEqual(
            arguments, Arguments(self.input, self.output, None, self.model)
        )
        with self.assertRaises(AttributeError):
            arguments.input = self.output

    def test_privacy_only_and_dual_output_forms_are_accepted(self):
        privacy = self.root / "privacy.png"
        privacy_only = parse_arguments(
            self.argv(output=privacy, output_option="--privacy-output")
        )
        dual = parse_arguments(
            [*self.argv(), "--privacy-output", str(privacy)]
        )

        self.assertEqual(
            privacy_only, Arguments(self.input, None, privacy, self.model)
        )
        self.assertEqual(dual, Arguments(self.input, self.output, privacy, self.model))

    def test_repeated_option_uses_last_value(self):
        other_input = self.root / "other.jpg"
        other_input.write_bytes(b"other")
        arguments = parse_arguments(
            ["--input", str(self.input), *self.argv(), "--input", str(other_input)]
        )
        self.assertEqual(arguments.input, other_input)

    def test_missing_required_option_exits_with_usage(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments(["--model", str(self.model), *self.argv()[2:]])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_missing_outputs_has_fixed_semantic_error(self):
        self.assert_validation_error(
            "error: at least one output path is required",
            ["--input", str(self.input), "--model", str(self.model)],
        )

    def test_obsolete_output_option_exits_with_usage(self):
        stderr = io.StringIO()
        argv = [
            "--input",
            str(self.input),
            "--output",
            str(self.output),
            "--model",
            str(self.model),
        ]
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_missing_option_value_exits_with_usage(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments([*self.argv(), "--input"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_unknown_option_exits_with_usage(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments([*self.argv(), "--unknown"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_help_exits_zero_on_stdout_without_validation(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
            parse_arguments(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--input", stdout.getvalue())

    def test_missing_input_has_fixed_error(self):
        self.input.unlink()
        self.assert_validation_error(
            "error: input image does not exist or is not a file", self.argv()
        )

    def test_input_directory_has_fixed_error(self):
        self.assert_validation_error(
            "error: input image does not exist or is not a file",
            ["--input", str(self.root), *self.argv()[2:]],
        )

    def test_missing_model_has_fixed_error(self):
        self.model.unlink()
        self.assert_validation_error("error: model does not exist or is not a file", self.argv())

    def test_non_onnx_model_has_fixed_error(self):
        model = self.root / "model.pt"
        model.write_bytes(b"model")
        self.assert_validation_error("error: model must be an ONNX file", self.argv(model=model))

    def test_onnx_suffix_is_case_insensitive(self):
        model = self.root / "model.ONNX"
        model.write_bytes(b"model")
        self.assertEqual(parse_arguments(self.argv(model=model)).model, model)

    def test_missing_output_parent_has_fixed_error_and_is_not_created(self):
        parent = self.root / "missing"
        output = parent / "output.jpg"
        self.assert_validation_error(
            "error: output directory does not exist", self.argv(output=output)
        )
        self.assertFalse(parent.exists())

    def test_missing_privacy_parent_has_fixed_error_and_is_not_created(self):
        parent = self.root / "missing"
        privacy = parent / "privacy.jpg"
        self.assert_validation_error(
            "error: output directory does not exist",
            self.argv(output=privacy, output_option="--privacy-output"),
        )
        self.assertFalse(parent.exists())

    def test_output_directory_has_fixed_error(self):
        self.assert_validation_error("error: output path must be a file", self.argv(output=self.root))

    def test_privacy_directory_has_fixed_error(self):
        self.assert_validation_error(
            "error: output path must be a file",
            self.argv(output=self.root, output_option="--privacy-output"),
        )

    def test_same_normalized_path_has_fixed_error(self):
        output = self.input.parent / "." / self.input.name
        self.assert_validation_error(
            "error: input and output paths must differ", self.argv(output=output)
        )

    def test_privacy_same_normalized_path_has_fixed_error(self):
        output = self.input.parent / "." / self.input.name
        self.assert_validation_error(
            "error: input and output paths must differ",
            self.argv(output=output, output_option="--privacy-output"),
        )

    def test_normalized_output_identity_has_fixed_error(self):
        (self.root / "nested").mkdir()
        privacy = self.root / "nested" / ".." / self.output.name
        self.assert_validation_error(
            "error: output paths must differ",
            [*self.argv(), "--privacy-output", str(privacy)],
        )

    def test_existing_output_file_is_accepted_without_modification(self):
        self.output.write_bytes(b"existing")
        arguments = parse_arguments(self.argv())
        self.assertEqual(arguments.annotated_output, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing")

    def test_bare_output_uses_current_directory_without_creating_file(self):
        previous_directory = Path.cwd()
        try:
            os.chdir(self.root)
            arguments = parse_arguments(self.argv(output=Path("bare.jpg")))
        finally:
            os.chdir(previous_directory)
        self.assertEqual(arguments.annotated_output, Path("bare.jpg"))
        self.assertFalse((self.root / "bare.jpg").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_output_symlink_to_input_is_rejected(self):
        try:
            self.output.symlink_to(self.input)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")
        self.assert_validation_error(
            "error: input and output paths must differ", self.argv()
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_privacy_symlink_to_input_is_rejected(self):
        try:
            self.output.symlink_to(self.input)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")
        self.assert_validation_error(
            "error: input and output paths must differ",
            self.argv(output_option="--privacy-output"),
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_output_symlinks_to_same_file_are_rejected(self):
        target = self.root / "target.png"
        privacy = self.root / "privacy.png"
        target.write_bytes(b"image")
        try:
            self.output.symlink_to(target)
            privacy.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")
        self.assert_validation_error(
            "error: output paths must differ",
            [*self.argv(), "--privacy-output", str(privacy)],
        )

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_output_hard_links_to_same_file_are_rejected(self):
        privacy = self.root / "privacy.png"
        self.output.write_bytes(b"image")
        try:
            os.link(self.output, privacy)
        except OSError as error:
            self.skipTest(f"hard-link creation is unavailable: {error.errno}")
        self.assert_validation_error(
            "error: output paths must differ",
            [*self.argv(), "--privacy-output", str(privacy)],
        )

    def test_distinct_existing_outputs_are_accepted_without_modification(self):
        privacy = self.root / "privacy.png"
        self.output.write_bytes(b"annotated")
        privacy.write_bytes(b"privacy")
        arguments = parse_arguments(
            [*self.argv(), "--privacy-output", str(privacy)]
        )
        self.assertEqual(arguments.annotated_output, self.output)
        self.assertEqual(arguments.privacy_output, privacy)
        self.assertEqual(self.output.read_bytes(), b"annotated")
        self.assertEqual(privacy.read_bytes(), b"privacy")

    def test_samefile_os_error_is_sanitized_as_non_identity(self):
        self.output.write_bytes(b"existing")
        with patch.object(Path, "samefile", side_effect=OSError("private detail")):
            arguments = parse_arguments(self.argv())
        self.assertEqual(arguments.annotated_output, self.output)

    def test_annotated_validation_precedes_privacy_validation(self):
        annotated = self.root / "missing-annotated" / "out.png"
        privacy = self.root / "missing-privacy" / "out.png"
        self.assert_validation_error(
            "error: output directory does not exist",
            self.argv(output=annotated) + ["--privacy-output", str(privacy)],
        )

    def test_validation_order_is_deterministic(self):
        self.input.unlink()
        self.model.unlink()
        self.assert_validation_error(
            "error: input image does not exist or is not a file", self.argv()
        )


if __name__ == "__main__":
    unittest.main()
