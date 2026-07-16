"""
Verify both optional destinations, statuses, diagnostics, and success timing.

The argument boundary and `process_image` are mocked; their own modules retain
path, inventory, and inference behavior coverage.
"""

import contextlib
import io
import runpy
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from phenocam.arguments import ArgumentValidationError, Arguments
from phenocam.inference.errors import InferenceError, OutputWriteError
from phenocam.__main__ import main
from phenocam.classes.selection import ClassConfigurationError, ModelClassesError


class RunTests(unittest.TestCase):
    def setUp(self):
        self.arguments = Arguments(
            input=Path("input.jpg"),
            annotated_output=Path("annotated.jpg"),
            privacy_output=Path("privacy.jpg"),
            model=Path("model.onnx"),
        )

    def call_main(self, argv=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_passes_paths_and_prints_formatted_duration(self):
        with patch("phenocam.__main__.parse_arguments", return_value=self.arguments), patch(
            "phenocam.__main__.process_image", return_value=1.2344
        ) as process:
            status, stdout, stderr = self.call_main(["ignored"])
        process.assert_called_once_with(
            self.arguments.model,
            self.arguments.input,
            self.arguments.annotated_output,
            self.arguments.privacy_output,
        )
        self.assertEqual(
            (status, stdout, stderr), (0, "Execution time: 1.234 s\n", "")
        )

    def test_each_output_combination_is_delegated_once_in_path_order(self):
        combinations = (
            (Path("annotated.jpg"), None),
            (None, Path("privacy.jpg")),
            (Path("annotated.jpg"), Path("privacy.jpg")),
        )
        for annotated, privacy in combinations:
            arguments = Arguments(
                self.arguments.input, annotated, privacy, self.arguments.model
            )
            with self.subTest(annotated=annotated, privacy=privacy), patch(
                "phenocam.__main__.parse_arguments", return_value=arguments
            ), patch("phenocam.__main__.process_image", return_value=0.5) as process:
                status, stdout, stderr = self.call_main([])
            process.assert_called_once_with(
                arguments.model, arguments.input, annotated, privacy
            )
            self.assertEqual(
                (status, stdout, stderr), (0, "Execution time: 0.500 s\n", "")
            )

    def test_each_argument_error_is_exactly_reported(self):
        messages = [
            "error: input image does not exist or is not a file",
            "error: model does not exist or is not a file",
            "error: model must be an ONNX file",
            "error: at least one output path is required",
            "error: output directory does not exist",
            "error: output path must be a file",
            "error: input and output paths must differ",
            "error: output paths must differ",
        ]
        for message in messages:
            with self.subTest(message=message), patch(
                "phenocam.__main__.parse_arguments", side_effect=ArgumentValidationError(message)
            ), patch("phenocam.__main__.process_image") as process:
                status, stdout, stderr = self.call_main([])
            self.assertEqual((status, stdout, stderr), (1, "", f"{message}\n"))
            process.assert_not_called()

    def test_inference_error_has_fixed_diagnostic(self):
        with patch("phenocam.__main__.parse_arguments", return_value=self.arguments), patch(
            "phenocam.__main__.process_image", side_effect=InferenceError("private detail")
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual((status, stdout, stderr), (1, "", "error: inference failed\n"))
        self.assertNotIn("private detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_class_configuration_error_has_fixed_diagnostic(self):
        error = ClassConfigurationError()
        error.__cause__ = RuntimeError("private configuration detail\nTraceback")
        with patch("phenocam.__main__.parse_arguments", return_value=self.arguments), patch(
            "phenocam.__main__.process_image", side_effect=error
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: class configuration is invalid\n"),
        )
        self.assertNotIn("private configuration detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_model_classes_error_has_fixed_diagnostic(self):
        error = ModelClassesError()
        error.__cause__ = RuntimeError("private model detail\nTraceback")
        with patch("phenocam.__main__.parse_arguments", return_value=self.arguments), patch(
            "phenocam.__main__.process_image", side_effect=error
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: model classes are incompatible\n"),
        )
        self.assertNotIn("private model detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_output_write_error_has_fixed_diagnostic(self):
        with patch("phenocam.__main__.parse_arguments", return_value=self.arguments), patch(
            "phenocam.__main__.process_image", side_effect=OutputWriteError("private detail")
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: output image could not be written\n"),
        )
        self.assertNotIn("private detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_missing_arguments_keep_argparse_status_two_and_stderr_usage(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("phenocam.__main__.process_image") as process, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main([])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        process.assert_not_called()

    def test_unknown_option_keeps_argparse_status_two_and_stderr_usage(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("phenocam.__main__.process_image") as process, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["--unknown"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        process.assert_not_called()

    def test_obsolete_output_option_keeps_argparse_status_two(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = [
            "--input",
            "input.jpg",
            "--model",
            "model.onnx",
            "--output",
            "output.jpg",
        ]
        with patch("phenocam.__main__.process_image") as process, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        process.assert_not_called()

    def test_help_keeps_argparse_status_zero_and_uses_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("phenocam.__main__.process_image") as process, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--input", help_text)
        self.assertIn("--annotated-output", help_text)
        self.assertIn("--privacy-output", help_text)
        self.assertNotIn("--output", help_text)
        self.assertNotIn("Execution time:", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        process.assert_not_called()

    def test_main_guard_converts_return_value_to_process_status(self):
        message = "error: input image does not exist or is not a file"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "phenocam.arguments.parse_arguments", side_effect=ArgumentValidationError(message)
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ), warnings.catch_warnings(), self.assertRaises(SystemExit) as error:
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("phenocam", run_name="__main__")
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), f"{message}\n")


if __name__ == "__main__":
    unittest.main()
