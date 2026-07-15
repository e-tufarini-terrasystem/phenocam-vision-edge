"""
Verify process statuses, streams, configuration errors, and success timing.

Argument and inference boundaries are mocked. Inventory validation and
Ultralytics behavior remain covered by their own modules.
"""

import contextlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from arguments import ArgumentValidationError, Arguments
from inference import GammaConfigurationError, InferenceError, OutputWriteError
from run import main
from selection import ClassConfigurationError, ModelClassesError


class RunTests(unittest.TestCase):
    def setUp(self):
        self.arguments = Arguments(
            input=Path("input.jpg"),
            output=Path("output.jpg"),
            model=Path("model.onnx"),
        )

    def call_main(self, argv=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_passes_paths_and_prints_formatted_duration(self):
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", return_value=1.2344
        ) as annotate:
            status, stdout, stderr = self.call_main(["ignored"])
        annotate.assert_called_once_with(
            self.arguments.model, self.arguments.input, self.arguments.output
        )
        self.assertEqual(
            (status, stdout, stderr), (0, "Execution time: 1.234 s\n", "")
        )

    def test_each_argument_error_is_exactly_reported(self):
        messages = [
            "error: input image does not exist or is not a file",
            "error: model does not exist or is not a file",
            "error: model must be an ONNX file",
            "error: output directory does not exist",
            "error: output path must be a file",
            "error: input and output paths must differ",
        ]
        for message in messages:
            with self.subTest(message=message), patch(
                "run.parse_arguments", side_effect=ArgumentValidationError(message)
            ), patch("run.annotate_image") as annotate:
                status, stdout, stderr = self.call_main([])
            self.assertEqual((status, stdout, stderr), (1, "", f"{message}\n"))
            annotate.assert_not_called()

    def test_inference_error_has_fixed_diagnostic(self):
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", side_effect=InferenceError("private detail")
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual((status, stdout, stderr), (1, "", "error: inference failed\n"))
        self.assertNotIn("private detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_class_configuration_error_has_fixed_diagnostic(self):
        error = ClassConfigurationError()
        error.__cause__ = RuntimeError("private configuration detail\nTraceback")
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", side_effect=error
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
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", side_effect=error
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: model classes are incompatible\n"),
        )
        self.assertNotIn("private model detail", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_gamma_configuration_error_has_fixed_diagnostic(self):
        error = GammaConfigurationError("private gamma detail")
        error.__cause__ = RuntimeError("private cause\nTraceback")
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", side_effect=error
        ):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: invalid gamma configuration\n"),
        )
        self.assertNotIn("private gamma detail", stderr)
        self.assertNotIn("private cause", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_output_write_error_has_fixed_diagnostic(self):
        with patch("run.parse_arguments", return_value=self.arguments), patch(
            "run.annotate_image", side_effect=OutputWriteError("private detail")
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
        with patch("run.annotate_image") as annotate, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main([])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        annotate.assert_not_called()

    def test_unknown_option_keeps_argparse_status_two_and_stderr_usage(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("run.annotate_image") as annotate, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["--unknown"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        annotate.assert_not_called()

    def test_help_keeps_argparse_status_zero_and_uses_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("run.annotate_image") as annotate, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--input", stdout.getvalue())
        self.assertNotIn("Execution time:", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        annotate.assert_not_called()

    def test_main_guard_converts_return_value_to_process_status(self):
        message = "error: input image does not exist or is not a file"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "arguments.parse_arguments", side_effect=ArgumentValidationError(message)
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ), self.assertRaises(SystemExit) as error:
            runpy.run_module("run", run_name="__main__")
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), f"{message}\n")


if __name__ == "__main__":
    unittest.main()
