"""
Verify optional products, deletion delegation, diagnostics, and success timing.

The argument boundary and `process_image` are mocked; their own modules retain
path, inventory, inference, and transaction-order coverage. This boundary owns
fixed metadata and source-deletion statuses without exposing private causes.
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
from phenocam.metadata import MetadataWriteError
from phenocam.source import MetadataDeleteError, SourceDeleteError


class RunTests(unittest.TestCase):
    def setUp(self):
        self.arguments = Arguments(
            input=Path("input.jpg"),
            annotated_output=Path("annotated.jpg"),
            privacy_output=Path("privacy.jpg"),
            model=Path("model.onnx"),
            meta=Path("input.meta"),
            delete_input_on_detection=False,
            input_identity=(17, 23),
            metadata_identity=(17, 24),
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
            model_path=self.arguments.model,
            input_path=self.arguments.input,
            annotated_output_path=self.arguments.annotated_output,
            privacy_output_path=self.arguments.privacy_output,
            metadata_path=self.arguments.meta,
            delete_input_on_detection=self.arguments.delete_input_on_detection,
            input_identity=self.arguments.input_identity,
            metadata_identity=self.arguments.metadata_identity,
        )
        self.assertEqual(
            (status, stdout, stderr), (0, "Execution time: 1.234 s\n", "")
        )

    def test_each_output_combination_is_delegated_once(self):
        combinations = (
            (None, None),
            (Path("annotated.jpg"), None),
            (None, Path("privacy.jpg")),
            (Path("annotated.jpg"), Path("privacy.jpg")),
        )
        for annotated, privacy in combinations:
            arguments = Arguments(
                self.arguments.input,
                annotated,
                privacy,
                self.arguments.model,
                self.arguments.meta,
                self.arguments.delete_input_on_detection,
                self.arguments.input_identity,
                self.arguments.metadata_identity,
            )
            with self.subTest(annotated=annotated, privacy=privacy), patch(
                "phenocam.__main__.parse_arguments", return_value=arguments
            ), patch("phenocam.__main__.process_image", return_value=0.5) as process:
                status, stdout, stderr = self.call_main([])
            process.assert_called_once_with(
                model_path=arguments.model,
                input_path=arguments.input,
                annotated_output_path=annotated,
                privacy_output_path=privacy,
                metadata_path=arguments.meta,
                delete_input_on_detection=arguments.delete_input_on_detection,
                input_identity=arguments.input_identity,
                metadata_identity=arguments.metadata_identity,
            )
            self.assertEqual(
                (status, stdout, stderr), (0, "Execution time: 0.500 s\n", "")
            )

    def test_enabled_deletion_intent_and_identity_are_delegated(self):
        arguments = Arguments(
            self.arguments.input,
            None,
            None,
            self.arguments.model,
            self.arguments.meta,
            True,
            self.arguments.input_identity,
            self.arguments.metadata_identity,
        )
        with patch(
            "phenocam.__main__.parse_arguments", return_value=arguments
        ), patch("phenocam.__main__.process_image", return_value=0.25) as process:
            status, stdout, stderr = self.call_main([])

        process.assert_called_once_with(
            model_path=arguments.model,
            input_path=arguments.input,
            annotated_output_path=None,
            privacy_output_path=None,
            metadata_path=arguments.meta,
            delete_input_on_detection=True,
            input_identity=arguments.input_identity,
            metadata_identity=arguments.metadata_identity,
        )
        self.assertEqual(
            (status, stdout, stderr), (0, "Execution time: 0.250 s\n", "")
        )

    def test_each_argument_error_is_exactly_reported(self):
        messages = [
            "error: input image does not exist or is not a file",
            "error: model does not exist or is not a file",
            "error: model must be an ONNX file",
            "error: at least one output path, metadata path, or input deletion is required",
            "error: input image must not be a symbolic link when deletion is enabled",
            "error: output directory does not exist",
            "error: output path must be a file",
            "error: input and output paths must differ",
            "error: output paths must differ",
            "error: metadata file does not exist or is not a file",
            "error: metadata file must use the .meta extension",
            "error: output path cannot be stored in metadata",
            "error: metadata path must differ from input, model, and output paths",
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

    def test_metadata_write_error_has_fixed_diagnostic(self):
        error = MetadataWriteError("private detail")
        error.__cause__ = RuntimeError("private cause\nTraceback")
        with patch(
            "phenocam.__main__.parse_arguments", return_value=self.arguments
        ), patch("phenocam.__main__.process_image", side_effect=error):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: metadata file could not be updated\n"),
        )
        self.assertNotIn("private", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_source_delete_error_has_fixed_diagnostic_without_timing(self):
        error = SourceDeleteError("private path")
        error.__cause__ = RuntimeError("private cause\nTraceback")
        with patch(
            "phenocam.__main__.parse_arguments", return_value=self.arguments
        ), patch("phenocam.__main__.process_image", side_effect=error):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr),
            (1, "", "error: input image could not be deleted\n"),
        )
        self.assertNotIn("private", stderr)
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

    def test_metadata_delete_error_has_fixed_diagnostic_without_timing(self):
        error = MetadataDeleteError("private path")
        with patch(
            "phenocam.__main__.parse_arguments", return_value=self.arguments
        ), patch("phenocam.__main__.process_image", side_effect=error):
            status, stdout, stderr = self.call_main([])
        self.assertEqual(
            (status, stdout, stderr), (1, "", "error: metadata file could not be deleted\n")
        )

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
        self.assertIn("--meta", help_text)
        self.assertIn("--delete-input-on-detection", help_text)
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
