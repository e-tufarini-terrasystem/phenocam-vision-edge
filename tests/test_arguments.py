"""Verify CLI action requirements, path aliases, file identities, and metadata paths."""

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
        self.metadata = self.root / "image.meta"
        self.input.write_bytes(b"image")
        self.model.write_bytes(b"model")
        self.metadata.write_bytes(b"metadata")
        input_stat = self.input.stat()
        self.input_identity = (input_stat.st_dev, input_stat.st_ino)

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
            arguments,
            Arguments(
                self.input,
                self.output,
                None,
                self.model,
                None,
                False,
                self.input_identity,
            ),
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
            privacy_only,
            Arguments(
                self.input,
                None,
                privacy,
                self.model,
                None,
                False,
                self.input_identity,
            ),
        )
        self.assertEqual(
            dual,
            Arguments(
                self.input,
                self.output,
                privacy,
                self.model,
                None,
                False,
                self.input_identity,
            ),
        )

    def test_metadata_option_retains_cli_path_spelling(self):
        spelling = self.root / "." / self.metadata.name
        arguments = parse_arguments([*self.argv(), "--meta", str(spelling)])
        self.assertEqual(
            arguments,
            Arguments(
                self.input,
                self.output,
                None,
                self.model,
                spelling,
                False,
                self.input_identity,
            ),
        )

    def test_deletion_flag_is_boolean_and_captures_input_identity(self):
        arguments = parse_arguments([*self.argv(), "--delete-input-on-detection"])

        self.assertTrue(arguments.delete_input_on_detection)
        self.assertEqual(arguments.input_identity, self.input_identity)

    def test_deletion_flag_accepts_no_value(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            parse_arguments([*self.argv(), "--delete-input-on-detection", "true"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("usage:", stderr.getvalue())

    def test_deletion_allows_no_image_outputs(self):
        arguments = parse_arguments(
            [
                "--input",
                str(self.input),
                "--model",
                str(self.model),
                "--delete-input-on-detection",
            ]
        )

        self.assertIsNone(arguments.annotated_output)
        self.assertIsNone(arguments.privacy_output)
        self.assertTrue(arguments.delete_input_on_detection)

    def test_metadata_only_without_deletion_is_accepted(self):
        arguments = parse_arguments(
            [
                "--input",
                str(self.input),
                "--model",
                str(self.model),
                "--meta",
                str(self.metadata),
            ],
        )
        self.assertIsNone(arguments.annotated_output)
        self.assertIsNone(arguments.privacy_output)
        self.assertEqual(arguments.meta, self.metadata)
        self.assertFalse(arguments.delete_input_on_detection)
        self.assertEqual(self.input.read_bytes(), b"image")
        self.assertEqual(self.metadata.read_bytes(), b"metadata")

    def test_metadata_only_still_validates_the_metadata_file(self):
        wrong_extension = self.root / "image.txt"
        wrong_extension.write_bytes(b"metadata")
        for path, error in (
            (self.root / "missing.meta", "error: metadata file does not exist or is not a file"),
            (self.root, "error: metadata file does not exist or is not a file"),
            (wrong_extension, "error: metadata file must use the .meta extension"),
        ):
            with self.subTest(path=path):
                self.assert_validation_error(
                    error,
                    ["--input", str(self.input), "--model", str(self.model), "--meta", str(path)],
                )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_input_symlink_is_accepted_only_without_deletion(self):
        link = self.root / "input-link.jpg"
        try:
            link.symlink_to(self.input)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")

        accepted = parse_arguments(
            ["--input", str(link), *self.argv()[2:]]
        )
        self.assertEqual(accepted.input_identity, self.input_identity)
        self.assert_validation_error(
            "error: input image must not be a symbolic link when deletion is enabled",
            [
                "--input",
                str(link),
                *self.argv()[2:],
                "--delete-input-on-detection",
            ],
        )

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_hard_link_input_is_accepted_with_deletion(self):
        link = self.root / "input-hard-link.jpg"
        try:
            os.link(self.input, link)
        except OSError as error:
            self.skipTest(f"hard-link creation is unavailable: {error.errno}")

        arguments = parse_arguments(
            [
                "--input",
                str(link),
                *self.argv()[2:],
                "--delete-input-on-detection",
            ]
        )
        self.assertEqual(arguments.input_identity, self.input_identity)

    def test_input_stat_failure_uses_fixed_input_error(self):
        real_stat = self.input.stat()
        with patch.object(Path, "stat", side_effect=[real_stat, OSError("private")]):
            self.assert_validation_error(
                "error: input image does not exist or is not a file", self.argv()
            )

    def test_metadata_suffix_is_case_insensitive(self):
        metadata = self.root / "image.META"
        metadata.write_bytes(b"metadata")
        self.assertEqual(
            parse_arguments([*self.argv(), "--meta", str(metadata)]).meta,
            metadata,
        )

    def test_missing_metadata_has_fixed_error(self):
        missing = self.root / "missing.meta"
        self.assert_validation_error(
            "error: metadata file does not exist or is not a file",
            [*self.argv(), "--meta", str(missing)],
        )

    def test_metadata_directory_has_fixed_error(self):
        directory = self.root / "directory.meta"
        directory.mkdir()
        self.assert_validation_error(
            "error: metadata file does not exist or is not a file",
            [*self.argv(), "--meta", str(directory)],
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_metadata_symlink_is_rejected(self):
        link = self.root / "link.meta"
        try:
            link.symlink_to(self.metadata)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error.errno}")
        self.assert_validation_error(
            "error: metadata file does not exist or is not a file",
            [*self.argv(), "--meta", str(link)],
        )

    def test_metadata_wrong_suffix_has_fixed_error(self):
        metadata = self.root / "image.txt"
        metadata.write_bytes(b"metadata")
        self.assert_validation_error(
            "error: metadata file must use the .meta extension",
            [*self.argv(), "--meta", str(metadata)],
        )

    @unittest.skipUnless(hasattr(os, "link"), "hard links are unavailable")
    def test_metadata_aliases_input_model_or_outputs_are_rejected(self):
        privacy = self.root / "privacy.jpg"
        self.output.write_bytes(b"output")
        privacy.write_bytes(b"privacy")
        protected = (self.input, self.model, self.output, privacy)
        for index, target in enumerate(protected):
            alias = self.root / f"alias-{index}.meta"
            try:
                os.link(target, alias)
            except OSError as error:
                self.skipTest(f"hard-link creation is unavailable: {error.errno}")
            argv = [*self.argv(), "--privacy-output", str(privacy), "--meta", str(alias)]
            with self.subTest(target=target):
                self.assert_validation_error(
                    "error: metadata path must differ from input, model, and output paths",
                    argv,
                )
            alias.unlink()

    def test_metadata_output_identity_by_resolution_is_rejected(self):
        self.assert_validation_error(
            "error: metadata path must differ from input, model, and output paths",
            self.argv(output=self.metadata) + ["--meta", str(self.metadata)],
        )

    def test_metadata_rejects_carriage_return_or_line_feed_in_outputs(self):
        for character in ("\r", "\n"):
            output = self.root / f"unsafe{character}name.jpg"
            with self.subTest(character=repr(character)):
                self.assert_validation_error(
                    "error: output path cannot be stored in metadata",
                    self.argv(output=output) + ["--meta", str(self.metadata)],
                )

    def test_output_line_breaks_remain_accepted_without_metadata(self):
        output = self.root / "compatible\nname.jpg"
        self.assertEqual(parse_arguments(self.argv(output=output)).annotated_output, output)

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

    def test_missing_actions_has_fixed_semantic_error(self):
        self.assert_validation_error(
            "error: at least one output path, metadata path, or input deletion is required",
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

    def test_direct_and_normalized_input_replacement_is_accepted(self):
        (self.root / "nested").mkdir()
        original = self.input.read_bytes()
        for option in ("--annotated-output", "--privacy-output"):
            for output in (self.input, self.root / "nested" / ".." / self.input.name):
                with self.subTest(option=option, output=output):
                    arguments = parse_arguments(self.argv(output=output, output_option=option))
                    self.assertEqual(getattr(arguments, option[2:].replace("-", "_")), output)
                    self.assertEqual(self.input.read_bytes(), original)

    def test_input_replacement_with_deletion_is_accepted(self):
        for option in ("--annotated-output", "--privacy-output"):
            with self.subTest(option=option):
                arguments = parse_arguments(
                    self.argv(output=self.input, output_option=option)
                    + ["--delete-input-on-detection"],
                )
                self.assertTrue(arguments.delete_input_on_detection)
                self.assertEqual(self.input.read_bytes(), b"image")

    def test_deletion_captures_metadata_identity_without_modifying_files(self):
        arguments = parse_arguments(
            self.argv() + ["--meta", str(self.metadata), "--delete-input-on-detection"]
        )
        current = self.metadata.stat()
        self.assertEqual(arguments.metadata_identity, (current.st_dev, current.st_ino))
        self.assertEqual(self.metadata.read_bytes(), b"metadata")

    def test_metadata_identity_stat_failure_is_sanitized(self):
        real_stat = Path.stat

        def failing_stat(path, *, follow_symlinks=True):
            if path == self.metadata and not follow_symlinks:
                raise OSError("private path")
            return real_stat(path, follow_symlinks=follow_symlinks)

        with patch.object(Path, "stat", failing_stat):
            self.assert_validation_error(
                "error: metadata file does not exist or is not a file",
                self.argv() + ["--meta", str(self.metadata), "--delete-input-on-detection"],
            )

    def test_only_one_product_may_replace_input(self):
        self.assert_validation_error(
            "error: output paths must differ",
            self.argv(output=self.input) + ["--privacy-output", str(self.input)],
        )
        for annotated, privacy in ((self.input, self.output), (self.output, self.input)):
            with self.subTest(annotated=annotated, privacy=privacy):
                arguments = parse_arguments(
                    self.argv(output=annotated) + ["--privacy-output", str(privacy)]
                )
                self.assertEqual(arguments.annotated_output, annotated)
                self.assertEqual(arguments.privacy_output, privacy)

    def test_input_hard_link_aliases_are_rejected(self):
        os.link(self.input, self.output)
        for option in ("--annotated-output", "--privacy-output"):
            with self.subTest(option=option):
                self.assert_validation_error(
                    "error: input and output paths must differ",
                    self.argv(output_option=option),
                )

    def test_input_alias_through_symbolic_directory_is_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.assert_validation_error(
            "error: input and output paths must differ",
            self.argv(output=alias / self.input.name),
        )

    def test_symbolic_input_cannot_be_replaced_via_same_path_or_target(self):
        target = self.root / "target.jpg"
        self.input.rename(target)
        self.input.symlink_to(target)
        for option in ("--annotated-output", "--privacy-output"):
            for output in (self.input, target):
                with self.subTest(option=option, output=output):
                    self.assert_validation_error(
                        "error: input and output paths must differ",
                        self.argv(output=output, output_option=option),
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
