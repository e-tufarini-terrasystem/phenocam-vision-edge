"""Own ordered, verified deletion of an input and optional metadata entry.

This boundary checks current type and identity immediately before unlinking and
maps ordinary failures to a file-specific sanitized error. Detection policy and CLI
messages belong to callers, not this filesystem module.
"""

import stat


class SourceDeleteError(RuntimeError):
    """Report a source deletion failure without exposing private details."""

    def __init__(self, *_ignored):
        super().__init__()


class MetadataDeleteError(SourceDeleteError):
    """Report a metadata deletion failure without exposing private details."""


def delete_source(input_path, expected_identity, metadata_path=None, metadata_identity=None):
    # Explicit order and error types share the same identity/type safety checks.
    entries = ((input_path, expected_identity, SourceDeleteError),)
    if metadata_path is not None:
        entries += ((metadata_path, metadata_identity, MetadataDeleteError),)
    for _, identity, error in entries:
        if (
            type(identity) is not tuple
            or len(identity) != 2
            or any(type(value) is not int for value in identity)
        ):
            raise error()

    for path, identity, error in entries:
        try:
            current = path.stat(follow_symlinks=False)
            # Only the validated regular entry may be unlinked. A failure stops
            # the sequence; deletion of an earlier entry cannot be rolled back.
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
            ) != identity:
                raise error()
            path.unlink()
        except SourceDeleteError:
            raise
        except Exception:
            raise error() from None
