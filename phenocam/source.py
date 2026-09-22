"""Delete the input, then optional metadata, checking each file before unlinking.

Checks compare the regular-file type and saved device/inode pair. They do not
make stat and unlink atomic or roll back a completed deletion.
"""

import stat


class SourceDeleteError(RuntimeError):
    """Report a source deletion failure without exposing private details."""

    def __init__(self, *_ignored):
        super().__init__()


class MetadataDeleteError(SourceDeleteError):
    """Report a metadata deletion failure without exposing private details."""


def validate_deletion_identities(input_identity, metadata_path=None, metadata_identity=None):
    """Reject malformed identities before inference or any filesystem mutation."""
    identities = ((input_identity, SourceDeleteError),)
    if metadata_path is not None:
        identities += ((metadata_identity, MetadataDeleteError),)
    for identity, error in identities:
        if (
            type(identity) is not tuple
            or len(identity) != 2
            or any(type(value) is not int for value in identity)
        ):
            raise error()


def delete_source(input_path, expected_identity, metadata_path=None, metadata_identity=None):
    validate_deletion_identities(expected_identity, metadata_path, metadata_identity)
    # Check both identity tuples first; stat each entry only when its turn arrives.
    entries = ((input_path, expected_identity, SourceDeleteError),)
    if metadata_path is not None:
        entries += ((metadata_path, metadata_identity, MetadataDeleteError),)
    for path, identity, error in entries:
        try:
            current = path.stat(follow_symlinks=False)
            # Refuse entries whose current type or device/inode differs from validation.
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
