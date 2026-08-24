"""Own verified deletion of one validated source-image directory entry.

This boundary checks current type and identity immediately before unlinking and
maps ordinary failures to one sanitized error. Detection policy and public CLI
messages belong to callers, not this filesystem module.
"""

import stat


class SourceDeleteError(RuntimeError):
    """Report a source deletion failure without exposing private details."""

    def __init__(self, *_ignored):
        super().__init__()


def delete_source(input_path, expected_identity):
    if (
        type(expected_identity) is not tuple
        or len(expected_identity) != 2
        or any(type(value) is not int for value in expected_identity)
    ):
        raise SourceDeleteError()

    try:
        current = input_path.stat(follow_symlinks=False)
        # The directory entry must still be the regular file validated by the CLI.
        if not stat.S_ISREG(current.st_mode) or (
            current.st_dev,
            current.st_ino,
        ) != expected_identity:
            raise SourceDeleteError()
        input_path.unlink()
    except SourceDeleteError:
        raise
    except Exception:
        raise SourceDeleteError() from None
