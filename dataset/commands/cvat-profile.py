#!/usr/bin/env python3
"""Store the local CVAT personal access token without echoing it."""

from datetime import datetime, timezone
from getpass import getpass
import os

from cvat_sdk.core.auth import AuthStore, ProfileEntry, get_auth_store_path


PROFILE = "phenocam-local"
SERVER = "http://localhost:8080"


def main():
    token = os.environ.get("CVAT_ACCESS_TOKEN", "").strip()
    if not token:
        token = getpass("CVAT personal access token (input hidden): ").strip()
    if not token:
        raise SystemExit("error: the token cannot be empty")
    store = AuthStore()
    store.put_profile(
        PROFILE,
        ProfileEntry(
            server=SERVER,
            token=token,
            created_date=datetime.now(timezone.utc).isoformat(),
        ),
        set_default=True,
    )
    print(f"Saved CVAT profile {PROFILE!r} in {get_auth_store_path()}")


if __name__ == "__main__":
    main()
