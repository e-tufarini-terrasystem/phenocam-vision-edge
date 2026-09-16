"""Authenticated, bounded NASA Earthdata retrieval without credential logging."""

import http.cookiejar
import netrc
import os
import stat
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from ..common import DatasetError, sha256_file


_EARTHDATA_MACHINE = "urs.earthdata.nasa.gov"
_DATASET_ENV = Path(__file__).resolve().parents[2] / ".env"


def _env_credentials(path):
    path = Path(path)
    if not path.is_file():
        return None
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise DatasetError("Earthdata .env permissions must be 0600")
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetError("Earthdata .env cannot be read") from error
    for number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator:
            if key.startswith("EARTHDATA_"):
                raise DatasetError(f"Earthdata .env line {number} is malformed")
            continue
        if key not in {"EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '\"'}:
            value = value[1:-1]
        values[key] = value
    username = values.get("EARTHDATA_USERNAME", "")
    password = values.get("EARTHDATA_PASSWORD", "")
    if bool(username) != bool(password):
        raise DatasetError("Earthdata .env must define both username and password")
    return (username, password) if username else None


def _netrc_credentials(path):
    if not path.is_file():
        return None
    try:
        authentication = netrc.netrc(path).authenticators(_EARTHDATA_MACHINE)
    except (OSError, netrc.NetrcParseError) as error:
        raise DatasetError("Earthdata .netrc cannot be read") from error
    if not authentication:
        return None
    login, _, password = authentication
    return login, password


def authenticated_opener(netrc_path=None, env_path=None):
    credentials = None
    if netrc_path is None:
        username = os.environ.get("EARTHDATA_USERNAME", "")
        password = os.environ.get("EARTHDATA_PASSWORD", "")
        if bool(username) != bool(password):
            raise DatasetError(
                "Earthdata environment must define both username and password"
            )
        credentials = (username, password) if username else _env_credentials(
            _DATASET_ENV if env_path is None else env_path
        )
    if credentials is None:
        path = Path(netrc_path) if netrc_path else Path.home() / ".netrc"
        credentials = _netrc_credentials(path)
    if credentials is None:
        raise DatasetError(
            "Earthdata credentials are not configured in dataset/.env or ~/.netrc"
        )
    login, password = credentials
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, f"https://{_EARTHDATA_MACHINE}", login, password)
    cookies = http.cookiejar.CookieJar()
    return urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(manager),
        urllib.request.HTTPCookieProcessor(cookies),
    )


def retrieve(
    url,
    destination,
    expected_sha256,
    expected_size,
    opener,
    maximum_bytes,
    timeout=120,
):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(expected_size)
    maximum_bytes = int(maximum_bytes)
    if expected_size <= 0 or expected_size > maximum_bytes:
        raise DatasetError("Earthdata object size is outside the configured bound")
    if destination.exists():
        actual_size = destination.stat().st_size
        if actual_size <= maximum_bytes and sha256_file(destination) == expected_sha256:
            return (
                "cached"
                if actual_size == expected_size
                else "cached_cmr_size_mismatch"
            )
        raise DatasetError("existing Earthdata object does not match catalog metadata")
    if urllib.parse.urlparse(url).scheme != "https":
        raise DatasetError("Earthdata URL must use HTTPS")
    request = urllib.request.Request(
        url, headers={"User-Agent": "phenocam-vision-edge-dataset/1"}
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output, opener.open(
            request, timeout=timeout
        ) as response:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise DatasetError("Earthdata redirected to an insecure URL")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise DatasetError("Earthdata object exceeds the configured bound")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise DatasetError("Earthdata object checksum mismatch")
        os.replace(temporary, destination)
        return "downloaded" if total == expected_size else "downloaded_cmr_size_mismatch"
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
