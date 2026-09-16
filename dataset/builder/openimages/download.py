"""Bounded, resumable downloader for preselected public image candidates."""

import csv
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from ..common import DatasetError, inspect_image, require_columns, write_csv
from .schema import CANDIDATE_FIELDS, REJECTION_FIELDS


DOWNLOAD_FIELDS = CANDIDATE_FIELDS + (
    "retrieval_url",
    "local_path",
    "retrieval_date",
    "width",
    "height",
    "source_sha256",
    "decoded_sha256",
    "phash",
    "download_status",
)


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        if urllib.parse.urlparse(new_url).scheme != "https":
            raise DatasetError("insecure_redirect")
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _target_path(image_dir, row):
    subset = row["source_subset"]
    source_id = row["source_id"]
    if not source_id or any(character not in "0123456789abcdef" for character in source_id):
        raise DatasetError("unsafe_source_id")
    if subset not in {"train", "validation", "test"}:
        raise DatasetError("unsafe_source_subset")
    return Path(image_dir) / subset / f"{source_id}.source"


def _fetch_one(row, image_dir, eligibility, timeout, mirror_template):
    destination = _target_path(image_dir, row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    retrieval_url = mirror_template.format(
        subset=row["source_subset"], source_id=row["source_id"]
    )
    status = "cached"
    rotation = int(row["source_rotation_ccw"] or 0)
    if not destination.exists():
        parsed = urllib.parse.urlparse(retrieval_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise DatasetError("insecure_or_invalid_retrieval_url")
        request = urllib.request.Request(
            retrieval_url,
            headers={"User-Agent": "phenocam-vision-edge-dataset/1"},
        )
        opener = urllib.request.build_opener(_SecureRedirectHandler())
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
                    raise DatasetError("insecure_redirect")
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise DatasetError("unexpected_content_type")
                length = response.headers.get("Content-Length")
                maximum = int(eligibility["maximum_source_bytes"])
                if length is not None and int(length) > maximum:
                    raise DatasetError("image_size_bytes_out_of_range")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise DatasetError("image_size_bytes_out_of_range")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total == 0:
                raise DatasetError("empty_download")
            inspect_image(temporary, eligibility, rotation)
            os.replace(temporary, destination)
            status = "downloaded"
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    properties = inspect_image(destination, eligibility, rotation)
    result = dict(row)
    result.update(
        {
            "retrieval_url": retrieval_url,
            "local_path": destination.as_posix(),
            "retrieval_date": datetime.now(timezone.utc).date().isoformat(),
            "width": properties["width"],
            "height": properties["height"],
            "source_sha256": properties["source_sha256"],
            "decoded_sha256": properties["decoded_sha256"],
            "phash": properties["phash"],
            "download_status": status,
        }
    )
    return result


def download_candidates(
    shortlist_path,
    image_dir,
    manifest_path,
    rejection_path,
    config,
    compiled_classes=(),
    candidate_kinds=(),
    limit=None,
    workers=8,
    timeout=30,
):
    with Path(shortlist_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, CANDIDATE_FIELDS, shortlist_path)
        rows = list(reader)
    wanted_classes = set(compiled_classes)
    wanted_kinds = set(candidate_kinds)
    if wanted_classes:
        rows = [
            row
            for row in rows
            if wanted_classes.intersection(row["compiled_classes"].split(";"))
        ]
    if wanted_kinds:
        rows = [row for row in rows if row["candidate_kind"] in wanted_kinds]
    if limit is not None:
        rows = rows[: int(limit)]

    eligibility = config["image_eligibility"]
    mirror_template = config["open_images"]["image_mirror_template"]
    completed = []
    rejections = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        pending = {
            executor.submit(
                _fetch_one,
                row,
                image_dir,
                eligibility,
                float(timeout),
                mirror_template,
            ): (index, row)
            for index, row in enumerate(rows)
        }
        for future in as_completed(pending):
            index, row = pending[future]
            try:
                completed.append((index, future.result()))
            except Exception as error:
                reason = str(error) if isinstance(error, DatasetError) else "download_failed"
                rejections.append(
                    (
                        index,
                        {
                            "source_dataset": row["source_dataset"],
                            "source_version": row["source_version"],
                            "source_subset": row["source_subset"],
                            "source_id": row["source_id"],
                            "stage": "image_download",
                            "reason_code": reason,
                            "note": "",
                            "replacement_id": "",
                        },
                    )
                )
    completed.sort(key=lambda item: item[0])
    rejections.sort(key=lambda item: item[0])
    write_csv(manifest_path, DOWNLOAD_FIELDS, (row for _, row in completed))
    write_csv(rejection_path, REJECTION_FIELDS, (row for _, row in rejections))
    return {
        "requested": len(rows),
        "available": len(completed),
        "rejected": len(rejections),
    }
