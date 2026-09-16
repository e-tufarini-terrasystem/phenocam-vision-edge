"""Phenocam: download responsibility extracted without changing the data contract."""

import csv
from pathlib import Path
from ..common import DatasetError, require_columns, write_csv
from ..openimages.schema import REJECTION_FIELDS
from .authentication import authenticated_opener, retrieve
from .schema import ARCHIVE_DOWNLOAD_FIELDS, PLAN_FIELDS, _ARCHIVE_NAME


def download_archives(plan_path, archive_dir, manifest_path, rejection_path, config, netrc_path=None):
    with Path(plan_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, PLAN_FIELDS, plan_path)
        rows = list(reader)
    if sum(int(row["size_bytes"]) for row in rows) > int(
        config["phenocam"]["archive_download_budget_bytes"]
    ):
        raise DatasetError("PhenoCam archive plan exceeds the configured budget")
    opener = authenticated_opener(netrc_path)
    completed = []
    rejections = []
    archive_dir = Path(archive_dir)
    for row in rows:
        name = row["granule_id"].removeprefix("Phenocam_Images_V3.")
        if not _ARCHIVE_NAME.fullmatch(row["granule_id"]) or Path(name).name != name:
            raise DatasetError("unsafe PhenoCam granule name")
        destination = archive_dir / name
        try:
            status = retrieve(
                row["download_url"],
                destination,
                row["sha256"],
                row["size_bytes"],
                opener,
                maximum_bytes=config["phenocam"]["archive_download_budget_bytes"],
                timeout=300,
            )
            output = dict(row)
            output.update({"local_path": destination.as_posix(), "download_status": status})
            completed.append(output)
        except DatasetError as error:
            rejections.append(
                {
                    "source_dataset": "phenocam",
                    "source_version": config["phenocam"]["version"],
                    "source_subset": "",
                    "source_id": row["granule_id"],
                    "stage": "archive_download",
                    "reason_code": str(error),
                    "note": "",
                    "replacement_id": "",
                }
            )
    write_csv(manifest_path, ARCHIVE_DOWNLOAD_FIELDS, completed)
    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {"downloaded_or_cached": len(completed), "rejected": len(rejections)}
