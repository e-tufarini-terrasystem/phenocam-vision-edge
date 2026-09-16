"""Phenocam: frames responsibility extracted without changing the data contract."""

import csv
import os
import tarfile
import tempfile
from pathlib import PurePosixPath
from pathlib import Path
from ..common import DatasetError, inspect_image, require_columns, stable_rank, write_csv
from .schema import ARCHIVE_DOWNLOAD_FIELDS, FRAME_FIELDS, _FRAME_NAME


def _safe_members(archive, config):
    members = archive.getmembers()
    if len(members) > int(config["phenocam"]["maximum_archive_members"]):
        raise DatasetError("archive_member_limit_exceeded")
    total = 0
    visible = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise DatasetError("archive_unsafe_path")
        if member.issym() or member.islnk() or member.isdev():
            raise DatasetError("archive_unsafe_member_type")
        if not member.isfile():
            continue
        total += member.size
        if total > int(config["phenocam"]["maximum_archive_uncompressed_bytes"]):
            raise DatasetError("archive_uncompressed_limit_exceeded")
        filename = path.name
        if "_IR_" in filename:
            continue
        match = _FRAME_NAME.fullmatch(filename)
        if match:
            visible.append((member, match))
    return visible


def _frame_order(frames, seed, granule_id):
    by_day = {}
    for member, match in frames:
        day = match.group("day")
        candidate = (member, match)
        key = stable_rank(seed, granule_id, day, member.name)
        if day not in by_day or key < by_day[day][0]:
            by_day[day] = (key, candidate)
    first = [entry[1] for _, entry in sorted(by_day.items())]
    first_names = {member.name for member, _ in first}
    remaining = [candidate for candidate in frames if candidate[0].name not in first_names]
    remaining.sort(key=lambda item: stable_rank(seed, granule_id, item[0].name))
    return first + remaining


def sample_archives(download_manifest, frame_dir, output_path, rejection_path, config):
    with Path(download_manifest).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, ARCHIVE_DOWNLOAD_FIELDS, download_manifest)
        rows = list(reader)
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejections = []
    target_per_site = int(config["phenocam"]["candidate_frames_per_site"])
    eligibility = config["image_eligibility"]
    for row in rows:
        accepted_for_site = 0
        try:
            with tarfile.open(row["local_path"], "r:gz") as archive:
                frames = _safe_members(archive, config)
                for member, match in _frame_order(frames, config["seed"], row["granule_id"]):
                    if accepted_for_site >= target_per_site:
                        break
                    filename = PurePosixPath(member.name).name
                    source_id = filename[:-4]
                    destination = frame_dir / row["site_id"] / f"{source_id}.source"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        descriptor, temporary_name = tempfile.mkstemp(
                            prefix=f".{destination.name}.",
                            suffix=".tmp",
                            dir=destination.parent,
                        )
                        temporary = Path(temporary_name)
                        try:
                            with os.fdopen(descriptor, "wb") as output:
                                remaining = member.size
                                while remaining:
                                    chunk = extracted.read(min(1024 * 1024, remaining))
                                    if not chunk:
                                        raise DatasetError("archive_member_truncated")
                                    output.write(chunk)
                                    remaining -= len(chunk)
                                output.flush()
                                os.fsync(output.fileno())
                            os.replace(temporary, destination)
                        finally:
                            try:
                                temporary.unlink()
                            except OSError:
                                pass
                    try:
                        properties = inspect_image(destination, eligibility)
                    except DatasetError as error:
                        rejections.append(
                            {
                                "source_dataset": "phenocam",
                                "source_version": config["phenocam"]["version"],
                                "source_subset": "",
                                "source_id": source_id,
                                "stage": "frame_sampling",
                                "reason_code": str(error),
                                "note": "",
                                "replacement_id": "",
                            }
                        )
                        continue
                    timestamp = (
                        f"{match.group('year')}-{match.group('month')}-{match.group('day')}T"
                        f"{match.group('time')[0:2]}:{match.group('time')[2:4]}:"
                        f"{match.group('time')[4:6]}"
                    )
                    date = timestamp[:10]
                    accepted.append(
                        {
                            "source_dataset": "phenocam",
                            "source_version": config["phenocam"]["version"],
                            "source_id": source_id,
                            "original_url": f"{row['download_url']}#{member.name}",
                            "landing_url": config["phenocam"]["landing_url"],
                            "license_url": config["phenocam"]["license_url"],
                            "attribution": "PhenoCam Network / ORNL DAAC, Dataset v3",
                            "site_id": row["site_id"],
                            "camera_id": row["site_id"],
                            "sequence_id": f"{row['site_id']}:{date}",
                            "event_id": "",
                            "timestamp": timestamp,
                            "season": row["assigned_season"],
                            "primary_veg_type": row["primary_veg_type"],
                            "group_id": f"phenocam:{row['site_id']}:{date}",
                            "local_path": destination.as_posix(),
                            "width": properties["width"],
                            "height": properties["height"],
                            "source_sha256": properties["source_sha256"],
                            "decoded_sha256": properties["decoded_sha256"],
                            "phash": properties["phash"],
                            "candidate_kind": "phenocam_review",
                            "review_status": "pending",
                            "review_reason": "complete_manual_target_annotation_and_negative_verification",
                        }
                    )
                    accepted_for_site += 1
        except (DatasetError, OSError, tarfile.TarError) as error:
            rejections.append(
                {
                    "source_dataset": "phenocam",
                    "source_version": config["phenocam"]["version"],
                    "source_subset": "",
                    "source_id": row["granule_id"],
                    "stage": "archive_sampling",
                    "reason_code": str(error) if isinstance(error, DatasetError) else "archive_read_failed",
                    "note": "",
                    "replacement_id": "",
                }
            )
    write_csv(output_path, FRAME_FIELDS, accepted)
    from .openimages import REJECTION_FIELDS

    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {
        "candidate_frames": len(accepted),
        "sites": len({row["site_id"] for row in accepted}),
        "rejections": len(rejections),
    }
