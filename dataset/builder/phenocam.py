"""Index PhenoCam v3 granules through NASA CMR without fetching archives."""

import json
import csv
import os
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import PurePosixPath
from pathlib import Path

from .common import (
    DatasetError,
    atomic_text,
    inspect_image,
    require_columns,
    stable_rank,
    write_csv,
)
from .earthdata import authenticated_opener, retrieve


GRANULE_FIELDS = (
    "source_dataset",
    "source_version",
    "granule_id",
    "site_id",
    "year",
    "month",
    "beginning_datetime",
    "ending_datetime",
    "west",
    "east",
    "north",
    "south",
    "download_url",
    "s3_url",
    "size_bytes",
    "sha256",
)

_ARCHIVE_NAME = re.compile(
    r"^Phenocam_Images_V3\.(?P<site>.+)_(?P<year>\d{4})_(?P<month>\d{2})\.tar\.gz$"
)
_CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"


def _get_json(url, timeout=60):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.nasa.cmr.umm_results+json",
            "User-Agent": "phenocam-vision-edge-dataset/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if urllib.parse.urlparse(response.geturl()).scheme != "https":
                raise DatasetError("CMR redirected to an insecure URL")
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise DatasetError("cannot retrieve NASA CMR metadata") from error


def _related_url(umm, url_type):
    for item in umm.get("RelatedUrls", ()):
        if item.get("Type") == url_type:
            value = item.get("URL", "")
            if value:
                return value
    return ""


def _archive_information(umm):
    entries = umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", ())
    if not entries:
        return "", ""
    entry = entries[0]
    checksum = entry.get("Checksum", {})
    if checksum.get("Algorithm", "").upper() != "SHA-256":
        return entry.get("SizeInBytes", ""), ""
    return entry.get("SizeInBytes", ""), checksum.get("Value", "")


def _bounds(umm):
    rectangles = (
        umm.get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
        .get("BoundingRectangles", ())
    )
    if not rectangles:
        return "", "", "", ""
    rectangle = rectangles[0]
    return (
        rectangle.get("WestBoundingCoordinate", ""),
        rectangle.get("EastBoundingCoordinate", ""),
        rectangle.get("NorthBoundingCoordinate", ""),
        rectangle.get("SouthBoundingCoordinate", ""),
    )


def index_granules(output_dir, config, page_size=2000):
    source_config = config["phenocam"]
    concept_id = source_config["collection_concept_id"]
    page_size = max(1, min(2000, int(page_size)))
    rows = []
    expected_hits = None
    page_number = 1
    geojson_url = ""
    geojson_size = ""
    geojson_sha256 = ""
    while True:
        query = urllib.parse.urlencode(
            {
                "collection_concept_id": concept_id,
                "page_size": page_size,
                "page_num": page_number,
            }
        )
        result = _get_json(f"{_CMR_URL}?{query}")
        hits = int(result.get("hits", 0))
        expected_hits = hits if expected_hits is None else expected_hits
        items = result.get("items", ())
        if not items:
            break
        for item in items:
            umm = item.get("umm", {})
            granule_id = umm.get("GranuleUR", "")
            match = _ARCHIVE_NAME.fullmatch(granule_id)
            if not match:
                if granule_id.endswith("phenocam_images_sites_v3.geojson"):
                    geojson_url = _related_url(umm, "GET DATA")
                    geojson_size, geojson_sha256 = _archive_information(umm)
                continue
            download_url = _related_url(umm, "GET DATA")
            s3_url = _related_url(umm, "GET DATA VIA DIRECT ACCESS")
            if not download_url.startswith("https://") or not s3_url.startswith("s3://"):
                raise DatasetError(f"{granule_id}: missing approved download URLs")
            size_bytes, checksum = _archive_information(umm)
            if not size_bytes or not checksum:
                raise DatasetError(f"{granule_id}: missing size or SHA-256")
            temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
            west, east, north, south = _bounds(umm)
            rows.append(
                {
                    "source_dataset": "phenocam",
                    "source_version": source_config["version"],
                    "granule_id": granule_id,
                    "site_id": match.group("site"),
                    "year": match.group("year"),
                    "month": match.group("month"),
                    "beginning_datetime": temporal.get("BeginningDateTime", ""),
                    "ending_datetime": temporal.get("EndingDateTime", ""),
                    "west": west,
                    "east": east,
                    "north": north,
                    "south": south,
                    "download_url": download_url,
                    "s3_url": s3_url,
                    "size_bytes": int(size_bytes),
                    "sha256": checksum,
                }
            )
        page_number += 1
        if page_number * page_size > hits + page_size:
            break

    if expected_hits is None or len(rows) + (1 if geojson_url else 0) != expected_hits:
        raise DatasetError(
            f"CMR index incomplete: {len(rows)} archives plus site metadata, {expected_hits} expected"
        )
    rows.sort(key=lambda row: (row["site_id"], row["year"], row["month"]))
    output_dir = Path(output_dir)
    write_csv(output_dir / "granules.csv", GRANULE_FIELDS, rows)
    access = {
        "collection_concept_id": concept_id,
        "archive_count": len(rows),
        "site_count": len({row["site_id"] for row in rows}),
        "geojson_url": geojson_url,
        "geojson_size_bytes": int(geojson_size),
        "geojson_sha256": geojson_sha256,
        "archive_access": "earthdata_credentials_required",
        "operational_validation": "pending",
    }
    with atomic_text(output_dir / "index.json") as output:
        json.dump(access, output, indent=2, sort_keys=True)
        output.write("\n")
    return access


PLAN_FIELDS = GRANULE_FIELDS + ("primary_veg_type", "assigned_season")
ARCHIVE_DOWNLOAD_FIELDS = PLAN_FIELDS + ("local_path", "download_status")
FRAME_FIELDS = (
    "source_dataset",
    "source_version",
    "source_id",
    "original_url",
    "landing_url",
    "license_url",
    "attribution",
    "site_id",
    "camera_id",
    "sequence_id",
    "event_id",
    "timestamp",
    "season",
    "primary_veg_type",
    "group_id",
    "local_path",
    "width",
    "height",
    "source_sha256",
    "decoded_sha256",
    "phash",
    "candidate_kind",
    "review_status",
    "review_reason",
)

_FRAME_NAME = re.compile(
    r"^(?P<site>.+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<time>\d{6})\.jpg$",
    re.IGNORECASE,
)


def fetch_site_metadata(index_path, output_path, config, netrc_path=None):
    try:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError("cannot read PhenoCam index metadata") from error
    opener = authenticated_opener(netrc_path)
    status = retrieve(
        index["geojson_url"],
        output_path,
        index["geojson_sha256"],
        index["geojson_size_bytes"],
        opener,
        maximum_bytes=10 * 1024 * 1024,
    )
    return {"site_metadata": status, "operational_validation": "pending"}


def _season(month):
    month = int(month)
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def plan_archives(granules_path, sites_path, output_path, statistics_path, config):
    with Path(granules_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, GRANULE_FIELDS, granules_path)
        granules = list(reader)
    try:
        sites_document = json.loads(Path(sites_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError("cannot read PhenoCam site metadata") from error
    site_metadata = {}
    for feature in sites_document.get("features", ()):
        properties = feature.get("properties", {})
        site_id = str(properties.get("sitename", "")).strip()
        if site_id:
            site_metadata[site_id] = properties
    if len(site_metadata) < int(config["phenocam"]["minimum_sites"]):
        raise DatasetError("PhenoCam site metadata does not cover the minimum site count")

    by_site = {}
    for row in granules:
        by_site.setdefault(row["site_id"], []).append(row)
    available_sites = sorted(set(by_site) & set(site_metadata))
    seed = config["seed"]
    site_target = int(config["phenocam"]["candidate_sites"])
    budget = int(config["phenocam"]["archive_download_budget_bytes"])
    selected = []
    selected_sites = set()
    total_bytes = 0
    seasons = ("winter", "spring", "summer", "autumn")
    targets = {
        season: site_target // len(seasons) + (index < site_target % len(seasons))
        for index, season in enumerate(seasons)
    }
    pools = {}
    for season in seasons:
        choices = []
        for site in available_sites:
            candidates = [
                row
                for row in by_site[site]
                if _season(row["month"]) == season
                and int(row["size_bytes"]) >= 50 * 1024 * 1024
            ]
            if not candidates:
                continue
            choice = min(
                candidates,
                key=lambda row: (
                    int(row["size_bytes"]),
                    stable_rank(seed, site, row["year"], row["month"]),
                ),
            )
            choices.append(
                (
                    int(choice["size_bytes"]),
                    stable_rank(seed, "phenocam-plan", season, site),
                    site,
                    choice,
                )
            )
        pools[season] = iter(sorted(choices))

    season_counts = {season: 0 for season in seasons}
    while len(selected) < site_target:
        progress = False
        for season in seasons:
            if season_counts[season] >= targets[season]:
                continue
            for size, _, site, choice in pools[season]:
                if site in selected_sites:
                    continue
                if total_bytes + size > budget:
                    continue
                output = dict(choice)
                output["primary_veg_type"] = site_metadata[site].get(
                    "primary_veg_type", ""
                )
                output["assigned_season"] = season
                selected.append(output)
                selected_sites.add(site)
                season_counts[season] += 1
                total_bytes += size
                progress = True
                break
        if not progress:
            break
    if len(selected) < int(config["phenocam"]["minimum_sites"]):
        raise DatasetError("archive budget cannot cover the minimum PhenoCam site count")
    write_csv(output_path, PLAN_FIELDS, selected)
    statistics = {
        "archive_count": len(selected),
        "site_count": len(selected),
        "estimated_download_bytes": total_bytes,
        "season_counts": season_counts,
        "agriculture_site_count": sum(
            row["primary_veg_type"] == "AG" for row in selected
        ),
        "operational_validation": "pending",
    }
    with atomic_text(statistics_path) as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics


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
    from .openimages import REJECTION_FIELDS

    write_csv(rejection_path, REJECTION_FIELDS, rejections)
    return {"downloaded_or_cached": len(completed), "rejected": len(rejections)}


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
