"""Phenocam: schema responsibility extracted without changing the data contract."""

import re


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
