"""Index PhenoCam archives through NASA CMR and retrieve site metadata."""

import json
import urllib.request
from pathlib import Path
from ..common import DatasetError, atomic_text, write_csv
from .authentication import authenticated_opener, retrieve
from .schema import GRANULE_FIELDS, _ARCHIVE_NAME, _CMR_URL


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
