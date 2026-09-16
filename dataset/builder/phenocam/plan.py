"""Phenocam: plan responsibility extracted without changing the data contract."""

import json
import csv
from pathlib import Path
from ..common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from .schema import GRANULE_FIELDS, PLAN_FIELDS


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
