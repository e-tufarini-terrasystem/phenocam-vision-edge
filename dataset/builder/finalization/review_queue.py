"""Build and import the final blind negative-review queues."""

import csv
import json
from collections import Counter
from pathlib import Path

from ..annotation import NEGATIVE_EXPORT_FIELDS, _bundle_name, _materialize_preview, _negative_document
from ..common import DatasetError, atomic_text, require_columns, write_csv
from .negative_pool import identity as phenocam_identity
from .negative_pool import select


def _read(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _openimages_identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _browser_rows(rows, output_root, source_name, identity_function):
    browser = []
    for index, row in enumerate(rows, start=1):
        identity = identity_function(row)
        destination = output_root / "images" / source_name / _bundle_name(identity, index)
        _materialize_preview(row, destination)
        browser.append(
            {
                "source_identity": identity,
                "preview_uri": destination.resolve().as_uri(),
                "baseline_detections": json.loads(row.get("baseline_detections_json", "[]") or "[]"),
                "context": "; ".join(
                    value
                    for value in (row.get("site_id", ""), row.get("timestamp", ""), row.get("season", ""))
                    if value
                ),
            }
        )
    return browser


def _identity_csv(path, rows, identity_function):
    write_csv(
        path,
        ("source_identity",),
        ({"source_identity": identity_function(row)} for row in rows),
    )


def prepare_negative_reviews(dataset_root, config):
    root = Path(dataset_root) / "work"
    output_root = root / "annotation" / "final-negative-review"
    phenocam = _read(
        root / "review" / "phenocam" / "review.csv",
        ("source_dataset", "source_id", "provisional_role", "local_path"),
    )
    imported = _read(
        root / "annotation" / "imported" / "phenocam-positive.csv",
        ("source_identity", "annotation_count", "reviewer", "imported_at"),
    )
    prior_receipts, prior, selected_new, final_phenocam = select(
        phenocam,
        imported,
        root / "global" / "embeddings.npz",
        int(config["source_frames"]["phenocam_v3"]["negative"]),
        config["seed"],
    )
    openimages_screened = _read(
        root / "openimages" / "baseline-screened.csv",
        ("source_dataset", "source_subset", "source_id", "candidate_kind"),
    )
    openimages = sorted(
        (row for row in openimages_screened if row["candidate_kind"] == "negative_review"),
        key=_openimages_identity,
    )
    if len(openimages) != int(config["source_frames"]["open_images_v7"]["negative"]):
        raise DatasetError("final Open Images negative selection has an invalid size")

    queues = (
        (openimages, "openimages", _openimages_identity, "openimages-a", "Open Images — verifica negativi"),
        (selected_new, "phenocam-a", phenocam_identity, "phenocam-a", "PhenoCam — prima verifica nuovi negativi"),
        (final_phenocam, "phenocam-b", phenocam_identity, "phenocam-b", "PhenoCam — seconda verifica indipendente"),
    )
    pages = []
    for rows, name, identity_function, review_round, title in queues:
        browser = _browser_rows(rows, output_root, name, identity_function)
        page = output_root / f"{review_round}.html"
        with atomic_text(page) as output:
            output.write(_negative_document(browser, review_round, config["seed"], title))
        pages.append(page.resolve().as_uri())

    _identity_csv(output_root / "expected-openimages.csv", openimages, _openimages_identity)
    _identity_csv(output_root / "expected-phenocam-a.csv", selected_new, phenocam_identity)
    _identity_csv(output_root / "expected-phenocam-b.csv", final_phenocam, phenocam_identity)
    prior_by_identity = {row["source_identity"]: row for row in prior_receipts}
    write_csv(
        output_root / "phenocam-prior-first-review.csv",
        ("source_identity", "decision", "reviewer", "reviewed_at"),
        (
            {
                "source_identity": identity,
                "decision": "confirmed_negative",
                "reviewer": prior_by_identity[identity]["reviewer"],
                "reviewed_at": prior_by_identity[identity]["imported_at"],
            }
            for identity in sorted(prior_by_identity)
        ),
    )
    statistics = {
        "openimages_manual_reviews": len(openimages),
        "phenocam_prior_first_reviews_reused": len(prior),
        "phenocam_new_first_reviews": len(selected_new),
        "phenocam_second_reviews": len(final_phenocam),
        "pages": pages,
        "site_counts": dict(Counter(row["site_id"] for row in final_phenocam)),
        "season_counts": dict(Counter(row["season"] for row in final_phenocam)),
    }
    with atomic_text(output_root / "statistics.json") as output:
        json.dump(statistics, output, indent=2, sort_keys=True)
        output.write("\n")
    return statistics


def _review_export(path, expected_path, expected_round):
    rows = _read(path, NEGATIVE_EXPORT_FIELDS)
    expected = {row["source_identity"] for row in _read(expected_path, ("source_identity",))}
    by_identity = {row["source_identity"]: row for row in rows}
    if set(by_identity) != expected or len(by_identity) != len(rows):
        raise DatasetError("negative review export does not match its canonical queue")
    for row in rows:
        attributed = row["reviewer"] and row["reviewed_at"]
        valid_decision = row["decision"] in {"confirmed_negative", "target_present", "uncertain"}
        if row["review_round"] != expected_round or not attributed or not valid_decision:
            raise DatasetError("negative review export is incomplete or invalid")
    return by_identity


def import_negative_reviews(review_root, openimages_path, first_path, second_path, output_path):
    review_root = Path(review_root)
    openimages = _review_export(openimages_path, review_root / "expected-openimages.csv", "openimages-a")
    first = _review_export(first_path, review_root / "expected-phenocam-a.csv", "phenocam-a")
    second = _review_export(second_path, review_root / "expected-phenocam-b.csv", "phenocam-b")
    prior = {
        row["source_identity"]: row
        for row in _read(
            review_root / "phenocam-prior-first-review.csv",
            ("source_identity", "decision", "reviewer", "reviewed_at"),
        )
    }
    first.update(prior)
    if set(first) != set(second):
        raise DatasetError("PhenoCam first and second reviews cover different images")
    output = []
    for identity, row in sorted(openimages.items()):
        result = "accepted_negative" if row["decision"] == "confirmed_negative" else "requires_resolution"
        output.append({**row, "second_reviewer": "", "second_decision": "", "result": result})
    for identity in sorted(first):
        left, right = first[identity], second[identity]
        if left["reviewer"] == right["reviewer"]:
            raise DatasetError("PhenoCam negatives require two distinct reviewers")
        accepted = left["decision"] == right["decision"] == "confirmed_negative"
        output.append({**left, "second_reviewer": right["reviewer"], "second_decision": right["decision"], "result": "accepted_negative" if accepted else "requires_resolution"})
    fields = NEGATIVE_EXPORT_FIELDS + ("second_reviewer", "second_decision", "result")
    write_csv(output_path, fields, output)
    return {
        "rows": len(output),
        "accepted_negatives": sum(row["result"] == "accepted_negative" for row in output),
        "requires_resolution": sum(row["result"] != "accepted_negative" for row in output),
    }
