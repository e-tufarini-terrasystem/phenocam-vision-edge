"""Assemble the definitive independent negative review and import it."""

import json
from pathlib import Path

from ..annotation import NEGATIVE_EXPORT_FIELDS, _negative_document
from ..common import DatasetError, atomic_text, write_csv
from .negative_pool import identity as phenocam_identity
from .review_queue import _browser_rows, _read, _review_export
from .openimages_retry import again as retry_openimages_again
from .openimages_retry import prepare as prepare_openimages_retry


COMBINED_FIELDS = NEGATIVE_EXPORT_FIELDS + (
    "second_reviewer",
    "second_decision",
    "result",
)


def prepare_second_round(dataset_root, config, openimages_path, phenocam_path):
    work = Path(dataset_root) / "work"
    root = work / "annotation" / "final-negative-review" / "resolution"
    openimages = _review_export(
        openimages_path, root / "expected-openimages.csv", "openimages-resolution-a"
    )
    phenocam = _review_export(
        phenocam_path, root / "expected-phenocam.csv", "phenocam-resolution-a"
    )
    if any(row["decision"] != "confirmed_negative" for row in phenocam.values()):
        raise DatasetError("a PhenoCam replacement still contains a target")
    if any(row["decision"] != "confirmed_negative" for row in openimages.values()):
        return prepare_openimages_retry(dataset_root, config, openimages, phenocam)
    retained_openimages = _read(root / "retained-openimages.csv", NEGATIVE_EXPORT_FIELDS)
    retained_phenocam = _read(root / "retained-phenocam.csv", NEGATIVE_EXPORT_FIELDS)
    final_openimages = sorted((*retained_openimages, *openimages.values()), key=lambda row: row["source_identity"])
    final_phenocam = sorted((*retained_phenocam, *phenocam.values()), key=lambda row: row["source_identity"])
    if len(final_openimages) != 50 or len(final_phenocam) != 706:
        raise DatasetError("resolved negative composition is invalid")
    write_csv(root / "final-openimages-first.csv", NEGATIVE_EXPORT_FIELDS, final_openimages)
    write_csv(root / "final-phenocam-first.csv", NEGATIVE_EXPORT_FIELDS, final_phenocam)

    source_rows = _read(
        work / "review" / "phenocam" / "review.csv",
        ("source_dataset", "source_id", "local_path"),
    )
    by_identity = {phenocam_identity(row): row for row in source_rows}
    queue = [by_identity[row["source_identity"]] for row in final_phenocam]
    browser = _browser_rows(queue, root, "phenocam-b", phenocam_identity)
    page = root / "phenocam-b.html"
    with atomic_text(page) as output:
        output.write(
            _negative_document(
                browser,
                "phenocam-final-b",
                config["seed"],
                "PhenoCam — seconda verifica indipendente definitiva",
            )
        )
    write_csv(
        root / "expected-phenocam-b.csv",
        ("source_identity",),
        ({"source_identity": row["source_identity"]} for row in final_phenocam),
    )
    return {
        "openimages_negatives": len(final_openimages),
        "phenocam_negatives": len(final_phenocam),
        "second_review_page": page.resolve().as_uri(),
    }


def complete_retry(dataset_root, config, openimages_path):
    root = Path(dataset_root) / "work" / "annotation" / "final-negative-review" / "resolution"
    metadata = json.loads((root / "openimages-retry-metadata.json").read_text(encoding="utf-8"))
    retry = _review_export(
        openimages_path,
        root / "expected-openimages-retry.csv",
        metadata["review_round"],
    )
    if any(row["decision"] != "confirmed_negative" for row in retry.values()):
        return retry_openimages_again(dataset_root, config, retry)
    retained = _read(root / "retained-openimages-retry.csv", NEGATIVE_EXPORT_FIELDS)
    final_openimages = sorted((*retained, *retry.values()), key=lambda row: row["source_identity"])
    final_phenocam = _read(root / "final-phenocam-first.csv", NEGATIVE_EXPORT_FIELDS)
    if len(final_openimages) != 50 or len(final_phenocam) != 706:
        raise DatasetError("resolved negative composition is invalid")
    write_csv(root / "final-openimages-first.csv", NEGATIVE_EXPORT_FIELDS, final_openimages)
    source_rows = _read(
        Path(dataset_root) / "work" / "review" / "phenocam" / "review.csv",
        ("source_dataset", "source_id", "local_path"),
    )
    by_identity = {phenocam_identity(row): row for row in source_rows}
    queue = [by_identity[row["source_identity"]] for row in final_phenocam]
    browser = _browser_rows(queue, root, "phenocam-b", phenocam_identity)
    page = root / "phenocam-b.html"
    with atomic_text(page) as output:
        output.write(_negative_document(browser, "phenocam-final-b", config["seed"], "PhenoCam — seconda verifica indipendente definitiva"))
    write_csv(root / "expected-phenocam-b.csv", ("source_identity",), ({"source_identity": row["source_identity"]} for row in final_phenocam))
    return {"openimages_negatives": 50, "phenocam_negatives": 706, "second_review_page": page.resolve().as_uri()}


def import_final(review_root, second_path, output_path):
    root = Path(review_root)
    openimages = _read(root / "final-openimages-first.csv", NEGATIVE_EXPORT_FIELDS)
    first = {
        row["source_identity"]: row
        for row in _read(root / "final-phenocam-first.csv", NEGATIVE_EXPORT_FIELDS)
    }
    second = _review_export(
        second_path, root / "expected-phenocam-b.csv", "phenocam-final-b"
    )
    if set(first) != set(second):
        raise DatasetError("final PhenoCam reviews cover different images")
    output = []
    for row in openimages:
        output.append(
            {
                **row,
                "second_reviewer": "",
                "second_decision": "",
                "result": "accepted_negative",
            }
        )
    for identity in sorted(first):
        left, right = first[identity], second[identity]
        if left["reviewer"] == right["reviewer"]:
            raise DatasetError("PhenoCam negatives require two distinct reviewers")
        accepted = left["decision"] == right["decision"] == "confirmed_negative"
        output.append(
            {
                **left,
                "second_reviewer": right["reviewer"],
                "second_decision": right["decision"],
                "result": "accepted_negative" if accepted else "requires_resolution",
            }
        )
    write_csv(output_path, COMBINED_FIELDS, output)
    return {
        "rows": len(output),
        "accepted_negatives": sum(row["result"] == "accepted_negative" for row in output),
        "requires_resolution": sum(row["result"] != "accepted_negative" for row in output),
    }
