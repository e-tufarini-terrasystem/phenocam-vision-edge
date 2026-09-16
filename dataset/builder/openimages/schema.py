"""Openimages: schema responsibility extracted without changing the data contract."""



CANDIDATE_FIELDS = (
    "source_dataset",
    "source_version",
    "source_subset",
    "source_id",
    "original_url",
    "landing_url",
    "author",
    "author_profile_url",
    "license_url",
    "attribution",
    "source_rotation_ccw",
    "provenance_group_id",
    "candidate_kind",
    "compiled_classes",
    "source_classes",
    "annotations_json",
    "confuser",
    "review_status",
    "review_reason",
)


REJECTION_FIELDS = (
    "source_dataset",
    "source_version",
    "source_subset",
    "source_id",
    "stage",
    "reason_code",
    "note",
    "replacement_id",
)
