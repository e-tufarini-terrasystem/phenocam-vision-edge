"""Track duplicate components, balanced quotas, and PhenoCam review records."""



PHENOCAM_SELECTION_FIELDS = (
    "provisional_role",
    "duplicate_group_id",
    "manual_review_required",
    "review_scope",
    "review_status",
    "decision",
    "reviewer",
    "reviewed_at",
    "corrected_annotations_json",
    "second_reviewer",
    "second_reviewed_at",
    "second_review_decision",
    "note",
)


def _identity(row):
    return f"{row['source_dataset']}::{row['source_id']}"


class _Components:
    def __init__(self, identities):
        self.parent = {identity: identity for identity in identities}

    def find(self, identity):
        parent = self.parent
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def _balanced_targets(total, values):
    return {
        value: total // len(values) + (index < total % len(values))
        for index, value in enumerate(values)
    }


def review_row(row, role, component_id):
    """Keep review attribution and required scope identical in both allocation phases."""
    scope = "complete_manual_target_annotation" if role == "positive" else "independent_negative_verification"
    output = dict(row)
    output.update(
        {
            "provisional_role": role,
            "duplicate_group_id": component_id,
            "manual_review_required": "true",
            "review_scope": scope,
            "review_status": "pending",
            "decision": "",
            "reviewer": "",
            "reviewed_at": "",
            "corrected_annotations_json": "",
            "second_reviewer": "",
            "second_reviewed_at": "",
            "second_review_decision": "",
            "note": "",
        }
    )
    return output
