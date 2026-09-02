"""Create deterministic groups and balance them without crossing leakage boundaries."""

import hashlib
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack

from ..common import DatasetError
from .records import CLASS_NAMES


SPLITS = ("train", "val", "test_id")
FRACTIONS = (0.8, 0.1, 0.1)
PHASH_THRESHOLD = 6
SEED = 42


class _Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _stable_cost(group, split):
    identity = min(record["image_id"] for record in group)
    digest = hashlib.sha256(f"dataset-v3:{SEED}:{identity}:{split}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 * 1e-8


def _solve(groups, targets, metrics, vectors):
    split_count, group_count = len(SPLITS), len(groups)
    deviation_count = len(metrics) * split_count
    variable_count = group_count * split_count + 2 * deviation_count
    totals = sum(vectors, Counter())
    objective = np.zeros(variable_count)
    for metric_index, metric in enumerate(metrics):
        weight = (3 if metric.startswith("class_") else 1) / max(1, totals[metric])
        start = group_count * split_count + 2 * metric_index * split_count
        objective[start : start + 2 * split_count] = weight
    for group_index, group in enumerate(groups):
        for split_index, split in enumerate(SPLITS):
            objective[group_index * split_count + split_index] = _stable_cost(group, split)

    blocks, lower, upper = [], [], []
    assignment = lil_matrix((group_count, variable_count))
    for group_index in range(group_count):
        assignment[group_index, group_index * split_count : (group_index + 1) * split_count] = 1
    blocks.append(assignment)
    lower.extend([1] * group_count)
    upper.extend([1] * group_count)

    sizes = lil_matrix((split_count, variable_count))
    for split_index, target in enumerate(targets):
        for group_index, vector in enumerate(vectors):
            sizes[split_index, group_index * split_count + split_index] = vector["images"]
        lower.append(target)
        upper.append(target)
    blocks.append(sizes)

    deviations = lil_matrix((deviation_count, variable_count))
    row = 0
    for metric in metrics:
        for split_index, fraction in enumerate(FRACTIONS):
            for group_index, vector in enumerate(vectors):
                deviations[row, group_index * split_count + split_index] = vector[metric]
            offset = group_count * split_count + 2 * row
            deviations[row, offset], deviations[row, offset + 1] = -1, 1
            target = totals[metric] * fraction
            lower.append(target)
            upper.append(target)
            row += 1
    blocks.append(deviations)

    matrix = vstack(blocks).tocsr()
    integrality = np.r_[np.ones(group_count * split_count), np.zeros(2 * deviation_count)]
    bounds = Bounds(np.zeros(variable_count), np.r_[np.ones(group_count * split_count), np.full(2 * deviation_count, np.inf)])
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.10, "presolve": True},
    )
    if not result.success or result.x is None:
        raise DatasetError("group allocation has no feasible solution")
    allocated = {split: [] for split in SPLITS}
    for group_index, group in enumerate(groups):
        start = group_index * split_count
        split = SPLITS[int(np.argmax(result.x[start : start + split_count]))]
        allocated[split].extend(group)
    return allocated


def _phenocam_groups(records):
    components = _Components(len(records))
    cameras = {}
    for index, record in enumerate(records):
        camera = record["camera_id"] or record["site_id"]
        if camera in cameras:
            components.union(index, cameras[camera])
        else:
            cameras[camera] = index
    # pHash is conservative here: it groups candidates but never deletes them.
    for left, left_record in enumerate(records):
        left_hash = int(left_record["phash"], 16)
        for right in range(left + 1, len(records)):
            if (left_hash ^ int(records[right]["phash"], 16)).bit_count() <= PHASH_THRESHOLD:
                components.union(left, right)
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[components.find(index)].append(record)
    return list(grouped.values())


def _phenocam_vector(group):
    vector = Counter(images=len(group), positive=sum(bool(record["_classes"]) for record in group))
    for record in group:
        month = int(record["timestamp"][5:7])
        vector[f"season_{(month % 12) // 3}"] += 1
        if "_luminance" in record:
            value = record["_luminance"]
            vector["light_dark" if value < 40 else "light_low" if value < 80 else "light_normal" if value < 180 else "light_bright"] += 1
    return vector


def _open_images_vector(group):
    record = group[0]
    vector = Counter(images=1, positive=bool(record["_classes"]))
    vector[f"stratum_{record['primary_stratum']}"] = 1
    for class_id, count in record["_classes"].items():
        vector[f"class_{class_id}"] += count
        vector[f"image_class_{class_id}"] = 1
    return vector


def allocate(records):
    ordered = sorted(records, key=lambda row: row["image_id"])
    open_images = [[record] for record in ordered if record["source_dataset"] == "open_images"]
    phenocam = _phenocam_groups([record for record in ordered if record["source_dataset"] == "phenocam"])
    phenocam.sort(key=lambda group: min(record["image_id"] for record in group))
    internal = [record for record in ordered if record["source_dataset"] == "internal"]
    unknown = {record["source_dataset"] for record in records} - {"open_images", "phenocam", "internal"}
    if unknown or len(open_images) != 1279 or sum(map(len, phenocam)) != 721 or len(internal) != 240:
        raise DatasetError("dataset provenance differs from the audited v3 composition")

    for group in phenocam:
        material = "\x1f".join(sorted(record["image_id"] for record in group)).encode()
        group_id = "phenocam-cluster-" + hashlib.sha256(material).hexdigest()[:20]
        for record in group:
            record["_partition_group_id"] = group_id

    phenocam_metrics = ["positive"] + [f"season_{index}" for index in range(4)]
    if all("_luminance" in record for group in phenocam for record in group):
        phenocam_metrics += ["light_dark", "light_low", "light_normal", "light_bright"]
    pheno = _solve(phenocam, (577, 72, 72), phenocam_metrics, [_phenocam_vector(group) for group in phenocam])
    open_metrics = ["positive"] + [f"class_{class_id}" for class_id in CLASS_NAMES] + [f"image_class_{class_id}" for class_id in CLASS_NAMES]
    open_split = _solve(open_images, (1023, 128, 128), open_metrics, [_open_images_vector(group) for group in open_images])

    assignments = {split: pheno[split] + open_split[split] for split in SPLITS}
    assignments["test_ood"] = internal
    for split, split_records in assignments.items():
        for record in split_records:
            if record.get("_assigned_split"):
                raise DatasetError("an image was assigned to more than one split")
            record["_assigned_split"] = split
            if record["source_dataset"] == "internal":
                record["_partition_group_id"] = f"internal-camera:{record['camera_id'] or record['site_id']}"
            elif record["source_dataset"] == "open_images":
                record["_partition_group_id"] = record["group_id"]
    if sum(map(len, assignments.values())) != len(records):
        raise DatasetError("split allocation did not account for every image")
    return assignments
