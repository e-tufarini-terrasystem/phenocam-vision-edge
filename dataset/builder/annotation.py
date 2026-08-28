"""Build local CVAT bundles and blind negative-review packets."""

import csv
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .common import DatasetError, atomic_text, require_columns, stable_rank, write_csv
from .common import sha256_file


MAPPING_FIELDS = (
    "source_identity",
    "bundle_file_name",
    "local_path",
    "width",
    "height",
    "source_sha256",
    "review_scope",
    "annotation_source",
)

NEGATIVE_EXPORT_FIELDS = (
    "source_identity",
    "decision",
    "reviewer",
    "reviewed_at",
    "review_round",
    "note",
)

POSITIVE_IMPORT_FIELDS = (
    "source_identity",
    "bundle_file_name",
    "review_status",
    "annotation_count",
    "annotations_json",
    "annotator",
    "reviewer",
    "imported_at",
    "source_export_sha256",
)


def _read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, required, path)
        return list(reader)


def _openimages_identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _phenocam_identity(row):
    return f"{row['source_dataset']}::{row['source_id']}"


def _bundle_name(identity, index):
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"image_{index:04d}_{digest}.jpg"


def _materialize_preview(row, destination):
    destination = Path(destination)
    if destination.is_file():
        with Image.open(destination) as existing:
            if existing.size == (int(row["width"]), int(row["height"])):
                return "cached"
        raise DatasetError(f"stale annotation preview: {destination}")
    with Image.open(row["local_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        rotation = int(row.get("source_rotation_ccw", "") or 0)
        if rotation:
            image = image.rotate(rotation, expand=True)
        image.load()
    expected = (int(row["width"]), int(row["height"]))
    if image.size != expected:
        raise DatasetError(f"annotation preview dimensions disagree for {row['source_id']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        image.save(temporary, format="JPEG", quality=95, subsampling=0, optimize=True)
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return "created"


def _categories(config):
    return [
        # COCO category IDs are kept positive for CVAT compatibility; the
        # reviewed import maps names back to the builder's zero-based IDs.
        {"id": class_id + 1, "name": name, "supercategory": "target"}
        for name, class_id in sorted(
            config["compiled_class_ids"].items(), key=lambda item: item[1]
        )
    ]


def _write_reproducible_member(archive, source_path, archive_name, compression):
    info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, Path(source_path).read_bytes())


def _openimages_annotations(row):
    width, height = int(row["width"]), int(row["height"])
    annotations = []
    for source in json.loads(row["annotations_json"]):
        left = float(source["xmin"]) * width
        top = float(source["ymin"]) * height
        box_width = (float(source["xmax"]) - float(source["xmin"])) * width
        box_height = (float(source["ymax"]) - float(source["ymin"])) * height
        annotations.append(
            {
                "category_id": int(source["class_id"]) + 1,
                "bbox": [left, top, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "attributes": {"annotation_source": "open_images"},
            }
        )
    return annotations


def _phenocam_annotations(row):
    annotations = []
    for source in json.loads(row["baseline_detections_json"]):
        box_width = float(source["x2"]) - float(source["x1"])
        box_height = float(source["y2"]) - float(source["y1"])
        annotations.append(
            {
                "category_id": int(source["class_id"]) + 1,
                "bbox": [float(source["x1"]), float(source["y1"]), box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
                "attributes": {
                    "annotation_source": "baseline_suggestion",
                    "confidence": float(source["confidence"]),
                },
            }
        )
    return annotations


def _build_coco_bundle(name, rows, identity_function, annotation_function, output_root, config):
    bundle_root = Path(output_root) / "cvat" / name
    image_root = bundle_root / "images" / "default"
    coco_images = []
    coco_annotations = []
    mappings = []
    annotation_id = 1
    created = 0
    for image_id, row in enumerate(rows, start=1):
        identity = identity_function(row)
        file_name = _bundle_name(identity, image_id)
        destination = image_root / file_name
        created += _materialize_preview(row, destination) == "created"
        coco_images.append(
            {
                "id": image_id,
                # CVAT uploads these resources as bare filenames. Keeping the
                # COCO identity identical lets the annotation archive match an
                # already-created task without relying on path normalization.
                "file_name": file_name,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "source_identity": identity,
            }
        )
        source_annotations = annotation_function(row)
        for annotation in source_annotations:
            output = dict(annotation)
            output.update({"id": annotation_id, "image_id": image_id})
            coco_annotations.append(output)
            annotation_id += 1
        mappings.append(
            {
                "source_identity": identity,
                "bundle_file_name": file_name,
                "local_path": row["local_path"],
                "width": row["width"],
                "height": row["height"],
                "source_sha256": row["source_sha256"],
                "review_scope": row.get("review_scope", "complete_manual_target_annotation"),
                "annotation_source": (
                    "open_images" if row["source_dataset"] == "open_images" else "baseline_suggestion"
                ),
            }
        )
    coco = {
        "info": {
            "description": name,
            "annotations_are_ground_truth": False,
            "instructions": "Every frame and every box requires human review.",
        },
        "licenses": [],
        "categories": _categories(config),
        "images": coco_images,
        "annotations": coco_annotations,
    }
    annotation_path = bundle_root / "annotations" / "instances_default.json"
    with atomic_text(annotation_path) as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    write_csv(bundle_root / "mapping.csv", MAPPING_FIELDS, mappings)
    with atomic_text(bundle_root / "labels.json") as output:
        json.dump([{"name": item["name"], "attributes": []} for item in _categories(config)], output, indent=2)
        output.write("\n")
    archive_path = Path(output_root) / "cvat" / f"{name}.coco.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_STORED) as archive:
        _write_reproducible_member(
            archive,
            annotation_path,
            "annotations/instances_default.json",
            zipfile.ZIP_STORED,
        )
        for image in sorted(image_root.iterdir()):
            _write_reproducible_member(
                archive,
                image,
                f"images/default/{image.name}",
                zipfile.ZIP_STORED,
            )
    temporary_archive.replace(archive_path)
    annotation_archive = bundle_root / "annotations.coco.zip"
    temporary_annotations = annotation_archive.with_name(f".{annotation_archive.name}.tmp")
    with zipfile.ZipFile(temporary_annotations, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_reproducible_member(
            archive,
            annotation_path,
            "annotations/instances_default.json",
            zipfile.ZIP_DEFLATED,
        )
    temporary_annotations.replace(annotation_archive)
    return {
        "images": len(rows),
        "annotations": len(coco_annotations),
        "previews_created": created,
        "archive": archive_path.as_posix(),
        "annotation_archive": annotation_archive.as_posix(),
    }


def _negative_document(rows, review_round, seed, title):
    ordered = sorted(
        rows,
        key=lambda row: stable_rank(seed, "negative-review", review_round, row["source_identity"]),
    )
    serialized = json.dumps(ordered, ensure_ascii=True).replace("<", "\\u003c")
    fields = json.dumps(NEGATIVE_EXPORT_FIELDS)
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:15px system-ui,sans-serif;margin:0;background:#eef1f4;color:#17202a}}header{{position:sticky;top:0;z-index:2;background:#fff;padding:12px 18px;border-bottom:1px solid #ccd2d9}}
#cards{{max-width:1300px;margin:auto;padding:14px;display:grid;gap:14px}}article{{background:#fff;padding:10px;border:2px solid #ccd2d9;border-radius:8px}}article.done{{border-color:#218739}}
.frame{{position:relative;background:#111;width:fit-content;max-width:100%;margin:auto}}img{{display:block;width:auto;max-width:100%;max-height:72vh}}svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}rect{{fill:none;stroke:#ff3344;stroke-width:4}}
button,input{{padding:8px;margin:4px}}button.selected{{background:#176b38;color:#fff}}.meta{{font:13px ui-monospace,monospace;overflow-wrap:anywhere;margin:6px 0}}
</style></head><body><header><strong>{title} — <span id="progress"></span></strong>
<label> Revisore <input id="reviewer" autocomplete="name"></label><button id="export">Esporta CSV</button>
<div>Decisioni indipendenti: non consultare l’export dell’altro revisore. I box rossi sono suggerimenti del modello, non ground truth.</div></header><main id="cards"></main><script>
const rows={serialized};const fields={fields};const round={json.dumps(review_round)};const key='phenocam-negative-review-'+round;let saved={{}};try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(_e){{saved={{}}}}
const reviewer=document.getElementById('reviewer');reviewer.value=localStorage.getItem(key+'-reviewer')||'';reviewer.oninput=()=>localStorage.setItem(key+'-reviewer',reviewer.value);
function update(){{document.getElementById('progress').textContent=Object.keys(saved).length+'/'+rows.length}}
for(const row of rows){{const card=document.createElement('article');const frame=document.createElement('div');frame.className='frame';const img=document.createElement('img');img.loading='lazy';img.src=row.preview_uri;img.onload=()=>{{svg.setAttribute('viewBox',`0 0 ${{img.naturalWidth}} ${{img.naturalHeight}}`)}};const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');for(const box of row.baseline_detections){{const rect=document.createElementNS(svg.namespaceURI,'rect');rect.setAttribute('x',box.x1);rect.setAttribute('y',box.y1);rect.setAttribute('width',box.x2-box.x1);rect.setAttribute('height',box.y2-box.y1);svg.append(rect)}}frame.append(img,svg);const meta=document.createElement('div');meta.className='meta';meta.textContent=row.source_identity+' · '+row.context;const buttons=document.createElement('div');for(const [value,label] of [['confirmed_negative','Negativo confermato'],['target_present','Target presente'],['uncertain','Incerto']]){{const button=document.createElement('button');button.textContent=label;button.onclick=()=>{{saved[row.source_identity]={{decision:value,note:''}};localStorage.setItem(key,JSON.stringify(saved));for(const child of buttons.children)child.classList.remove('selected');button.classList.add('selected');card.classList.add('done');update()}};if(saved[row.source_identity]?.decision===value)button.classList.add('selected');buttons.append(button)}}if(saved[row.source_identity])card.classList.add('done');card.append(frame,meta,buttons);document.getElementById('cards').append(card)}}update();
function cell(value){{const text=String(value??'');return /[",\\n\\r]/.test(text)?'"'+text.replaceAll('"','""')+'"':text}}
document.getElementById('export').onclick=()=>{{if(!reviewer.value.trim()){{alert('Inserisci il revisore');return}}if(Object.keys(saved).length!==rows.length){{alert('Completa tutte le immagini');return}}const now=new Date().toISOString();const lines=[fields.join(',')];for(const row of rows){{const decision=saved[row.source_identity];lines.push([row.source_identity,decision.decision,reviewer.value.trim(),now,round,decision.note].map(cell).join(','))}}const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='negative-review-'+round+'.csv';link.click();URL.revokeObjectURL(link.href)}};
</script></body></html>"""


def _build_negative_packets(openimages_rows, phenocam_rows, output_root, config):
    preview_root = Path(output_root) / "negative-review" / "images"
    packets = []
    for source_name, rows, identity_function in (
        ("open-images", openimages_rows, _openimages_identity),
        ("phenocam", phenocam_rows, _phenocam_identity),
    ):
        browser_rows = []
        for index, row in enumerate(rows, start=1):
            identity = identity_function(row)
            destination = preview_root / source_name / _bundle_name(identity, index)
            _materialize_preview(row, destination)
            browser_rows.append(
                {
                    "source_identity": identity,
                    "preview_uri": destination.resolve().as_uri(),
                    "baseline_detections": json.loads(row.get("baseline_detections_json", "[]") or "[]"),
                    "context": "; ".join(
                        value for value in (row.get("site_id", ""), row.get("timestamp", ""), row.get("triage_reason", "")) if value
                    ),
                }
            )
        rounds = ("a",) if source_name == "open-images" else ("a", "b")
        for review_round in rounds:
            path = Path(output_root) / "negative-review" / f"{source_name}-round-{review_round}.html"
            with atomic_text(path) as output:
                output.write(
                    _negative_document(
                        browser_rows,
                        f"{source_name}-{review_round}",
                        config["seed"],
                        f"Revisione negativi {source_name} — round {review_round.upper()}",
                    )
                )
            packets.append(path.as_posix())
    write_csv(
        Path(output_root) / "negative-review" / "expected-open-images.csv",
        ("source_identity",),
        ({"source_identity": _openimages_identity(row)} for row in openimages_rows),
    )
    write_csv(
        Path(output_root) / "negative-review" / "expected-phenocam.csv",
        ("source_identity",),
        ({"source_identity": _phenocam_identity(row)} for row in phenocam_rows),
    )
    return {"openimages": len(openimages_rows), "phenocam": len(phenocam_rows), "packets": packets}


def build_annotation_bundles(dataset_root, output_root, config):
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    openimages_selection = _read_csv(
        dataset_root / "workspace/sources/open-images/baseline-screened.csv",
        ("source_dataset", "source_subset", "source_id", "annotations_json", "local_path"),
    )
    openimages_review = _read_csv(
        dataset_root / "workspace/reviews/open-images/review.csv",
        ("source_identity", "manual_review_required", "candidate_kind"),
    )
    required = {row["source_identity"]: row for row in openimages_review if row["manual_review_required"] == "true"}
    selected_by_identity = {_openimages_identity(row): row for row in openimages_selection}
    openimages_positive = []
    openimages_negative = []
    for identity, review in required.items():
        row = dict(selected_by_identity[identity])
        row.update({"review_scope": review["review_scope"], "triage_reason": review["triage_reason"]})
        (openimages_positive if review["candidate_kind"] == "positive_review" else openimages_negative).append(row)
    openimages_positive.sort(key=_openimages_identity)
    openimages_negative.sort(key=_openimages_identity)

    phenocam = _read_csv(
        dataset_root / "workspace/reviews/phenocam/review.csv",
        ("source_dataset", "source_id", "provisional_role", "baseline_detections_json", "local_path"),
    )
    phenocam_positive = sorted(
        (row for row in phenocam if row["provisional_role"] == "positive"),
        key=_phenocam_identity,
    )
    phenocam_negative = sorted(
        (row for row in phenocam if row["provisional_role"] == "negative"),
        key=_phenocam_identity,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "openimages_positive": _build_coco_bundle(
            "open-images-positive", openimages_positive, _openimages_identity, _openimages_annotations, output_root, config
        ),
        "phenocam_positive": _build_coco_bundle(
            "phenocam-positive", phenocam_positive, _phenocam_identity, _phenocam_annotations, output_root, config
        ),
        "negative_review": _build_negative_packets(
            openimages_negative, phenocam_negative, output_root, config
        ),
    }
    supplemental_screened = dataset_root / "workspace/sources/open-images/supplemental-baseline-screened.csv"
    supplemental_review_path = dataset_root / "workspace/reviews/open-images-supplement/review.csv"
    if supplemental_screened.is_file() and supplemental_review_path.is_file():
        supplemental_rows = _read_csv(
            supplemental_screened,
            ("source_dataset", "source_subset", "source_id", "annotations_json", "local_path"),
        )
        supplemental_review = _read_csv(
            supplemental_review_path,
            ("source_identity", "manual_review_required", "review_scope", "triage_reason"),
        )
        supplemental_by_identity = {
            _openimages_identity(row): row for row in supplemental_rows
        }
        supplemental_positive = []
        for review in supplemental_review:
            if review["manual_review_required"] != "true":
                continue
            row = dict(supplemental_by_identity[review["source_identity"]])
            row.update(
                {
                    "review_scope": review["review_scope"],
                    "triage_reason": review["triage_reason"],
                }
            )
            supplemental_positive.append(row)
        supplemental_positive.sort(key=_openimages_identity)
        result["openimages_supplement_positive"] = _build_coco_bundle(
            "open-images-supplement-positive",
            supplemental_positive,
            _openimages_identity,
            _openimages_annotations,
            output_root,
            config,
        )
    with atomic_text(output_root / "statistics.json") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    return result


def import_negative_reviews(
    openimages_path,
    phenocam_first_path,
    phenocam_second_path,
    output_path,
    expected_dir=None,
):
    def read(path, expected_round):
        rows = _read_csv(path, NEGATIVE_EXPORT_FIELDS)
        by_identity = {}
        for row in rows:
            if row["source_identity"] in by_identity or row["review_round"] != expected_round:
                raise DatasetError(f"invalid or duplicate negative review: {path}")
            if row["decision"] not in {"confirmed_negative", "target_present", "uncertain"}:
                raise DatasetError(f"invalid negative decision: {path}")
            if not row["reviewer"] or not row["reviewed_at"]:
                raise DatasetError(f"unattributed negative review: {path}")
            by_identity[row["source_identity"]] = row
        return by_identity

    openimages = read(openimages_path, "open-images-a")
    first = read(phenocam_first_path, "phenocam-a")
    second = read(phenocam_second_path, "phenocam-b")
    if expected_dir is not None:
        expected_dir = Path(expected_dir)
        expected_openimages = {
            row["source_identity"]
            for row in _read_csv(expected_dir / "expected-open-images.csv", ("source_identity",))
        }
        expected_phenocam = {
            row["source_identity"]
            for row in _read_csv(expected_dir / "expected-phenocam.csv", ("source_identity",))
        }
        if set(openimages) != expected_openimages or set(first) != expected_phenocam:
            raise DatasetError("negative review export does not match its canonical packet")
    if set(first) != set(second):
        raise DatasetError("PhenoCam negative review rounds contain different images")
    output_rows = []
    for identity, row in sorted(openimages.items()):
        result = (
            "accepted_negative"
            if row["decision"] == "confirmed_negative"
            else "requires_resolution"
        )
        output_rows.append(
            {
                **row,
                "second_reviewer": "",
                "second_decision": "",
                "result": result,
            }
        )
    for identity in sorted(first):
        left, right = first[identity], second[identity]
        if left["reviewer"] == right["reviewer"]:
            raise DatasetError("PhenoCam negative reviews require distinct reviewers")
        result = "accepted_negative" if left["decision"] == right["decision"] == "confirmed_negative" else "requires_resolution"
        output_rows.append(
            {
                **left,
                "second_reviewer": right["reviewer"],
                "second_decision": right["decision"],
                "result": result,
            }
        )
    fields = NEGATIVE_EXPORT_FIELDS + ("second_reviewer", "second_decision", "result")
    write_csv(output_path, fields, output_rows)
    return {
        "rows": len(output_rows),
        "accepted_negatives": sum(row["result"] == "accepted_negative" for row in output_rows),
        "requires_resolution": sum(row["result"] == "requires_resolution" for row in output_rows),
    }


def _load_coco_export(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if Path(name).name.startswith("instances_") and name.endswith(".json")
            ]
            if len(candidates) != 1:
                raise DatasetError("CVAT COCO export must contain one instances JSON")
            try:
                return json.loads(archive.read(candidates[0]))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetError("CVAT COCO export JSON is unreadable") from error
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError("CVAT COCO export is unreadable") from error


def import_positive_coco(bundle_dir, export_path, output_path, annotator, reviewer, config):
    if not annotator.strip() or not reviewer.strip():
        raise DatasetError("positive annotation import requires annotator and reviewer")
    mappings = _read_csv(
        Path(bundle_dir) / "mapping.csv",
        ("source_identity", "bundle_file_name", "width", "height"),
    )
    by_file = {row["bundle_file_name"]: row for row in mappings}
    if len(by_file) != len(mappings):
        raise DatasetError("annotation bundle contains duplicate filenames")
    coco = _load_coco_export(export_path)
    compiled_ids = config["compiled_class_ids"]
    category_rows = coco.get("categories", [])
    try:
        category_ids = [int(row["id"]) for row in category_rows]
        category_names = [row["name"] for row in category_rows]
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError("CVAT export contains a malformed category") from error
    if (
        len(category_ids) != len(set(category_ids))
        or len(category_names) != len(set(category_names))
        or set(category_names) != set(compiled_ids)
    ):
        raise DatasetError("CVAT export categories do not match the dataset contract")
    categories = dict(zip(category_ids, category_names))
    images = {}
    for image in coco.get("images", []):
        try:
            name = Path(image["file_name"]).name
        except (KeyError, TypeError) as error:
            raise DatasetError("CVAT export contains a malformed image") from error
        if name in images or name not in by_file:
            raise DatasetError("CVAT export contains an unknown or duplicate image")
        mapping = by_file[name]
        if (
            int(image.get("width", -1)) != int(mapping["width"])
            or int(image.get("height", -1)) != int(mapping["height"])
        ):
            raise DatasetError("CVAT export image dimensions do not match its bundle")
        images[name] = image
    if set(images) != set(by_file):
        raise DatasetError("CVAT export does not contain every bundle image")
    annotations_by_image = {int(image["id"]): [] for image in images.values()}
    seen_boxes = set()
    for annotation in coco.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            values = tuple(float(value) for value in annotation["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("CVAT export contains a malformed box") from error
        if image_id not in annotations_by_image or category_id not in categories:
            raise DatasetError("CVAT export annotation references an unknown image or category")
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise DatasetError("CVAT export contains a malformed box")
        left, top, width, height = values
        image = next(item for item in images.values() if int(item["id"]) == image_id)
        mapping = by_file[Path(image["file_name"]).name]
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or left + width > int(mapping["width"]) + 1
            or top + height > int(mapping["height"]) + 1
        ):
            raise DatasetError("CVAT export contains an out-of-bounds box")
        class_name = categories[category_id]
        key = (image_id, class_name, *(round(value, 6) for value in values))
        if key in seen_boxes:
            raise DatasetError("CVAT export contains a duplicate box")
        seen_boxes.add(key)
        annotations_by_image[image_id].append(
            {
                "class_id": compiled_ids[class_name],
                "class_name": class_name,
                "bbox_xywh": values,
            }
        )
    imported_at = datetime.now(timezone.utc).isoformat()
    export_sha256 = sha256_file(export_path)
    output_rows = []
    for file_name, image in sorted(images.items()):
        mapping = by_file[file_name]
        annotations = annotations_by_image[int(image["id"])]
        output_rows.append(
            {
                "source_identity": mapping["source_identity"],
                "bundle_file_name": file_name,
                "review_status": "complete" if annotations else "rejected_no_targets",
                "annotation_count": len(annotations),
                "annotations_json": json.dumps(annotations, sort_keys=True, separators=(",", ":")),
                "annotator": annotator.strip(),
                "reviewer": reviewer.strip(),
                "imported_at": imported_at,
                "source_export_sha256": export_sha256,
            }
        )
    write_csv(output_path, POSITIVE_IMPORT_FIELDS, output_rows)
    return {
        "images": len(output_rows),
        "accepted": sum(row["review_status"] == "complete" for row in output_rows),
        "rejected_no_targets": sum(row["review_status"] == "rejected_no_targets" for row in output_rows),
        "annotations": sum(int(row["annotation_count"]) for row in output_rows),
    }
