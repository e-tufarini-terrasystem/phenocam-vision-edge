"""Create auditable human-review sheets and a COCO annotation packet."""

import csv
import json
import math
from pathlib import Path

from .baseline import BASELINE_FIELDS
from .common import DatasetError, atomic_text, require_columns, sha256_file, stable_rank, write_csv
from .embeddings import CALIBRATION_FIELDS
from .selection import SELECTION_FIELDS


REVIEW_FIELDS = (
    "review_order",
    "triage_priority",
    "triage_reason",
    "source_identity",
    "local_path",
    "candidate_kind",
    "compiled_classes",
    "primary_stratum",
    "manual_review_required",
    "review_scope",
    "review_status",
    "decision",
    "reviewer",
    "reviewed_at",
    "corrected_annotations_json",
    "note",
)


def _identity(row):
    return f"{row['source_dataset']}:{row['source_subset']}:{row['source_id']}"


def _triage(row, required):
    if not required or "baseline_detection_count" not in row:
        return "", ""
    detected = int(row["baseline_detection_count"] or 0) > 0
    if row["candidate_kind"] == "negative_review" and detected:
        return "0", "negative_model_conflict"
    if row["candidate_kind"] == "positive_review" and not detected:
        return "0", "positive_model_miss"
    if row.get("baseline_full_crop_disagreement") == "true":
        return "1", "full_crop_disagreement"
    return "2", "required_source_review"


def create_openimages_review_packet(selection_path, output_dir, config, policy=None):
    with Path(selection_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, SELECTION_FIELDS, selection_path)
        rows = list(reader)
        baseline_fields = tuple(field for field in BASELINE_FIELDS if field in reader.fieldnames)
    positives = [row for row in rows if row["candidate_kind"] == "positive_review"]
    negatives = [row for row in rows if row["candidate_kind"] == "negative_review"]
    rare = {
        _identity(row)
        for row in positives
        if row["primary_stratum"] == "rare_environment_positive"
    }
    policy = policy or {
        "require_all_rare_frames": True,
        "reviewed_box_fraction": 0.10,
    }
    required = {_identity(row) for row in negatives}
    if policy["require_all_rare_frames"]:
        required.update(rare)
    model_conflicts = {
        _identity(row)
        for row in rows
        if baseline_fields
        and (
            (
                row["candidate_kind"] == "positive_review"
                and int(row["baseline_detection_count"] or 0) == 0
            )
            or (
                row["candidate_kind"] == "negative_review"
                and int(row["baseline_detection_count"] or 0) > 0
            )
        )
    }
    required.update(model_conflicts)
    reviewed_box_target = math.ceil(
        float(policy["reviewed_box_fraction"])
        * sum(len(json.loads(row["annotations_json"])) for row in positives)
    )
    reviewed_boxes = sum(
        len(json.loads(row["annotations_json"]))
        for row in positives
        if _identity(row) in required
    )
    remaining = sorted(
        (row for row in positives if _identity(row) not in required),
        key=lambda row: stable_rank(config["seed"], "box-review", _identity(row)),
    )
    for row in remaining:
        if reviewed_boxes >= reviewed_box_target:
            break
        required.add(_identity(row))
        reviewed_boxes += len(json.loads(row["annotations_json"]))

    review_rows = []
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    for image_number, row in enumerate(rows, start=1):
        identity = _identity(row)
        is_required = identity in required
        scopes = []
        if row["candidate_kind"] == "negative_review":
            scopes.append("independent_negative_verification")
        if identity in rare and policy["require_all_rare_frames"]:
            scopes.append("rare_class_and_box_review")
        elif is_required:
            scopes.append("sampled_class_and_box_review")
        triage_priority, triage_reason = _triage(row, is_required)
        review_row = {
                "review_order": "",
                "triage_priority": triage_priority,
                "triage_reason": triage_reason,
                "source_identity": identity,
                "local_path": row["local_path"],
                "candidate_kind": row["candidate_kind"],
                "compiled_classes": row["compiled_classes"],
                "primary_stratum": row["primary_stratum"],
                "manual_review_required": str(is_required).lower(),
                "review_scope": ";".join(scopes),
                "review_status": "pending" if is_required else "not_required",
                "decision": "" if is_required else "source_annotation_retained",
                "reviewer": "",
                "reviewed_at": "",
                "corrected_annotations_json": "",
                "note": "",
            }
        review_row.update({field: row[field] for field in baseline_fields})
        review_rows.append(review_row)
        if not is_required:
            continue
        width, height = int(row["width"]), int(row["height"])
        coco_images.append(
            {
                "id": image_number,
                "file_name": row["local_path"],
                "width": width,
                "height": height,
                "source_identity": identity,
                "review_scope": scopes,
            }
        )
        for annotation in json.loads(row["annotations_json"]):
            left = float(annotation["xmin"]) * width
            top = float(annotation["ymin"]) * height
            box_width = (float(annotation["xmax"]) - float(annotation["xmin"])) * width
            box_height = (float(annotation["ymax"]) - float(annotation["ymin"])) * height
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_number,
                    "category_id": int(annotation["class_id"]),
                    "bbox": [left, top, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "attributes": {
                        "source_class": annotation["source_class"],
                        "source_label_id": annotation["source_label_id"],
                        "occlusion": bool(annotation["occlusion"]),
                        "truncation": bool(annotation["truncation"]),
                    },
                }
            )
            annotation_id += 1

    review_rows.sort(
        key=lambda row: (
            row["manual_review_required"] != "true",
            int(row["triage_priority"] or 9),
            stable_rank(config["seed"], "review-order", row["source_identity"]),
        )
    )
    for review_order, row in enumerate(review_rows, start=1):
        row["review_order"] = review_order

    output_dir = Path(output_dir)
    write_csv(output_dir / "review.csv", REVIEW_FIELDS + baseline_fields, review_rows)
    category_ids = config["compiled_class_ids"]
    coco = {
        "info": {
            "description": "Open Images provisional selection review packet",
            "operational_validation": "pending",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 2.0",
                "url": "https://creativecommons.org/licenses/by/2.0/",
            }
        ],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "target"}
            for name, class_id in sorted(category_ids.items(), key=lambda item: item[1])
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    with atomic_text(output_dir / "annotations.coco.json") as output:
        json.dump(coco, output, indent=2, sort_keys=True)
        output.write("\n")
    instructions = """# Open Images review packet

Only rows with `manual_review_required=true` require a human decision.

- Verify every negative independently and reject it if any live target is visible.
- Verify every rare-class box and every sampled box for class, tight visible extent,
  occlusion, truncation, and frame-level target completeness.
- Use `decision=accept`, `correct`, or `reject`; set reviewer and an ISO-8601
  timestamp. Put corrected complete annotations in `corrected_annotations_json`.
- A model prediction is never sufficient evidence for acceptance or rejection.
- When present, `baseline_*` columns are suggestions for sorting review work;
  they never replace inspection of the source image and its annotations.
- Work in `review_order`: priority 0 contains source/model conflicts, priority 1
  contains full-frame/crop disagreements, and priority 2 is the remaining
  mandatory source review.
- Do not change source identity, provenance, license, or selection fields.

Duplicate-pair and SSCD-calibration reviews are separate mandatory queues.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {
        "selected_frames": len(rows),
        "manual_review_frames": len(required),
        "negative_review_frames": len(negatives),
        "rare_review_frames": len(rare),
        "require_all_rare_frames": bool(policy["require_all_rare_frames"]),
        "model_conflict_review_frames": len(model_conflicts),
        "reviewed_box_target": reviewed_box_target,
        "boxes_in_required_frames": reviewed_boxes,
    }


def create_sscd_calibration_packet(calibration_path, output_dir):
    with Path(calibration_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require_columns(reader, CALIBRATION_FIELDS, calibration_path)
        rows = list(reader)
    if len(rows) < 200:
        raise DatasetError("SSCD calibration packet requires at least 200 pairs")
    output_dir = Path(output_dir)
    write_csv(output_dir / "review.csv", CALIBRATION_FIELDS, rows)
    browser_rows = []
    for row in rows:
        output = dict(row)
        output["left_uri"] = Path(row["left_path"]).resolve().as_uri()
        output["right_uri"] = Path(row["right_path"]).resolve().as_uri()
        output["exact_file_match"] = (
            sha256_file(row["left_path"]) == sha256_file(row["right_path"])
        )
        browser_rows.append(output)
    serialized = json.dumps(browser_rows, ensure_ascii=True).replace("<", "\\u003c")
    fields = json.dumps(CALIBRATION_FIELDS)
    document = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisione calibrazione SSCD</title>
<style>
body{{font:15px system-ui,sans-serif;margin:0;background:#f4f5f7;color:#18202a}}
header{{position:sticky;top:0;z-index:2;background:#fff;padding:12px 18px;border-bottom:1px solid #ccd2d9}}
header input{{padding:7px;width:220px}} button{{padding:8px 12px;margin:3px;cursor:pointer}}
#pairs{{display:grid;grid-template-columns:1fr;gap:18px;padding:16px;max-width:1700px;margin:auto}}
.pair{{background:#fff;border:2px solid #d8dde3;border-radius:8px;padding:10px}}
.pair.done{{border-color:#27864b}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
figure{{margin:0;min-width:0}} figure a{{display:block;background:#111}}
img{{display:block;width:100%;height:min(58vh,620px);object-fit:contain;background:#111;cursor:zoom-in}}
figcaption{{font:13px ui-monospace,monospace;padding:7px;background:#eef1f4;overflow-wrap:anywhere}}
.meta{{font-size:13px;overflow-wrap:anywhere;margin:7px 0}} .exact{{font-weight:800;color:#b42318}}
.duplicate.selected{{background:#b42318;color:#fff}} .distinct.selected{{background:#167647;color:#fff}}
@media(max-width:850px){{.images{{grid-template-columns:1fr}}img{{height:55vh}}}}
</style>
</head>
<body>
<header>
  <strong>Calibrazione SSCD — <span id="progress">0/{len(rows)}</span></strong>
  <label> Revisore <input id="reviewer" autocomplete="name"></label>
  <button id="export">Esporta CSV completato</button>
  <div>Segna “duplicato” solo se le due immagini rappresentano la stessa immagine/copia, non soltanto la stessa scena fissa.</div>
  <div>Filename o timestamp diversi non bastano. Se “file identico” è SÌ, scegli “Duplicato”. Se è NO, confronta il contenuto: può comunque essere una copia trasformata. Clicca un’immagine per aprirla a piena risoluzione.</div>
</header>
<main id="pairs"></main>
<script>
const rows={serialized};
const fields={fields};
const storageKey='phenocam-sscd-calibration-v1';
let saved=[];try{{saved=JSON.parse(localStorage.getItem(storageKey)||'[]');}}catch(_error){{saved=[];}}
const decisions=Array.from({{length:rows.length}},(_,index)=>saved[index]??null);
const root=document.getElementById('pairs');
function update(){{document.getElementById('progress').textContent=`${{decisions.filter(Boolean).length}}/${{rows.length}}`;}}
rows.forEach((row,index)=>{{
  const card=document.createElement('section'); card.className='pair';
  const images=document.createElement('div'); images.className='images';
  for(const side of ['left','right']){{
    const frame=document.createElement('figure');const link=document.createElement('a');link.href=row[side+'_uri'];link.target='_blank';link.rel='noopener';
    const img=document.createElement('img');img.loading='lazy';img.src=row[side+'_uri'];img.alt=row[side+'_identity'];
    const caption=document.createElement('figcaption');caption.textContent=row[side+'_identity'];
    link.appendChild(img);frame.append(link,caption);images.appendChild(frame);
  }}
  const meta=document.createElement('div');meta.className='meta';meta.textContent=`#${{index+1}} · cosine=${{row.embedding_cosine}} · ${{row.similarity_band}} · file identico: ${{row.exact_file_match?'SÌ':'NO'}}`;if(row.exact_file_match)meta.classList.add('exact');
  const duplicate=document.createElement('button');duplicate.className='duplicate';duplicate.textContent='Duplicato';
  const distinct=document.createElement('button');distinct.className='distinct';distinct.textContent='Distinto';
  function choose(value){{decisions[index]=value;localStorage.setItem(storageKey,JSON.stringify(decisions));duplicate.classList.toggle('selected',value==='true');distinct.classList.toggle('selected',value==='false');card.classList.toggle('done',Boolean(value));update();}}
  duplicate.onclick=()=>choose('true'); distinct.onclick=()=>choose('false');
  card.append(images,meta,duplicate,distinct);root.appendChild(card);
  if(decisions[index])choose(decisions[index]);
}});
const reviewerInput=document.getElementById('reviewer');reviewerInput.value=localStorage.getItem(storageKey+'-reviewer')||'';reviewerInput.oninput=()=>localStorage.setItem(storageKey+'-reviewer',reviewerInput.value);
function csvCell(value){{const text=String(value??'');return /[\",\\n\\r]/.test(text)?'\"'+text.replaceAll('\"','\"\"')+'\"':text;}}
document.getElementById('export').onclick=()=>{{
  const reviewer=document.getElementById('reviewer').value.trim();
  if(!reviewer){{alert('Inserisci il nome del revisore.');return;}}
  if(decisions.some(value=>!value)){{alert('Completa tutte le coppie.');return;}}
  const reviewedAt=new Date().toISOString();
  const output=rows.map((row,index)=>({{...row,review_status:'complete',reviewer,reviewed_at:reviewedAt,is_copy:decisions[index]}}));
  const lines=[fields.join(','),...output.map(row=>fields.map(field=>csvCell(row[field])).join(','))];
  const blob=new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='sscd-calibration-reviewed.csv';link.click();URL.revokeObjectURL(link.href);
}};
</script>
</body>
</html>
"""
    with atomic_text(output_dir / "index.html") as output:
        output.write(document)
    instructions = """# SSCD calibration review

Open `index.html`, enter the reviewer name, and classify all 200 pairs.

- `Duplicato`: same source image or a transformed/cropped copy with no independent event.
- `Distinto`: merely similar content, repeated fixed-camera background, or a different event.

Export the completed CSV and place it at
`dataset/work/review/sscd-calibration/sscd-calibration-reviewed.csv`.
Do not edit identities, paths, cosine values, or similarity bands.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {"calibration_pairs": len(rows), "review_required": True}
