"""Annotation: negative responsibility extracted without changing the data contract."""

import json
from pathlib import Path
from ..common import DatasetError, atomic_text, stable_rank, write_csv
from .preview import _materialize_preview
from .records import NEGATIVE_EXPORT_FIELDS, _bundle_name, _openimages_identity, _phenocam_identity, _read_csv


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
