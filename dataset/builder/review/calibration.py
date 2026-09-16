"""Review: calibration responsibility extracted without changing the data contract."""

import csv
import json
from pathlib import Path
from ..common import DatasetError, atomic_text, require_columns, sha256_file, write_csv
from ..embeddings import CALIBRATION_FIELDS


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
`dataset/workspace/reviews/sscd-calibration/sscd-calibration-reviewed.csv`.
Do not edit identities, paths, cosine values, or similarity bands.
"""
    with atomic_text(output_dir / "README.md") as output:
        output.write(instructions)
    return {"calibration_pairs": len(rows), "review_required": True}
