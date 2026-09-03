"""Write a local, read-only visual preview of proposed COCO annotations."""

import json
from pathlib import Path

from ..common import atomic_text


def write_preview(path, images, annotations, categories, image_prefix="images/default/"):
    category_names = {item["id"]: item["name"] for item in categories}
    by_image = {}
    for item in annotations:
        by_image.setdefault(item["image_id"], []).append({
            "bbox": item["bbox"], "class_name": category_names[item["category_id"]],
        })
    rows = [{
        "name": item["file_name"], "width": item["width"], "height": item["height"],
        "boxes": by_image.get(item["id"], []),
    } for item in images]
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    prefix = json.dumps(str(image_prefix))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>PhenoCam proposal review</title><style>
body{{margin:1rem;background:#111;color:#eee;font:14px system-ui}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}figure{{margin:0;background:#222;padding:.5rem}}.frame{{position:relative}}img,svg{{display:block;width:100%;height:auto}}svg{{position:absolute;inset:0}}rect{{fill:none;stroke:#ff3b30;stroke-width:3}}text{{fill:#fff;stroke:#000;stroke-width:3;paint-order:stroke;font-size:18px}}figcaption{{padding-top:.4rem;overflow-wrap:anywhere}}
</style></head><body><h1>Model proposals — mandatory human review</h1>
<p>Predictions are suggestions, not ground truth.</p><main id="items"></main><script>
const rows={payload},prefix={prefix},ns='http://www.w3.org/2000/svg',main=document.getElementById('items');
for(const row of rows){{const figure=document.createElement('figure'),frame=document.createElement('div');frame.className='frame';const image=document.createElement('img');image.loading='lazy';image.src=prefix+row.name;const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${{row.width}} ${{row.height}}`);for(const item of row.boxes){{const [x,y,w,h]=item.bbox,rect=document.createElementNS(ns,'rect');rect.setAttribute('x',x);rect.setAttribute('y',y);rect.setAttribute('width',w);rect.setAttribute('height',h);const label=document.createElementNS(ns,'text');label.setAttribute('x',x+2);label.setAttribute('y',Math.max(18,y+18));label.textContent=item.class_name;svg.append(rect,label)}}const caption=document.createElement('figcaption');caption.textContent=row.name+' · '+row.boxes.length+' proposals';frame.append(image,svg);figure.append(frame,caption);main.append(figure)}}
</script></body></html>"""
    with atomic_text(Path(path)) as output:
        output.write(document)
