#!/usr/bin/env python3

import json
import base64
from pathlib import Path
from io import BytesIO

from datasets import load_from_disk
from datasets import Image as HFImage
from PIL import Image

from pipeline import ArchAIaGPT

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# DATASET_PATH = "/home/mohithr/pw/storage/archaia_dataset_hf"
# IMAGES_ROOT = "/home/mohithr/pw/storage/dataset_v1"
DATASET_PATH="/home/mohithr/pw/storage/archaia_dataset_hf_v2"
IMAGES_ROOT="/home/urmid/archaia/"

QUERIES = [
    "Reconstructed Pottery Vessel",
    "Bes plaque",
    "Incense burner",
    "Loom weight",
    "Stone weight",
    "Coarse ware",
    "vessel neck",
    "figurine",
    "seal",
    "stamp seal",
    "bichrome sherd",
    "painted lid",
    "bulla",
    "statuette",
    "lithic",
    "greenstone axe",
    "ceramic lamp",
    "pestle",
    "tube",
    "glyptic",
    "stone vessel"
]

TOP_K = 5

# ─────────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────────
GROUND_TRUTH = {
    "Reconstructed Pottery Vessel": ("https://opencontext.org/subjects/b9cf166f-f66b-4d44-8cd1-292071753161", True),
    "Bes plaque": ("https://opencontext.org/subjects/6a43e6ff-1ecc-43ec-88df-d0b91ed1530e", False),
    "Incense burner": ("https://opencontext.org/media/ee7b6e1e-7e6a-46d9-aa12-0ec1857df0e9", False),
    "Loom weight": ("https://opencontext.org/subjects/3b9b8bbb-c037-455b-64ab-0f9d68e706d4", False),
    "Stone weight": ("https://opencontext.org/subjects/090d74c0-d96b-4e85-4821-9609c7d62478", False),
    "Coarse ware": ("https://opencontext.org/subjects/de16cd59-acf3-4019-b83c-f5d535a85022", True),
    "vessel neck": ("https://opencontext.org/subjects/7c982900-0821-4b7e-9ad1-5130a9d4004b", False),
    "figurine": ("https://opencontext.org/subjects/f2df5fd8-0373-4d02-a830-87175fb3165e", False),
    "seal": ("https://opencontext.org/subjects/7ecc5209-85d2-4b34-b9d5-6fd072cc7995", False),
    "stamp seal": ("https://opencontext.org/subjects/90696d5c-9051-6e67-bc00-a55c4079def7", True),
    "bichrome sherd": ("https://opencontext.org/subjects/22d16b37-8141-4b1c-8a1a-bf2ed7941034", True),
    "painted lid": ("https://opencontext.org/subjects/5d59828f-3819-474b-3fd5-c6da3ffadc1b", True),
    "bulla": ("https://opencontext.org/subjects/f127f29d-5a89-45c1-9dfa-f62d989ad13b", True),
    "statuette": ("https://opencontext.org/subjects/585ef91c-5ec4-4a20-cd7b-bec18b547420", False),
    "lithic": ("https://opencontext.org/subjects/e6f0a62c-b093-48c1-91d7-e24a6df8eb57", True),
    "greenstone axe": ("https://opencontext.org/subjects/9162d650-455c-a015-8bba-78aa7240baf2", True),
    "ceramic lamp": ("https://opencontext.org/media/ff39c40a-8992-4ea4-88f6-b1a3fe626b16", False),
    "pestle": ("https://opencontext.org/subjects/50e877bb-c026-4495-bd1f-ae699bb75309", True),
    "tube": ("https://opencontext.org/subjects/b9191b43-13e8-4fea-aa89-5c675d45ab3d", True),
    "glyptic": ("https://opencontext.org/media/01eff193-3e32-43c6-a7ee-09b1495594f6", False),
    "stone vessel": ("https://opencontext.org/media/583e0742-0ca1-4956-6da4-4b2ffbb269b9", True),
}

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
ds = load_from_disk(DATASET_PATH)["train"]

# ─────────────────────────────────────────────
# LOAD PIPELINES
# ───────────────────────────────────────────── 
def load_pipeline(encoder_type):
    base = Path(__file__).resolve().parent
    idx_dir = base / ("indexes_qwen" if encoder_type == "qwen" else "indexes")

    return ArchAIaGPT(
        text_index_path=str(idx_dir / "text.faiss"),
        image_index_path=str(idx_dir / "image.faiss"),
        meta_path=str(idx_dir / "metadata.jsonl"),
        encoder_type=encoder_type,
        device="cuda",
    )

pipe_clip = load_pipeline("clip")
pipe_qwen = load_pipeline("qwen")

# ─────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────
def load_image(idx):
    row = ds[idx]

    # Try HF image columns first
    for col in ds.column_names:
        if col.startswith("image_") and col != "image_paths":
            if isinstance(ds.features.get(col), HFImage):
                img = row.get(col)
                if img:
                    return img

    # Fallback to image_paths
    try:
        raw = row.get("image_paths", [])
        paths = json.loads(raw) if isinstance(raw, str) else raw
    except:
        return None

    for p in paths:
        full = Path(IMAGES_ROOT) / p
        if full.exists():
            try:
                return Image.open(full).convert("RGB")
            except:
                pass

    return None

def to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────
# HTML HEADER
# ─────────────────────────────────────────────
html = ["""
<html>
<head>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
h1 { border-bottom: 2px solid black; padding-bottom: 5px; }
.row { display: flex; gap: 20px; margin-bottom: 40px; }
.col { width: 50%; }
.card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 12px;
    background: #fafafa;
}
img {
    max-width: 200px;
    max-height: 200px;
    display: block;
    margin-bottom: 8px;
}
.meta { font-size: 0.9em; line-height: 1.4; }
</style>
</head>
<body>
"""]

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
for q in QUERIES:
    link, _ = GROUND_TRUTH.get(q, ("#", False))

    html.append(f"""
    <h1>
        Query: {q}
    </h1>
    <p>
        <b>Ground Truth:</b>
        <a href="{link}" target="_blank">{link}</a>
    </p>
    """)

    out_clip = pipe_clip.search(query=q, top_k=TOP_K, generate=False)
    out_qwen = pipe_qwen.search(query=q, top_k=TOP_K, generate=False)

    html.append("<div class='row'>")

    # CLIP
    html.append("<div class='col'><h2>CLIP</h2>")
    for r in out_clip.results:
        img = load_image(r.idx)

        if img:
            b64 = to_base64(img)
            img_tag = f"<img src='data:image/jpeg;base64,{b64}'/>"
        else:
            img_tag = "<p>No image</p>"

        import uuid
        try:
            ds_row = ds[r.idx]
            raw_uuid = ds_row.get("uuid_hex", "")
            uuid_val = str(uuid.UUID(raw_uuid)) if raw_uuid else ""
            
            project_val = str(ds_row.get("project_label", "-"))
            material_val = str(ds_row.get("recovered_material", "-"))
            class_val = str(ds_row.get("item_class_label", "-"))
            date_val = f"{ds_row.get('earliest')} to {ds_row.get('latest')}"
            loc_val = f"[{ds_row.get('longitude')}, {ds_row.get('latitude')}]"
            
            rec_text = ds_row.get("recovered_text_fields_json")
            cond_notes = str(ds_row.get("recovered_condition", "-"))
            if rec_text:
                try:
                    rtj = json.loads(rec_text) if isinstance(rec_text, str) else rec_text
                    if isinstance(rtj, dict) and "Condition Notes" in rtj:
                        cond_notes = ", ".join(rtj["Condition Notes"])
                except:
                    pass
        except:
            uuid_val = ""
            project_val = material_val = class_val = date_val = loc_val = cond_notes = "-"

        oc_link = f"https://opencontext.org/subjects/{uuid_val}" if uuid_val else "#"
        link_html = f"<a href='{oc_link}' target='_blank'>OpenContext Link</a>" if uuid_val else "No Link"

        title = (r.label or r.artifact_id or "")
        match = q.lower() in title.lower()
        title_color = "green" if match else "black"

        html.append(f"""
        <div class="card">
            {img_tag}
            <b style="color:{title_color}">{title}</b>
            <div class="meta">
                <b>Link:</b> {link_html}<br>
                <b>Project:</b> {project_val}<br>
                <b>Class/Type:</b> {class_val}<br>
                <b>Material:</b> {material_val}<br>
                <b>Dates:</b> {date_val}<br>
                <b>Location:</b> {loc_val}<br>
                <b>Condition:</b> {cond_notes}<br>
                <b>Description:</b> {r.description or "-"}<br>
                Score: {r.fused_score:.3f}
            </div>
        </div>
        """)

    html.append("</div>")

    # QWEN
    html.append("<div class='col'><h2>QWEN</h2>")
    for r in out_qwen.results:
        img = load_image(r.idx)

        if img:
            b64 = to_base64(img)
            img_tag = f"<img src='data:image/jpeg;base64,{b64}'/>"
        else:
            img_tag = "<p>No image</p>"

        import uuid
        try:
            ds_row = ds[r.idx]
            raw_uuid = ds_row.get("uuid_hex", "")
            uuid_val = str(uuid.UUID(raw_uuid)) if raw_uuid else ""
            
            project_val = str(ds_row.get("project_label", "-"))
            material_val = str(ds_row.get("recovered_material", "-"))
            class_val = str(ds_row.get("item_class_label", "-"))
            date_val = f"{ds_row.get('earliest')} to {ds_row.get('latest')}"
            loc_val = f"[{ds_row.get('longitude')}, {ds_row.get('latitude')}]"
            
            rec_text = ds_row.get("recovered_text_fields_json")
            cond_notes = str(ds_row.get("recovered_condition", "-"))
            if rec_text:
                try:
                    rtj = json.loads(rec_text) if isinstance(rec_text, str) else rec_text
                    if isinstance(rtj, dict) and "Condition Notes" in rtj:
                        cond_notes = ", ".join(rtj["Condition Notes"])
                except:
                    pass
        except:
            uuid_val = ""
            project_val = material_val = class_val = date_val = loc_val = cond_notes = "-"

        oc_link = f"https://opencontext.org/subjects/{uuid_val}" if uuid_val else "#"
        link_html = f"<a href='{oc_link}' target='_blank'>OpenContext Link</a>" if uuid_val else "No Link"

        title = (r.label or r.artifact_id or "")
        match = q.lower() in title.lower()
        title_color = "green" if match else "black"

        html.append(f"""
        <div class="card">
            {img_tag}
            <b style="color:{title_color}">{title}</b>
            <div class="meta">
                <b>Link:</b> {link_html}<br>
                <b>Project:</b> {project_val}<br>
                <b>Class/Type:</b> {class_val}<br>
                <b>Material:</b> {material_val}<br>
                <b>Dates:</b> {date_val}<br>
                <b>Location:</b> {loc_val}<br>
                <b>Condition:</b> {cond_notes}<br>
                <b>Description:</b> {r.description or "-"}<br>
                Score: {r.fused_score:.3f}
            </div>
        </div>
        """)

    html.append("</div>")
    html.append("</div>")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
html.append("</body></html>")

with open("report.html", "w") as f:
    f.write("\n".join(html))

print("✅ Saved to report.html")