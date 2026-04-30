#!/usr/bin/env python3
"""
Gradio application for ArchAIaGPT, an archaeological assistant.
Provides a multimodal retrieval-augmented generation (RAG) interface
for researchers to query artifact databases using text and images.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
import uuid
from io import BytesIO
import base64
from typing import List, Optional

# Local imports for retrieval and pipeline logic
from retrieval.retriever import Retriever
import gradio as gr
from datasets import load_from_disk
from pipeline import ArchAIaGPT
from config import DATASET_PATH, IMAGES_ROOT, TOP_K, TEXT_WEIGHT
from utils.feedback_manager import FeedbackManager
from generation.factory import get_generator

# Global states for the pipeline and dataset
pipe: ArchAIaGPT = None
dataset = None                  
project_choices: list = []      
feedback_mgr = FeedbackManager()


def _normalize_image_path_list(raw_paths) -> List[str]:
    if raw_paths is None:
        return []
    if isinstance(raw_paths, list):
        return [str(p) for p in raw_paths if p]
    if isinstance(raw_paths, str):
        try:
            parsed = json.loads(raw_paths)
        except json.JSONDecodeError:
            return [raw_paths] if raw_paths.strip() else []
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed]
    return []


def _safe_json_load(raw_value):
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _opencontext_url(row):
    raw_uuid = str(row.get("uuid_hex", "") or "").strip()
    if not raw_uuid:
        return ""
    try:
        return f"https://opencontext.org/subjects/{uuid.UUID(raw_uuid)}"
    except (ValueError, TypeError):
        return ""


def _format_range(start, stop):
    if start and stop:
        return f"{start} to {stop}"
    return str(start or stop or "-")


def _format_coords(row):
    lon = row.get("longitude", "-")
    lat = row.get("latitude", "-")
    if lon == "-" and lat == "-":
        return "-"
    return f"[{lon}, {lat}]"


def _condition_notes(row):
    cond = str(row.get("recovered_condition", "-") or "-")
    rec_text = _safe_json_load(row.get("recovered_text_fields_json"))
    notes = rec_text.get("Condition Notes")
    if isinstance(notes, list) and notes:
        return ", ".join(str(n) for n in notes if n)
    return cond


def _artifact_html(idx: int, result) -> str:
    row = dataset[idx]
    title = result.label or row.get("label") or result.artifact_id or f"Artifact {idx}"
    oc_url = _opencontext_url(row)
    project_val = row.get("project_label") or row.get("project") or "-"
    class_val = row.get("item_class_label") or row.get("recovered_object_type") or "-"
    material_val = row.get("recovered_material") or "-"
    date_val = _format_range(row.get("earliest"), row.get("latest"))
    loc_val = _format_coords(row)
    cond_notes = _condition_notes(row)
    desc = result.description or row.get("description") or "No technical description available."

    images = load_artifact_images(idx)
    if images:
        image_html = "".join(
            f"<img class='artifact-img' src='data:image/jpeg;base64,{_to_base64(img)}' />"
            for img in images[:4]
        )
    else:
        image_html = "<div class='artifact-noimg'>No image available</div>"

    link_html = f"<a href='{oc_url}' target='_blank' rel='noopener noreferrer'>OpenContext</a>" if oc_url else "No OpenContext link"

    return f"""
    <div class="artifact-card">
      <div class="artifact-images">{image_html}</div>
      <div class="artifact-body">
        <div class="artifact-header">
          <div>
            <div class="artifact-title">{title}</div>
            <div class="artifact-subtitle">Score: {result.fused_score:.3f} | Artifact {result.idx}</div>
          </div>
          <div class="artifact-link">{link_html}</div>
        </div>
        <div class="artifact-meta">
          <div><b>Project:</b> {project_val}</div>
          <div><b>Class/Type:</b> {class_val}</div>
          <div><b>Material:</b> {material_val}</div>
          <div><b>Dates:</b> {date_val}</div>
          <div><b>Location:</b> {loc_val}</div>
          <div><b>Condition:</b> {cond_notes}</div>
        </div>
        <div class="artifact-desc">{desc}</div>
      </div>
    </div>
    """


def load_artifact_images(idx: int):
    """
    Attempts to load images associated with a specific artifact index.
    Supports both HuggingFace dataset 'Image' types and local file paths.
    """
    if dataset is None:
        return []
    
    row = dataset[idx]
    images = []

    # Check for HuggingFace Dataset Image columns
    from datasets import Image as HFImage
    hf_img_cols = [
        c for c in dataset.column_names
        if c.startswith("image_") and c != "image_paths"
        and isinstance(dataset.features.get(c), HFImage)
    ]
    
    if hf_img_cols:
        for col in sorted(hf_img_cols):
            img = row.get(col)
            if img:
                images.append(img)
                break
    
    # Fallback: Load from local filesystem using image_paths metadata
    elif "image_paths" in dataset.column_names:
        from PIL import Image as PILImage
        raw_paths = row.get("image_paths", "[]")
        
        try:
            rel_paths = _normalize_image_path_list(raw_paths)
            
            for rp in rel_paths:
                clean_path = rp.lstrip("/")
                abs_path = Path(IMAGES_ROOT) / clean_path

                if abs_path.exists():
                    images.append(PILImage.open(abs_path).convert("RGB"))
                else:
                    alt_path = Path(DATASET_PATH).parent / clean_path
                    if alt_path.exists():
                        images.append(PILImage.open(alt_path).convert("RGB"))

                # Limit to 4 images to prevent UI clutter
                if len(images) >= 4: 
                    break
        except Exception as e:
            # Silent fail for image loading to avoid crashing the search results
            pass

    return images

def search_fn(query, image_query, top_k, text_weight, project_filter, do_generate, embedding_model, generation_model):
    """
    The primary search handler for the Gradio interface.
    Orchestrates embedding selection, retrieval, and LLM generation.
    """
    if not (query and query.strip()) and image_query is None:
        return "Please enter a query or upload an image.", [], "", {}

    # Map model names from UI to internal identifiers
    emb_name = embedding_model.split(" (")[0]

    emb_map = {
        # "BM25": "bm25",
        "EmbeddingGemma-300m": "gemma",
        "CLIP": "clip",
        "Qwen3-VL-Embedding-2B": "qwen3-vl",
        # "e5-omni-3B": "e5-omni",
        # "VLM2Vec-Qwen2VL-2B": "vlm2vec"
    }
    emb_type = emb_map.get(emb_name, "clip")
    
    # Update the retriever if the user changed the embedding model
    if emb_type != pipe.retriever.model_type:
        print(f"Loading retriever with model: {emb_type}")
        pipe.retriever = Retriever(model_type=emb_type, device=pipe.device)

    # Generation model configuration
    gen_model_map = {
        "gpt-5-nano": "gpt-5-nano",
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "InternVL3-1B": "OpenGVLab/InternVL3-1B",
        "Ovis2-1B": "AIDC-AI/Ovis2-1B",
        "gemini-3-flash-preview": "gemini"
    }
    gen_key = gen_model_map.get(generation_model, generation_model)

    gen_override = None
    if gen_key:
        # Determine the backend provider based on the model name
        backend_provider = "openai" if "gpt" in gen_key.lower() else "gemini" if "gemini" in gen_key.lower() else "vllm"
        if gen_key.startswith("Qwen/"):
             backend_provider = "qwen3-vl"
        elif gen_key.startswith("OpenGVLab/"):
             backend_provider = "internvl3"
        elif gen_key.startswith("AIDC-AI/"):
             backend_provider = "ovis2"

        gen_override = get_generator(backend=backend_provider, model_name=gen_key, device=pipe.device)

    # Apply site-level filtering if selected
    filters = None
    if project_filter and project_filter != "All":
        filters = {"project": project_filter}

    # Execute the multimodal pipeline
    outputs = pipe.search(
        query=query.strip() if query else None,
        image_query=image_query,
        top_k=int(top_k),
        text_weight=float(text_weight),
        filters=filters,
        generate=bool(do_generate),
        generator_override=gen_override
    )

    results = outputs.results
    
    evidence_cards = []
    for r in results:
        evidence_cards.append(_artifact_html(r.idx, r))
    evidence_html = "\n".join(evidence_cards) if evidence_cards else "<div class='artifact-empty'>No artifact metadata found.</div>"

    details_list = []
    for i, r in enumerate(results, 1):
        lbl = r.label or f"Artifact {r.idx}"
        desc = r.description or "No technical description available."
        details_list.append(f"### {i}. {lbl}\n{desc}\n")

    details_md = "\n".join(details_list) if details_list else "No artifact metadata found."
    final_answer = outputs.answer if outputs.answer else "No generative response was produced."

    # Pack interaction context for feedback submission
    interaction_info = {
        "query": query,
        "configuration": {
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "top_k": top_k,
            "text_weight": text_weight
        },
        "retrieved_artifacts": [r.artifact_id for r in results],
        "generated_response": final_answer
    }

    return final_answer, evidence_html, details_md, interaction_info

def save_feedback_fn(interaction_info, rating, comment):
    """Persistence hook for saving user quality ratings."""
    feedback_mgr.save_feedback(
        query=interaction_info.get("query", ""),
        configuration=interaction_info.get("configuration", {}),
        retrieved_artifacts=interaction_info.get("retrieved_artifacts", []),
        generated_response=interaction_info.get("generated_response", ""),
        feedback=rating,
        feedback_text=comment
    )
    return "Thank you for your feedback."

def perform_4way_battle(query, image_query, top_k, text_weight, project_filter, do_generate, emb_a, emb_b, gen_a, gen_b):
    """
    Executes a side-by-side comparison of different embedding and generation configurations.
    """
    if gen_a == gen_b:
        return "Comparison requires distinct models.", "", "", "", [], "", {}

    # Matrix of 4 combinations: (Emb A, Gen A), (Emb A, Gen B), (Emb B, Gen A), (Emb B, Gen B)
    res_1, gallery, details, info_1 = search_fn(query, image_query, top_k, text_weight, project_filter, do_generate, emb_a, gen_a)
    res_2, _, _, info_2 = search_fn(query, image_query, top_k, text_weight, project_filter, do_generate, emb_a, gen_b)
    res_3, _, _, info_3 = search_fn(query, image_query, top_k, text_weight, project_filter, do_generate, emb_b, gen_a)
    res_4, _, _, info_4 = search_fn(query, image_query, top_k, text_weight, project_filter, do_generate, emb_b, gen_b)

    battle_context = {
        "query": query,
        "models": {"1": (emb_a, gen_a), "2": (emb_a, gen_b), "3": (emb_b, gen_a), "4": (emb_b, gen_b)},
        "responses": [res_1, res_2, res_3, res_4],
        "all_info": [info_1, info_2, info_3, info_4]
    }
    
    if emb_a == emb_b:
        return res_1, res_2, "(Same embedding model)", "(Same embedding model)", gallery, details, battle_context

    return res_1, res_2, res_3, res_4, gallery, details, battle_context

def save_battle_feedback_fn(battle_info, winner, comment):
    """Persistence hook for battle result submissions."""
    feedback_mgr.save_feedback(
        query=battle_info.get("query", ""),
        configuration={"mode": "battle", "models": battle_info.get("models")},
        retrieved_artifacts=[info["retrieved_artifacts"] for info in battle_info.get("all_info", [])],
        generated_response="; ".join(battle_info.get("responses", [])),
        feedback=winner,
        feedback_text=comment
    )
    return "Evaluation recorded."

def build_app() -> gr.Blocks:
    """Configures the Gradio UI layout and event bindings."""
    embedding_models = [
        # "BM25 (Text only)",
        "EmbeddingGemma-300m (Text only)",
        "CLIP (Multimodal)",
        "Qwen3-VL-Embedding-2B (Multimodal)",
        # "e5-omni-3B (Multimodal)",
        # "VLM2Vec-Qwen2VL-2B (Multimodal)"
    ]
    generation_models = [
        "gpt-5-nano", 
        "Qwen3-VL-2B-Instruct", 
        "InternVL3-1B", 
        "Ovis2-1B", 
        "gemini-3-flash-preview"
    ]

    with gr.Blocks(title="ArchAIaGPT — Archaeological RAG Assistant", css="""
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            font-size: 1.1rem;
            opacity: 0.78;
            margin-bottom: 1.5rem;
        }
        .artifact-card {
            border: 1px solid var(--border-color-primary, #e5e7eb);
            border-radius: 8px;
            padding: 16px;
            background-color: var(--background-fill-secondary, #fafafa);
        }
        .artifact-card + .artifact-card {
            margin-top: 20px;
        }
        .artifact-title {
            margin-top: 0;
            margin-bottom: 12px;
            color: var(--body-text-color, inherit);
            font-size: 1.25rem;
            font-weight: 700;
        }
        .artifact-images {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }
        .artifact-img {
            max-height: 200px;
            max-width: 300px;
            object-fit: contain;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .artifact-noimg {
            font-style: italic;
            opacity: 0.7;
            margin-bottom: 12px;
        }
        .artifact-meta {
            margin-bottom: 16px;
            font-size: 1em;
            line-height: 1.6;
            color: var(--body-text-color, inherit);
        }
        .artifact-meta a {
            color: inherit;
            text-decoration: underline;
        }
        .artifact-desc {
            margin-top: 10px;
            font-size: 0.95em;
            line-height: 1.5;
            color: var(--body-text-color, inherit);
        }
        .artifact-empty {
            padding: 12px 0;
            opacity: 0.8;
        }
    """) as app:
        gr.HTML("""
        <div class="main-title">ArchAIaGPT</div>
        <div class="subtitle">Retrieval-Augmented Archaeological Assistant</div>
        """)
        
        interaction_state = gr.State({})
        battle_state = gr.State({})

        with gr.Tabs():
            with gr.TabItem("Single Query"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_text = gr.Textbox(
                            label="Query",
                            placeholder="e.g. Describe pottery fragments from the Archaic period found in Sardis...",
                            lines=3,
                        )
                        query_img = gr.Image(label="Image Query (Optional)", type="pil")

                        with gr.Accordion("⚙️ Search Settings", open=False):
                            emb_model = gr.Dropdown(embedding_models, value="CLIP (Multimodal)", label="Embedding Model")
                            gen_model = gr.Dropdown(generation_models, value="gpt-5-nano", label="Generation Model")
                            do_generate = gr.Checkbox(value=True, label="Enable Generation")
                            k_slider = gr.Slider(1, 40, step=1, value=int(TOP_K), label="Top-K Artifacts")
                            w_slider = gr.Slider(0, 1, value=TEXT_WEIGHT, label="Text Weight")
                            proj_f = gr.Dropdown(["All"] + project_choices, value="All", label="Filter by Project")

                        submit_btn = gr.Button("🔍 Search", variant="primary")

                    with gr.Column(scale=2):
                        response_box = gr.Markdown("*Enter a question and click Search.*")

                        gr.Markdown("### Retrieved Artifacts")
                        gallery = gr.HTML(value="*Search results will appear here.*")

                        details = gr.Markdown("### Technical Details\n*Result descriptions will load here.*")

                        with gr.Row(visible=False) as feedback_row:
                            with gr.Group():
                                gr.Markdown("### Technical Evaluation")
                                with gr.Row():
                                    with gr.Column():
                                        rating_input = gr.Radio([str(i) for i in range(1, 11)], label="Accuracy (1-10)")
                                    with gr.Column():
                                        comment_input = gr.Textbox(label="Notes", lines=2)
                                f_submit = gr.Button("Submit Evaluation", variant="secondary")
                                f_status = gr.Markdown()

                submit_btn.click(
                    search_fn,
                    [query_text, query_img, k_slider, w_slider, proj_f, do_generate, emb_model, gen_model],
                    [response_box, gallery, details, interaction_state]
                ).then(lambda: gr.update(visible=True), None, feedback_row)

                f_submit.click(save_feedback_fn, [interaction_state, rating_input, comment_input], f_status)

            with gr.TabItem("Battle Mode"):
                with gr.Row():
                    with gr.Column(scale=1):
                        battle_text = gr.Textbox(label="Battle Query", lines=2)
                        battle_img = gr.Image(label="Visual Sample", type="pil")

                        with gr.Group():
                            gr.Markdown("### Arena Configuration")
                            emb_a_sel = gr.Dropdown(embedding_models, value="CLIP (Multimodal)", label="Embedding A")
                            emb_b_sel = gr.Dropdown(embedding_models, value="EmbeddingGemma-300m (Text only)", label="Embedding B")
                            gen_enable = gr.Checkbox(value=True, label="Enable Generation")

                            with gr.Row() as gen_settings:
                                gen_a_sel = gr.Dropdown(generation_models, value="gpt-5-nano", label="Generator A")
                                gen_b_sel = gr.Dropdown(generation_models, value="gpt-4o-mini", label="Generator B")

                            gen_enable.change(lambda x: gr.update(visible=x), inputs=[gen_enable], outputs=[gen_settings])

                        battle_run = gr.Button("Initialize Battle", variant="primary")

                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column():
                                output_a1 = gr.Markdown("### Model A1 (E_A, M_A)")
                                output_a2 = gr.Markdown("### Model A2 (E_A, M_B)")
                            with gr.Column():
                                output_b1 = gr.Markdown("### Model B1 (E_B, M_A)")
                                output_b2 = gr.Markdown("### Model B2 (E_B, M_B)")

                        gr.Markdown("### Retrieved Artifacts")
                        battle_gallery = gr.HTML(value="*Battle results will appear here.*")

                        battle_details = gr.Markdown("### Technical Details\n*Result descriptions will load here.*")

                        with gr.Group():
                            gr.Markdown("### Comparative Evaluation")
                            with gr.Row():
                                with gr.Column():
                                    winner_sel = gr.Radio(["A1", "A2", "B1", "B2", "Draw"], label="Preferred Result")
                                with gr.Column():
                                    battle_notes = gr.Textbox(label="Rationale", lines=2)
                            battle_submit = gr.Button("Record Comparison", variant="secondary")
                            battle_status = gr.Markdown()

                battle_run.click(
                    perform_4way_battle,
                    [battle_text, battle_img, k_slider, w_slider, proj_f, gen_enable, emb_a_sel, emb_b_sel, gen_a_sel, gen_b_sel],
                    [output_a1, output_a2, output_b1, output_b2, battle_gallery, battle_details, battle_state]
                )
                battle_submit.click(save_battle_feedback_fn, [battle_state, winner_sel, battle_notes], battle_status)

    return app

def init_globals():
    """Initializes global resources and shared models."""
    global pipe, dataset, project_choices
    
    # Load the artifact dataset metadata
    print(f"Initializing HF dataset: {DATASET_PATH}")
    dataset = load_from_disk(str(DATASET_PATH))
    
    projects_list = set()
    for i in range(len(dataset)):
        proj = dataset[i].get("project", "")
        if proj and proj.strip():
            projects_list.add(proj.strip())
    project_choices = sorted(list(projects_list))
    
    # Instantiate the primary search pipeline
    pipe = ArchAIaGPT(device="cuda" if torch.cuda.is_available() else "cpu")

# Cold-start initialization
init_globals()
demo = build_app()

def main():
    parser = argparse.ArgumentParser(description="ArchAIaGPT Web Application")
    parser.add_argument("--share", action="store_true", help="Launch with public link")
    parser.add_argument("--port", type=int, default=7861, help="Gradio server port")
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="stone"),
    )

if __name__ == "__main__":
    main()
