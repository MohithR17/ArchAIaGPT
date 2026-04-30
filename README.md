# ArchAIaGPT

ArchAIaGPT is a high-performance **Retrieval-Augmented Generation (RAG)** system designed for archaeological research. It allows researchers to query artifact databases using natural language, images, or multimodal combinations.

The system integrates several state-of-the-art retrieval models (CLIP, Gemma, Qwen3-VL, E5-Omni, VLM2Vec) with advanced LLM backends (GPT-5, Gemini, Qwen-VL) to provide grounded, analytical responses based on archaeological evidence.

---

## Key Features

*   **Multimodal Retrieval**: Query using complex text descriptions, artifact images, or a weighted combination of both.
*   **Plug-and-Play Encoders**: Support for multiple embedding backends including:
    *   **CLIP**: General purpose multimodal search.
    *   **Gemma**: High-precision text embedding for catalog data.
    *   **Qwen3-VL / VLM2Vec / E5-Omni**: Instruction-aware multimodal embeddings.
    *   **BM25**: Keyword-based search for exact matches.
*   **Battle Mode (Arena)**: Side-by-side evaluation of different embedding and generation configurations to benchmark model performance.
*   **Hierarchical Analytical Context**: Optimized for the *Archaia* dataset, retrieving across multiple levels of taxonomic and physical descriptions.
*   **Technical Evaluation**: Built-in feedback management system to record accuracy ratings and qualitative notes for continuous model improvement.

---

## Installation

### 1. Clone the Repository
```bash
git clone git@github.com:macabdul9/ArchAIaGPT.git
cd ArchAIaGPT
```

### 2. Setup Environment
We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management. Install `uv` first, then set up your environment:

```bash
# Install uv if you haven't already
curl -LsSf https://astral-sh.uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternatively using simple conda

```bash
conda create -n archaia_full python=3.11 pip -y
conda activate archaia_full
pip install -r requirements_clean.txt
```

Note: for models such as gemma you need to pass huggingface token

```bash
python -m pip install --user huggingface_hub
huggingface-cli login
```


### 3. API Configuration
Create a `.env` file in the root directory or set the following environment variables:
```bash
OPENAI_API_KEY=your_api_key_here
GEMINI_API_KEY=your_gemini_key_here  # Optional
```

---

## Project Structure

```text
ArchAIaGPT/
├── app.py                # Main Gradio application
├── pipeline.py           # RAG orchestration logic
├── config.py             # Global configuration and hyperparameters
├── requirements.txt      # Project dependencies
├── retrieval/            # Dual-index search and score fusion
├── embeddings/           # Encoder implementations and indexing scripts
├── generation/           # LLM factory and prompt management
├── scripts/              # Launch and utility scripts
└── indexes/              # (Git ignored) Local FAISS indexes and metadata
```

---

## Usage Guide

### 1. Building Vector Indexes
To initialize the system, you must build the embeddings for your dataset:
```bash
bash scripts/build_all_indexes.sh
```
This script will iterate through the configured models and generate FAISS/Pickle indexes.

### 2. Launching the Assistant
Run the web interface with hot-reloading enabled or via the standard launch script:
```bash
bash scripts/launch.sh
```
Access the UI at `http://localhost:7861`.

### 3. Standalone Embedding Calculation
For standalone research tasks, you can calculate specific embeddings without launching the full UI:
```bash
bash scripts/calculate_embeddings.sh \
  --text "Your artifact description" \
  --image_path "path/to/image.jpg"
```

---

## Development & Feedback
The system includes a **Feedback Manager** that logs user quality ratings to `feedback.jsonl` and `feedback.csv`. This data is used to fine-tune retrieval weights and refine prompt engineering.

### Codebase Standards
This project follows professional Python standards with type-hinted interfaces, modular component design, and grounded error handling.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
