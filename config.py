"""
Central configuration module for ArchAIaGPT.
Defines paths, model identifiers, and hyperparameters for retrieval and generation.
"""

from pathlib import Path

# Filesystem Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path("/home/mohithr/pw/storage/archaia_dataset_hf_v3")
IMAGES_ROOT = Path("/home/urmid/archaia/")

INDEX_DIR = PROJECT_ROOT / "indexes"
TEXT_INDEX = INDEX_DIR / "text.faiss"
IMAGE_INDEX = INDEX_DIR / "image.faiss"
META_STORE = INDEX_DIR / "metadata.jsonl"

# Embedding Models
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# Note: clip-vit-base-patch32 projects to a 512-dimensional vector space.
CLIP_DIM = 512

# Retrieval Settings
TOP_K = 8
TEXT_WEIGHT = 0.5
IMAGE_WEIGHT = 0.5

# Metadata fields used for text embedding generation
TEXT_FIELDS = [
    "description",
    "level_1_description",
    "level_2_description",
    "level_3_description",
    "level_4_description",
    "level_5_description",
]

# LLM Selection & Inference Parameters
# Compatible with local vLLM instances and OpenAI cloud APIs.
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OPENAI_MODEL = "gpt-5-nano"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

# Default system instructions for grounding the LLM
SYSTEM_PROMPT = (
    "You are ArchAIaGPT, an expert archaeological assistant. "
    "Use the retrieved artifact records to answer the user's query precisely. "
    "Do not hallucinated details outside provided context. Always cite specific artifact IDs."
)
