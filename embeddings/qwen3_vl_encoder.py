import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoImageProcessor, AutoTokenizer

from .base_encoder import BaseEncoder

_DEFAULT_TEXT_INSTRUCTION = "Represent the given text for retrieval"
_DEFAULT_IMAGE_INSTRUCTION = "Represent the given image for retrieval"
_DEFAULT_QUERY_INSTRUCTION = "Retrieve relevant content matching the user query"

_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{content}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


class Qwen3VLEncoder(BaseEncoder):
    """Qwen3-VL embedding encoder using transformers, not vLLM."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
        max_length: int = 8192,
        text_instruction: str = _DEFAULT_TEXT_INSTRUCTION,
        image_instruction: str = _DEFAULT_IMAGE_INSTRUCTION,
        query_instruction: str = _DEFAULT_QUERY_INSTRUCTION,
    ):
        super().__init__(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.text_instruction = text_instruction
        self.image_instruction = image_instruction
        self.query_instruction = query_instruction

        print(f"[Qwen3VLEncoder] Loading {model_path} on {self.device} ...")

        load_kwargs = dict(
            dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.tokenizer.padding_side = "left"
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.image_token = getattr(self.tokenizer, "image_token", "<|image_pad|>")
        self.vision_start_token = getattr(self.tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(self.tokenizer, "vision_end_token", "<|vision_end|>")

        cfg = self.model.config
        if hasattr(cfg, "hidden_size"):
            self.dim = cfg.hidden_size
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.dim = cfg.text_config.hidden_size
        else:
            self.dim = 4096

        print(f"[Qwen3VLEncoder] Ready. Embedding dim = {self.dim}")

    def _make_text_prompt(self, text: str, instruction: str) -> str:
        return _PROMPT_TEMPLATE.format(instruction=instruction, content=text)

    def _make_image_messages(self, images: List[Image.Image], instruction: str) -> List[list]:
        return [[{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": instruction},
            ],
        }] for img in images]

    def _render_image_prompts(self, messages: List[list]) -> List[str]:
        rendered = []
        for msg in messages:
            text = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered.append(text + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return rendered

    def _expand_image_tokens(self, texts: List[str], image_grid_thw: torch.Tensor) -> List[str]:
        expanded = list(texts)
        merge_length = self.image_processor.merge_size ** 2
        index = 0
        for i in range(len(expanded)):
            while self.image_token in expanded[i]:
                num_image_tokens = int(image_grid_thw[index].prod().item()) // merge_length
                expanded[i] = expanded[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            expanded[i] = expanded[i].replace("<|placeholder|>", self.image_token)
        return expanded

    @torch.no_grad()
    def _forward(self, inputs: dict) -> torch.Tensor:
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        embs = _last_token_pool(last_hidden, inputs["attention_mask"])
        return F.normalize(embs, p=2, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            prompts = [self._make_text_prompt(t, self.text_instruction) for t in batch]
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embs = self._forward(inputs)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_images(self, images: List[Optional[Image.Image]], batch_size: int = 4) -> np.ndarray:
        all_embs: List[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            valid_idxs = [j for j, img in enumerate(batch) if img is not None]
            valid_imgs = [batch[j] for j in valid_idxs]

            chunk_embs = np.zeros((len(batch), self.dim), dtype=np.float32)
            if valid_imgs:
                messages = self._make_image_messages(valid_imgs, self.image_instruction)
                rendered = self._render_image_prompts(messages)
                image_inputs = self.image_processor(images=valid_imgs, return_tensors="pt")
                rendered = self._expand_image_tokens(rendered, image_inputs["image_grid_thw"])

                text_inputs = self.tokenizer(
                    rendered,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {**text_inputs, **image_inputs}
                inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
                embs = self._forward(inputs).cpu().float().numpy()
                for out_j, orig_j in enumerate(valid_idxs):
                    chunk_embs[orig_j] = embs[out_j]
            all_embs.append(chunk_embs)
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        orig = self.text_instruction
        self.text_instruction = self.query_instruction
        emb = self.encode_texts([query])[0].flatten()
        self.text_instruction = orig
        return emb

    @torch.no_grad()
    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        orig = self.image_instruction
        self.image_instruction = self.query_instruction
        emb = self.encode_images([image])[0].flatten()
        self.image_instruction = orig
        return emb
