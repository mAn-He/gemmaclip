#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small Vision-Language Model (sVLM) training script

Architecture: LLaVA-style with a frozen vision encoder (CLIP ViT-L/14-336),
a frozen sLLM (Gemma-3 4B IT in 4-bit), and a trainable 2-layer MLP projector.
QLoRA is applied to the LLM attention and MLP blocks. The model is trained for
Multimodal Chain-of-Thought (MCoT) reasoning on ScienceQA.

This script is designed to run on a single ~24GB GPU (e.g., RTX 3090/4090, L4).

Step 1: Environment Setup (CUDA 11.8 compatible)

Run these commands in your shell first:

    conda create -n svlm python=3.10 -y
    conda activate svlm
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install transformers==4.41.2 datasets==2.19.0 accelerate==0.30.1
    pip install peft==0.10.0 bitsandbytes==0.41.3
    pip install trl==0.8.6
    pip install flash-attn==2.5.8 --no-build-isolation

Notes:
- Ensure your GPU architecture supports FlashAttention-2 (Ampere or newer).
- Optionally set your Hugging Face token in the environment: export HF_TOKEN=... 

Usage:
    python svlm_gemma_clip_scienceqa.py \
      --output_dir ./outputs \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 1e-4 \
      --bf16 True \
      --max_train_samples 2000  # optional for quick run

"""

from __future__ import annotations

import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import HfFolder
from huggingface_hub import login as hf_login

# ------------------------------
# Utilities and Configuration
# ------------------------------

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):  # type: ignore[attr-defined]
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):  # type: ignore[attr-defined]
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------
# Argument Parsing
# ------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sVLM (Gemma-3 4B IT + CLIP L/14-336) with LLaVA-style projector on ScienceQA")

    # Core I/O
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and logs")
    parser.add_argument("--projector_path", type=str, default=None, help="Optional path to a saved projector state_dict to resume")

    # Data
    parser.add_argument("--dataset_name", type=str, default="derek-thomas/ScienceQA", help="Dataset name on the Hub")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration (use None for default config)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional cap on number of training samples for quick runs")
    parser.add_argument("--image_size", type=int, default=336, help="CLIP expected image size; 336 for ViT-L/14-336")

    # Models
    parser.add_argument("--llm_name", type=str, default="google/gemma-3-4b-it", help="sLLM model repo id")
    parser.add_argument("--vision_name", type=str, default="openai/clip-vit-large-patch14-336", help="Vision encoder repo id")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Micro-batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for projector and LoRA params")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for linear scheduler")

    # Precision / Performance
    parser.add_argument("--bf16", type=str2bool, default=True, help="Use bfloat16 if supported")
    parser.add_argument("--fp16", type=str2bool, default=False, help="Use float16 if bf16 not available")
    parser.add_argument("--use_flash_attention_2", type=str2bool, default=True, help="Use FlashAttention-2 if available")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True, help="Enable gradient checkpointing for memory savings")

    # LoRA/QLoRA
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Logging
    parser.add_argument("--log_every", type=int, default=10, help="Log every N update steps")

    # Auth
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models; otherwise use env or CLI login")

    return parser


# ------------------------------
# Special Tokens and Conversation Templates
# ------------------------------

SPECIAL_TOKENS = {
    "start_of_turn": "<start_of_turn>",
    "end_of_turn": "<end_of_turn>",
    "image": "<image>",
}

USER_TEMPLATE = (
    "{sot}user\n{image}\n{question}\nContext: {context}\nOptions: {choices}{eot}"
)

MODEL_TEMPLATE = (
    "{sot}model\n**Step-by-step reasoning:**\n{explanation}\n\n**Final Answer:**\n{answer}{eot}"
)


# ------------------------------
# Dataset Preprocessing
# ------------------------------

def format_scienceqa_sample(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform one ScienceQA sample into the LLaVA conversational format.
    Expects fields like: question, choices, answer (or correct option), explanation, context, and image.
    Skips examples without images when using the 'image' config.
    """
    # ScienceQA variants can have slightly different field names. We try to be robust.
    question = example.get("question")
    context = example.get("hint") or example.get("context") or ""
    explanation = example.get("explanation") or example.get("rationale") or example.get("solution") or ""

    # Choices may be list of strings under key 'choices' or 'choices_text'
    choices_list = example.get("choices") or example.get("choices_text") or []
    if isinstance(choices_list, dict):
        # sometimes stored as {"A": ..., "B": ...}
        # convert to ordered string
        alpha = sorted(choices_list.keys())
        choices_list = [f"{k}. {choices_list[k]}" for k in alpha]
    choices_str = "\n".join([f"- {c}" for c in choices_list]) if choices_list else "(no options)"

    # Answer may be an index or a string choice
    answer = example.get("answer")
    # Some configs use 'answer' as index; map to text when possible
    if isinstance(answer, int) and isinstance(choices_list, list) and 0 <= answer < len(choices_list):
        answer_text = choices_list[answer]
    else:
        answer_text = str(answer)

    # Image presence; with 'image' config, there should be a PIL Image in example['image']
    image = example.get("image")
    if image is None:
        # Skip sample if no image available
        return None

    # Build conversation strings
    user_prompt = USER_TEMPLATE.format(
        sot=SPECIAL_TOKENS["start_of_turn"],
        eot=SPECIAL_TOKENS["end_of_turn"],
        image=SPECIAL_TOKENS["image"],
        question=question or "",
        context=context or "",
        choices=choices_str,
    )
    model_target = MODEL_TEMPLATE.format(
        sot=SPECIAL_TOKENS["start_of_turn"],
        eot=SPECIAL_TOKENS["end_of_turn"],
        explanation=explanation or "",
        answer=answer_text or "",
    )

    return {
        "prompt": user_prompt,
        "response": model_target,
        "image": image,
    }


# ------------------------------
# Model: LLaVAGemma Wrapper
# ------------------------------

class Projector(nn.Module):
    """
    Two-layer MLP with GELU: maps CLIP hidden size -> Gemma hidden size.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Batch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class LLaVAGemma(nn.Module):
    """
    Encapsulates frozen vision encoder, frozen 4-bit Gemma-3 4B IT, and a trainable projector.
    """

    def __init__(
        self,
        vision_model: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        llm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        projector: Projector,
        image_token_id: int,
        use_flash_attention_2: bool = True,
    ):
        super().__init__()
        self.vision_model = vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False

        self.image_processor = image_processor
        self.llm = llm
        self.tokenizer = tokenizer
        self.projector = projector
        self.image_token_id = image_token_id

        # Try to ensure flash attention if requested
        self.use_flash_attention_2 = use_flash_attention_2

    @torch.no_grad()
    def encode_images(self, images: List[Any], device: torch.device) -> torch.Tensor:
        # Process images with CLIP processor
        batch = self.image_processor(images=images, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device)
        outputs = self.vision_model(pixel_values=pixel_values)
        vision_hidden = outputs.last_hidden_state  # [B, num_patches+1, vision_dim]
        return vision_hidden

    def build_inputs(
        self,
        prompts: List[str],
        responses: List[str],
        projected_images: torch.Tensor,
        device: torch.device,
    ) -> Batch:
        """
        Build inputs_embeds by tokenizing prompt+response, then replacing the single <image> token position
        with the projected image sequence embeddings. Labels are -100 for prompt and image tokens, and the
        token ids for response tokens.
        """
        assert len(prompts) == len(responses) == projected_images.size(0)

        # Tokenize prompt + response to derive labels alignment
        # We will later inject image embeddings and adjust labels accordingly
        combined_texts = [p + r for p, r in zip(prompts, responses)]
        tokenized = self.tokenizer(
            combined_texts,
            padding=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )

        batch_inputs_embeds: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for i, input_ids in enumerate(tokenized["input_ids"]):
            ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            # Find the boundary index: end of prompt within combined text.
            # We can get prompt tokenization alone to find its length.
            prompt_ids = self.tokenizer(
                prompts[i], padding=False, truncation=True, return_tensors=None, add_special_tokens=True
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Locate the image token index in the prompt part
            # It must exist by construction
            try:
                image_pos_in_prompt = prompt_ids.index(self.image_token_id)
            except ValueError:
                raise RuntimeError("<image> token not found in the prompt tokens. Ensure special tokens added correctly.")

            # Compute positions in the combined sequence where prompt tokens lie
            # combined ids start with exactly the prompt ids followed by response ids, by our construction.
            # We'll construct base input embeddings for text tokens
            text_embeds = self.llm.get_input_embeddings()(ids)  # [L, hidden]

            # Split into: [pre_image_tokens] + [image_token] + [post_image_prompt_tokens] + [response_tokens]
            pre_image_len = image_pos_in_prompt
            post_image_prompt_len = prompt_len - image_pos_in_prompt - 1  # after image token within prompt
            response_len = ids.size(0) - prompt_len

            pre_image_embeds = text_embeds[:pre_image_len]
            post_image_prompt_embeds = text_embeds[pre_image_len + 1 : prompt_len]
            response_embeds = text_embeds[prompt_len:]

            # Get projected image embeddings for this sample
            proj_img = projected_images[i]  # [num_img_tokens, hidden]

            # Compose inputs_embeds sequence: pre_image + proj_img + post_prompt + response
            inputs_embeds = torch.cat([pre_image_embeds, proj_img, post_image_prompt_embeds, response_embeds], dim=0)

            # Attention mask: ones for all tokens
            attention_mask = torch.ones(inputs_embeds.size(0), dtype=torch.long, device=device)

            # Labels: -100 for everything up to end of prompt (including image tokens), and token ids for response
            labels = torch.full((inputs_embeds.size(0),), fill_value=-100, dtype=torch.long, device=device)
            # We need response token ids
            response_ids = ids[prompt_len:]
            # Place response ids aligned at the tail
            labels[-response_len:] = response_ids if response_len > 0 else labels[-response_len:]

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad to the same length within the batch (right pad)
        max_len = max(x.size(0) for x in batch_inputs_embeds)
        hidden_size = batch_inputs_embeds[0].size(-1)

        padded_inputs = torch.zeros((len(prompts), max_len, hidden_size), dtype=batch_inputs_embeds[0].dtype, device=device)
        padded_mask = torch.zeros((len(prompts), max_len), dtype=torch.long, device=device)
        padded_labels = torch.full((len(prompts), max_len), fill_value=-100, dtype=torch.long, device=device)

        for i in range(len(prompts)):
            L = batch_inputs_embeds[i].size(0)
            padded_inputs[i, :L] = batch_inputs_embeds[i]
            padded_mask[i, :L] = batch_attention_mask[i]
            padded_labels[i, :L] = batch_labels[i]

        return Batch(inputs_embeds=padded_inputs, attention_mask=padded_mask, labels=padded_labels)

    def forward(self, batch: Batch) -> torch.Tensor:
        outputs = self.llm(
            inputs_embeds=batch.inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        )
        return outputs.loss


# ------------------------------
# Collate Function
# ------------------------------

class DataCollator:
    def __init__(
        self,
        model: LLaVAGemma,
        device: torch.device,
        max_images_per_batch: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.max_images_per_batch = max_images_per_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Batch:
        images = [f["image"] for f in features]
        prompts = [f["prompt"] for f in features]
        responses = [f["response"] for f in features]

        with torch.no_grad():
            vision_hidden = self.model.encode_images(images, device=self.device)  # [B, N_img_tokens, vision_dim]
        # Project to LLM hidden size
        projected = self.model.projector(vision_hidden)  # [B, N_img_tokens, llm_hidden]

        batch = self.model.build_inputs(prompts=prompts, responses=responses, projected_images=projected, device=self.device)
        return batch


# ------------------------------
# LoRA Target Modules (Gemma)
# ------------------------------

def get_gemma_lora_modules(llm: nn.Module) -> List[str]:
    """
    Return typical target module names for QLoRA on Gemma attention and MLP.
    We include projections often named as below in HF Llama/Gemma-like models.
    """
    candidate_names = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ]

    present = set()
    for name, _ in llm.named_modules():
        base = name.split(".")[-1]
        if base in candidate_names:
            present.add(base)
    # Fallback to return candidates even if not found; PEFT will warn/skip as needed
    return sorted(list(present)) if present else candidate_names


# ------------------------------
# Training Loop
# ------------------------------

def train(
    args: argparse.Namespace,
    model: LLaVAGemma,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
):
    model.train()
    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_update_steps = args.num_train_epochs * num_update_steps_per_epoch

    for epoch in range(args.num_train_epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / args.log_every
                    print(f"Epoch {epoch+1} | Step {global_step}/{total_update_steps} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.6e}")
                    running_loss = 0.0

        # End of epoch logging
        if running_loss > 0:
            avg_loss = running_loss / ((step + 1) % args.log_every)
            print(f"Epoch {epoch+1} | tail avg loss {avg_loss:.4f}")


# ------------------------------
# Main
# ------------------------------

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()

    # Hugging Face authentication (optional)
    env_token = os.getenv("HF_TOKEN", None) or os.getenv("HUGGINGFACE_TOKEN", None)
    stored_token = HfFolder.get_token()
    hf_token = args.hf_token or env_token or stored_token

    # If token provided explicitly, perform programmatic login
    if args.hf_token:
        try:
            hf_login(token=args.hf_token, add_to_git_credential=True)
            print("Logged in to Hugging Face via provided token.")
        except Exception as e:
            print(f"Warning: programmatic HF login failed: {e}")
    else:
        # If no explicit token and not already logged in nor env var set, fail fast with guidance
        if env_token is None and stored_token is None:
            raise RuntimeError(
                "Hugging Face token required for 'google/gemma-3-4b-it'. "
                "Accept the model terms on the Hub and login via 'huggingface-cli login' or pass --hf_token / set HF_TOKEN."
            )

    # Precision selection
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(args.bf16 and bf16_available)
    use_fp16 = bool(args.fp16 and not use_bf16)

    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=False)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": [
            SPECIAL_TOKENS["start_of_turn"],
            SPECIAL_TOKENS["end_of_turn"],
            SPECIAL_TOKENS["image"],
        ]
    })
    image_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["image"])  # type: ignore[arg-type]

    # Load 4-bit LLM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    attn_impl = "flash_attention_2" if (args.use_flash_attention_2) else None

    try:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl is not None:
            print(f"Warning: failed to enable FlashAttention-2 ({e}). Falling back to default attention.")
            llm = AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map="auto",
            )
        else:
            raise
    # Resize embeddings in case we added special tokens
    if added_tokens and added_tokens > 0:
        llm.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=args.gradient_checkpointing)

    # Apply QLoRA
    target_modules = get_gemma_lora_modules(llm)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    # Optional: enable gradient checkpointing for memory
    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable()

    # Load CLIP vision encoder and processor (frozen)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_name)
    vision_model = CLIPVisionModel.from_pretrained(args.vision_name)
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad = False

    # Build projector: map CLIP hidden -> LLM hidden
    # Infer dims dynamically
    with torch.no_grad():
        vision_dim = vision_model.config.hidden_size  # e.g., 1024 for ViT-L/14-336
    llm_hidden = llm.config.hidden_size  # type: ignore[attr-defined]
    projector = Projector(input_dim=vision_dim, output_dim=llm_hidden).to(device)

    # Optionally load projector weights
    if args.projector_path and os.path.isfile(args.projector_path):
        projector.load_state_dict(torch.load(args.projector_path, map_location="cpu"))
        print(f"Loaded projector weights from {args.projector_path}")

    # Build composite model
    llava = LLaVAGemma(
        vision_model=vision_model.to(device),
        image_processor=image_processor,
        llm=llm,
        tokenizer=tokenizer,
        projector=projector,
        image_token_id=image_token_id,
        use_flash_attention_2=args.use_flash_attention_2,
    )

    # Freeze all parameters except projector and LoRA-adapted LLM params
    for p in llava.parameters():
        p.requires_grad = False
    for p in llava.projector.parameters():
        p.requires_grad = True
    # PEFT marks LoRA parameters as requires_grad=True already
    for n, p in llava.llm.named_parameters():
        # Keep only LoRA params trainable
        if "lora_" in n:
            p.requires_grad = True

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_ds = dataset["train"]

    # Map to formatted samples
    def map_fn(ex):
        out = format_scienceqa_sample(ex)
        return out if out is not None else {"prompt": None, "response": None, "image": None}

    mapped = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    # Filter out Nones
    mapped = mapped.filter(lambda x: x["prompt"] is not None and x["response"] is not None and x["image"] is not None)

    if args.max_train_samples is not None and args.max_train_samples > 0:
        mapped = mapped.select(range(min(args.max_train_samples, len(mapped))))

    # DataLoader
    collator = DataCollator(model=llava, device=device)
    train_loader = DataLoader(
        mapped,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
    )

    # Optimizer and Scheduler
    # Only trainable params: projector + LoRA
    trainable_params = [p for p in llava.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Enable CuDNN/TF32 optimizations when safe
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    print("Starting training...")
    train(args=args, model=llava, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, device=device)

    # Save outputs: projector and LoRA adapter
    projector_path = os.path.join(args.output_dir, "projector.pt")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(llava.projector.state_dict(), projector_path)
    print(f"Saved projector to {projector_path}")

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    llava.llm.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    # Save tokenizer (with special tokens)
    tok_dir = os.path.join(args.output_dir, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    print(f"Saved tokenizer (with special tokens) to {tok_dir}")


if __name__ == "__main__":
    main() 