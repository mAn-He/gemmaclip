#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for the sVLM (Gemma-3 4B IT + CLIP ViT-L/14-336 + 2-layer projector)

Loads the base model in 4-bit, applies the saved LoRA adapters, loads the saved
projector weights, encodes the input image via CLIP, and generates a response to
an input question using the same LLaVA-style formatting as training.

Example:
    python inference_gemma_clip.py \
      --adapter_dir /home/hseung/side_project/gemmaclip/outputs/lora_adapter \
      --projector_path /home/hseung/side_project/gemmaclip/outputs/projector.pt \
      --tokenizer_dir /home/hseung/side_project/gemmaclip/outputs/tokenizer \
      --image_path /path/to/image.jpg \
      --question "What is shown in the image?" \
      --context "" \
      --choices "A cat;A dog;A bird" \
      --max_new_tokens 256 --temperature 0.2 --top_p 0.9
"""

from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
    set_seed,
)
from peft import PeftModel
from huggingface_hub import HfFolder
from huggingface_hub import login as hf_login

# Special tokens and templates must match training
SPECIAL_TOKENS = {
    "start_of_turn": "<start_of_turn>",
    "end_of_turn": "<end_of_turn>",
    "image": "<image>",
}

USER_TEMPLATE = (
    "{sot}user\n{image}\n{question}\nContext: {context}\nOptions: {choices}{eot}"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference for LLaVA-style Gemma-CLIP sVLM")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to saved LoRA adapter directory")
    parser.add_argument("--projector_path", type=str, required=True, help="Path to saved projector .pt file")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Path to saved tokenizer directory (recommended)")

    parser.add_argument("--llm_name", type=str, default="google/gemma-3-4b-it", help="Base LLM repo id")
    parser.add_argument("--vision_name", type=str, default="openai/clip-vit-large-patch14-336", help="Vision encoder repo id")

    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="User question")
    parser.add_argument("--context", type=str, default="", help="Optional context")
    parser.add_argument("--choices", type=str, default="", help="Optional semicolon-separated choices, e.g., 'A;B;C'")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--bf16", type=lambda x: x.lower() in ("1","true","yes","y"), default=True, help="Use bfloat16 if supported")
    parser.add_argument("--fp16", type=lambda x: x.lower() in ("1","true","yes","y"), default=False, help="Use float16 if bf16 unavailable")

    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")

    parser.add_argument("--hf_token", type=str, default=True, help="Hugging Face token for gated models; otherwise use env or CLI login")
    parser.add_argument("--output_file", type=str, default="inference_results.md", help="Output markdown file to save results")

    return parser


class Projector(nn.Module):
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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_results_to_md(args, response_text: str):
    """Save inference results to markdown file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare content
    md_content = f"""# sVLM Inference Results

## Inference Run: {timestamp}

### Image
**Path:** `{args.image_path}`

### Query
**Question:** {args.question}

### Context
{args.context if args.context else "*No context provided*"}

### Choices
{chr(10).join([f"- {choice.strip()}" for choice in args.choices.split(";")]) if args.choices else "*No choices provided*"}

### Model Response
```
{response_text}
```

### Configuration
- **Model:** {args.llm_name}
- **Vision:** {args.vision_name}
- **Adapter:** {args.adapter_dir}
- **Projector:** {args.projector_path}
- **Max New Tokens:** {args.max_new_tokens}
- **Temperature:** {args.temperature}
- **Top-p:** {args.top_p}

---

"""
    
    # Append to file (create if doesn't exist)
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        md_content = md_content + existing_content
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Results saved to {args.output_file}")


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = get_device()

    hf_token = os.getenv("HF_TOKEN", None)
    if args.hf_token:
        try:
            hf_login(token=args.hf_token, add_to_git_credential=True)
            print("Logged in to Hugging Face via provided token.")
        except Exception as e:
            print(f"Warning: programmatic HF login failed: {e}")
    else:
        # derive from env or CLI login
        hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN", None) or HfFolder.get_token()
        if hf_token is None:
            raise RuntimeError(
                "Hugging Face token required for 'google/gemma-3-4b-it'. "
                "Accept the model terms on the Hub and login via 'huggingface-cli login' or pass --hf_token / set HF_TOKEN."
            )

    # Tokenizer
    if args.tokenizer_dir and os.path.isdir(args.tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                SPECIAL_TOKENS["start_of_turn"],
                SPECIAL_TOKENS["end_of_turn"],
                SPECIAL_TOKENS["image"],
            ]
        })
    image_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["image"])  # type: ignore[arg-type]

    # Precision
    bf16_available = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(args.bf16 and bf16_available)
    use_fp16 = bool(args.fp16 and not use_bf16)
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # LLM base in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    llm = None
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"Warning: failed to enable FlashAttention-2 ({e}). Falling back to default attention.")
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
    # Resize embeddings to match training (when special tokens were added)
    # The saved tokenizer includes the correct vocab size
    llm.resize_token_embeddings(len(tokenizer))
    
    # Apply adapters
    llm = PeftModel.from_pretrained(llm, args.adapter_dir, is_trainable=False)

    # Vision
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_name)
    vision_model = CLIPVisionModel.from_pretrained(args.vision_name).to(device)
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad = False

    # Projector
    vision_dim = vision_model.config.hidden_size
    llm_hidden = llm.config.hidden_size  # type: ignore[attr-defined]
    projector = Projector(vision_dim, llm_hidden).to(device)
    projector.load_state_dict(torch.load(args.projector_path, map_location="cpu"))
    projector.eval()

    # Input image
    image = Image.open(args.image_path).convert("RGB")

    # Build prompt
    if args.choices.strip():
        choices_str = "\n".join([f"- {c.strip()}" for c in args.choices.split(";")])
    else:
        choices_str = "(no options)"
    prompt = USER_TEMPLATE.format(
        sot=SPECIAL_TOKENS["start_of_turn"],
        eot=SPECIAL_TOKENS["end_of_turn"],
        image=SPECIAL_TOKENS["image"],
        question=args.question,
        context=args.context,
        choices=choices_str,
    )

    # Tokenize prompt
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)
    attention_mask_text = torch.ones_like(input_ids)

    # Find <image> token index
    ids_list = input_ids.tolist()
    if image_token_id not in ids_list:
        raise RuntimeError("<image> token not found after tokenization.")
    image_pos = ids_list.index(image_token_id)

    # Encode image and project
    with torch.no_grad():
        clip_inputs = image_processor(images=[image], return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].to(device)
        vision_out = vision_model(pixel_values=pixel_values)
        vision_hidden = vision_out.last_hidden_state  # [1, N_img_tokens, vision_dim]
        proj_img = projector(vision_hidden)[0]       # [N_img_tokens, llm_hidden]

        # Get text embeds and ensure dtype consistency
        text_embeds = llm.get_input_embeddings()(input_ids)  # [L, hidden]
        # Ensure projector output matches text embeddings dtype
        proj_img = proj_img.to(text_embeds.dtype)
        # Compose embeddings: pre_image + proj_img + post_image
        pre = text_embeds[:image_pos]
        post = text_embeds[image_pos+1:]
        inputs_embeds = torch.cat([pre, proj_img, post], dim=0).unsqueeze(0)  # [1, L+Nimg-1, hidden]

        # Attention mask
        attn_mask = torch.ones(inputs_embeds.size()[:2], dtype=torch.long, device=device)

        # Generate
        gen_out = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0.0),
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract model response
    full_text = tokenizer.decode(gen_out[0], skip_special_tokens=False)
    
    # Extract just the model's response (after the prompt)
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    if prompt_text in full_text:
        response_text = full_text[len(prompt_text):].strip()
    else:
        response_text = full_text.strip()
    
    # Clean up response
    if response_text.startswith("<start_of_turn>model"):
        response_text = response_text.replace("<start_of_turn>model", "").strip()
    if response_text.endswith("<end_of_turn>"):
        response_text = response_text.replace("<end_of_turn>", "").strip()
    
    print("===== Generation =====")
    print("Query:", args.question)
    print("Context:", args.context[:100] + "..." if len(args.context) > 100 else args.context)
    print("Choices:", args.choices)
    print("Model Response:", response_text)
    
    # Save results to markdown file
    save_results_to_md(args, response_text)


if __name__ == "__main__":
    main() 