# Gemma-CLIP sVLM (LLaVA-style) - Training and Inference

This project trains a small Vision-Language Model (sVLM) by connecting a frozen CLIP vision encoder (ViT-L/14-336) to a frozen Gemma-3 4B IT LLM via a trainable 2-layer MLP projector, and applies QLoRA to the LLM. Training data: ScienceQA (image config). Target: single ~24GB GPU.

## 1) Conda Environment (CUDA 11.8)

```bash
# Create and activate env
conda create -n svlm python=3.10 -y
conda activate svlm

# Install CUDA 11.8 PyTorch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Core libraries (Gemma 3 requires transformers >= 4.50.0)
pip install "transformers>=4.50.0" datasets==2.19.0 accelerate==0.30.1
pip install peft==0.10.0 bitsandbytes==0.41.3
pip install trl==0.8.6

# Optional but recommended for memory/speed (Ampere+ GPUs)
pip install flash-attn==2.5.8 --no-build-isolation
```

Accept the model license and login to Hugging Face first. You must be logged in and have accepted the `google/gemma-3-4b-it` terms:

1. Open the model page and click "Access" to accept: https://huggingface.co/google/gemma-3-4b-it
2. Login from CLI:

```bash
huggingface-cli login
```

Alternatively, set your token in the environment:

```bash
export HF_TOKEN=hf_********************************
```

## 2) Training

```bash
python /home/hseung/side_project/gemmaclip/train_svlm_gemma_clip_scienceqa.py \
  --output_dir /home/hseung/side_project/gemmaclip/outputs \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --bf16 True \
  --max_train_samples 2000  # optional quick run
```

Artifacts saved to `outputs/`:
- `projector.pt` (2-layer MLP)
- `lora_adapter/` (PEFT adapter)
- `tokenizer/` (tokenizer with added special tokens)

## 3) Inference

```bash
python /home/hseung/side_project/gemmaclip/inference_gemma_clip.py \
  --adapter_dir /home/hseung/side_project/gemmaclip/outputs/lora_adapter \
  --projector_path /home/hseung/side_project/gemmaclip/outputs/projector.pt \
  --tokenizer_dir /home/hseung/side_project/gemmaclip/outputs/tokenizer \
  --image_path /path/to/image.jpg \
  --question "Which option is correct?" \
  --context "" \
  --choices "A;B;C;D" \
  --max_new_tokens 256 --temperature 0.2 --top_p 0.9
```

## Notes
- Both CLIP vision encoder and Gemma LLM are frozen; only projector and LoRA params train.
- Training expects the ScienceQA `image` configuration (includes PIL images).
- FlashAttention-2 is used if installed and GPU supports it. 