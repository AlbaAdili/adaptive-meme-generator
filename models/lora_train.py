
import os
import math
import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor


# ============================================================
# CONFIG 
# ============================================================
MODEL_ID = "runwayml/stable-diffusion-v1-5"

DATA_DIR = "data/meme_train"    
OUTPUT_DIR = "lora_weights/meme_lora" 

TRAIN_PROMPT = "a funny meme, internet meme style, high contrast, clean, sharp"


RESOLUTION = 512
BATCH_SIZE = 1
EPOCHS = 2
LR = 1e-4
RANK = 8
GRAD_ACCUM_STEPS = 1

SEED = 42


# ============================================================
# UTIL
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            files.append(os.path.join(folder, f))
    files.sort()
    return files


# ============================================================
# DATASET
# ============================================================
class MemeImageDataset(Dataset):
    def __init__(self, folder: str, resolution: int = 512):
        self.files = list_images(folder)
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No images found in {folder}. Put .png/.jpg images there."
            )

        self.tf = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return {"pixel_values": self.tf(img), "path": path}


# ============================================================
# LoRA injection helpers (robust: no hidden_size attribute needed)
# ============================================================
def _infer_hidden_size_from_name(unet, name: str) -> int:
    """
    Determine attention hidden size from module name, robust across diffusers versions.
    """
    # SD1.5 UNet uses block_out_channels, and attention hidden_size equals the channel dim
    # for each block.
    block_out = list(unet.config.block_out_channels)  # e.g. [320, 640, 1280, 1280]

    if name.startswith("mid_block"):
        return block_out[-1]

    # down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn...
    if name.startswith("down_blocks"):
        # name like: down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor
        parts = name.split(".")
        i = int(parts[1])
        return block_out[i]

    if name.startswith("up_blocks"):
        parts = name.split(".")
        i = int(parts[1])
        # up_blocks go reverse: 0 corresponds to last channel
        return block_out[::-1][i]

    # fallback
    return block_out[0]


def add_lora_to_unet(unet, rank: int = 8):
    """
    Replace attention processors with LoRAAttnProcessor.
    Works even if attention slicing had been enabled earlier.
    """
    # IMPORTANT: avoid SlicedAttnProcessor (it breaks LoRA injection)
    if hasattr(unet, "set_default_attn_processor"):
        unet.set_default_attn_processor()

    lora_attn_procs = {}
    cross_attention_dim = getattr(unet.config, "cross_attention_dim", 768)

    for name in unet.attn_processors.keys():
        # attn1 = self-attn (no cross attention)
        is_self_attn = name.endswith("attn1.processor")
        hidden_size = _infer_hidden_size_from_name(unet, name)
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=None if is_self_attn else cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    return lora_attn_procs


def lora_parameters(unet):
    """
    Collect LoRA params only.
    """
    params = []
    for _, proc in unet.attn_processors.items():
        for p in proc.parameters():
            if p.requires_grad:
                params.append(p)
    return params


# ============================================================
# TRAIN
# ============================================================
def main():
    seed_all(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device} | dtype: {dtype}")
    print("Loading Stable Diffusion pipeline...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)


    # Freeze everything first
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Scheduler for adding noise during training
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    print("Injecting LoRA into UNet attention processors...")
    add_lora_to_unet(pipe.unet, rank=RANK)

    # Enable training mode on LoRA processors
    pipe.unet.train()

    # Only LoRA params trainable
    params = lora_parameters(pipe.unet)
    if len(params) == 0:
        raise RuntimeError("No LoRA parameters found. Injection failed.")

    optimizer = torch.optim.AdamW(params, lr=LR)

    dataset = MemeImageDataset(DATA_DIR, resolution=RESOLUTION)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))

    print(f"Training LoRA on {len(dataset)} meme images")
    print(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | LR={LR} | Rank={RANK}")

    global_step = 0
    pipe.tokenizer.model_max_length = min(pipe.tokenizer.model_max_length, 77)

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)  # (B,3,H,W)

            # 1) Encode images -> latents (B,4,H/8,W/8)
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

            # 2) Sample noise + timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )

            # 3) Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4) Text conditioning (single prompt for quick demo)
            with torch.no_grad():
                tokens = pipe.tokenizer(
                    [TRAIN_PROMPT] * bsz,
                    padding="max_length",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                encoder_hidden_states = pipe.text_encoder(tokens)[0]

            # 5) Predict noise
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                noise_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"Epoch {epoch+1} finished. Avg loss: {running_loss/len(loader):.4f}")

    # ============================================================
    # SAVE LoRA WEIGHTS
    # ============================================================
    print("Saving LoRA weights...")

    # Most robust for your generate_image.py:
    # - If pipe.save_lora_weights exists, it matches pipe.load_lora_weights(...)
    if hasattr(pipe, "save_lora_weights"):
        pipe.save_lora_weights(OUTPUT_DIR)
        print(f"Saved with pipe.save_lora_weights → {OUTPUT_DIR}")
    else:
        # Fallback: save attention processors
        pipe.unet.save_attn_procs(OUTPUT_DIR)
        print(f" Saved with unet.save_attn_procs → {OUTPUT_DIR}")

    # Save a tiny info file too
    with open(os.path.join(OUTPUT_DIR, "training_info.txt"), "w") as f:
        f.write(f"MODEL_ID={MODEL_ID}\n")
        f.write(f"DATA_DIR={DATA_DIR}\n")
        f.write(f"TRAIN_PROMPT={TRAIN_PROMPT}\n")
        f.write(f"EPOCHS={EPOCHS}\n")
        f.write(f"BATCH_SIZE={BATCH_SIZE}\n")
        f.write(f"LR={LR}\n")
        f.write(f"RANK={RANK}\n")

    print("Done.")


if __name__ == "__main__":
    main()
