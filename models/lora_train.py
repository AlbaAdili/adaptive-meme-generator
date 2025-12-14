
import os
import inspect
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler


try:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LORA_PROC
    LORA_PROC_NAME = "LoRAAttnProcessor2_0"
except Exception:
    from diffusers.models.attention_processor import LoRAAttnProcessor as LORA_PROC
    LORA_PROC_NAME = "LoRAAttnProcessor"


# ======================
# CONFIG
# ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-4
RANK = 4  # used only if your diffusers supports it

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ======================
# DATASET (IMAGES ONLY)
# ======================
class MemeDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB"))


# ======================
# LoRA processor builder (version-safe)
# ======================
def _make_lora_processor(hidden_size: int, cross_attention_dim: int | None):
    """
    Build a LoRA attention processor across different diffusers versions.

    Some versions accept:
      - LoRAAttnProcessor(hidden_size, cross_attention_dim, rank=...)
      - LoRAAttnProcessor(hidden_size=..., cross_attention_dim=..., rank=...)
      - LoRAAttnProcessor(hidden_size, cross_attention_dim)
      - LoRAAttnProcessor2_0(hidden_size, cross_attention_dim)
    """
    sig = inspect.signature(LORA_PROC.__init__)
    params = list(sig.parameters.keys())  # includes 'self'

    kwargs = {}
    # common names
    if "hidden_size" in params:
        kwargs["hidden_size"] = hidden_size
    if "cross_attention_dim" in params:
        kwargs["cross_attention_dim"] = cross_attention_dim

    # rank names differ (or not present)
    if "rank" in params:
        kwargs["rank"] = RANK
    if "lora_rank" in params:
        kwargs["lora_rank"] = RANK

    # If it supports kwargs, use them.
    if len(kwargs) > 0:
        try:
            return LORA_PROC(**kwargs)
        except TypeError:
            pass

    # Otherwise try positional: (hidden_size, cross_attention_dim)
    try:
        return LORA_PROC(hidden_size, cross_attention_dim)
    except TypeError:
        pass

    # Last resort: no-arg constructor (may create no weights on some builds)
    return LORA_PROC()


def _hidden_size_for_attn_name(unet, name: str) -> int:
    """
    Infer hidden size from attention processor name.
    This matches diffusers Stable Diffusion UNet naming conventions.
    """
    block_out = list(unet.config.block_out_channels)  # e.g. [320, 640, 1280, 1280]

    if name.startswith("mid_block"):
        return block_out[-1]

    if name.startswith("down_blocks"):
        # down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attnX.processor
        idx = int(name.split(".")[1])
        return block_out[idx]

    if name.startswith("up_blocks"):
        # up_blocks.{i} uses reversed channels
        idx = int(name.split(".")[1])
        return list(reversed(block_out))[idx]

    # fallback
    return block_out[0]


def add_lora_to_unet(unet):
    print(f"Injecting {LORA_PROC_NAME} into UNet attention processors...")
    cross_dim = getattr(unet.config, "cross_attention_dim", None)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        hs = _hidden_size_for_attn_name(unet, name)

        # self-attn vs cross-attn
        # attn1 = self-attention => cross_attention_dim=None
        # attn2 = cross-attention => cross_attention_dim = unet.config.cross_attention_dim
        ca_dim = None if "attn1" in name else cross_dim

        lora_attn_procs[name] = _make_lora_processor(hs, ca_dim)

    unet.set_attn_processor(lora_attn_procs)

    # Collect trainable LoRA params from the processors dict
    trainable = []
    for proc in unet.attn_processors.values():
        if hasattr(proc, "parameters"):
            for p in proc.parameters():
                p.requires_grad_(True)
                trainable.append(p)

    # If no params, your diffusers build’s LoRA processor is incompatible
    if len(trainable) == 0:
        raise RuntimeError(
            "LoRA processors were set, but no trainable parameters were created.\n"
            "This usually means your diffusers version has a LoRA processor class "
            "that requires different constructor args.\n"
            "Quick fix: run `pip show diffusers` and tell me the version."
        )

    return trainable


# ======================
# MAIN TRAIN
# ======================
def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze base model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Add LoRA and get trainable params
    trainable_params = add_lora_to_unet(pipe.unet)
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # Prepare conditioning (empty prompt)
    with torch.no_grad():
        tokens = pipe.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        empty_emb = pipe.text_encoder(tokens)[0]  # (1, 77, hidden)

    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training LoRA on {len(dataset)} images")
    pipe.unet.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images in pbar:
            images = images.to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            enc = empty_emb.repeat(latents.shape[0], 1, 1)

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=enc,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        print(f"Epoch {epoch+1} done")

    # Save LoRA
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if hasattr(pipe.unet, "save_attn_procs"):
        pipe.unet.save_attn_procs(OUTPUT_DIR)
    elif hasattr(pipe, "save_lora_weights"):
        pipe.save_lora_weights(OUTPUT_DIR)
    else:
   
        pipe.unet.save_pretrained(OUTPUT_DIR)

    print(f" LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
