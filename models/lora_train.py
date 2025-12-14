# models/lora_train.py
import os
import inspect
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler

# diffusers 0.36.0: this wrapper is the right way to get trainable LoRA params
try:
    from diffusers.loaders import AttnProcsLayers
except Exception:
    # Some installs use a slightly different import path
    from diffusers.loaders.attn_procs import AttnProcsLayers

# Prefer 2.0 processor if available
try:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LORA_PROC
    LORA_NAME = "LoRAAttnProcessor2_0"
except Exception:
    from diffusers.models.attention_processor import LoRAAttnProcessor as LORA_PROC
    LORA_NAME = "LoRAAttnProcessor"


# ======================
# CONFIG
# ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 1          # keep 1 for speed (you can set 2 later)
BATCH_SIZE = 1
LR = 1e-4
RANK = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ======================
# DATASET
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
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


# ======================
# Helpers
# ======================
def _hidden_size_for_attn_name(unet, name: str) -> int:
    # SD1.5 block_out_channels usually [320, 640, 1280, 1280]
    block_out = list(unet.config.block_out_channels)

    if name.startswith("mid_block"):
        return block_out[-1]

    if name.startswith("down_blocks"):
        idx = int(name.split(".")[1])
        return block_out[idx]

    if name.startswith("up_blocks"):
        idx = int(name.split(".")[1])
        return list(reversed(block_out))[idx]

    return block_out[0]


def _make_lora_processor(hidden_size: int, cross_attention_dim: int | None):
    """
    Build LoRA processor across diffusers variants:
    - Some accept kwargs hidden_size/cross_attention_dim
    - Some accept positional (hidden_size, cross_attention_dim)
    - Rank kw name can be rank or lora_rank or not supported
    """
    sig = inspect.signature(LORA_PROC.__init__)
    params = sig.parameters

    # try kwargs first
    kwargs = {}
    if "hidden_size" in params:
        kwargs["hidden_size"] = hidden_size
    if "cross_attention_dim" in params:
        kwargs["cross_attention_dim"] = cross_attention_dim

    # optional rank fields (some builds don't support these!)
    if "rank" in params:
        kwargs["rank"] = RANK
    elif "lora_rank" in params:
        kwargs["lora_rank"] = RANK

    # attempt kwargs construction
    if kwargs:
        try:
            return LORA_PROC(**kwargs)
        except TypeError:
            pass

    # attempt positional
    try:
        return LORA_PROC(hidden_size, cross_attention_dim)
    except TypeError:
        pass

    # last resort
    return LORA_PROC()


def add_lora_to_unet(unet):
    print(f"Injecting {LORA_NAME} into UNet attention processors...")

    cross_dim = getattr(unet.config, "cross_attention_dim", None)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        hs = _hidden_size_for_attn_name(unet, name)
        ca_dim = None if "attn1" in name else cross_dim
        lora_attn_procs[name] = _make_lora_processor(hs, ca_dim)

    unet.set_attn_processor(lora_attn_procs)

    # ✅ IMPORTANT: AttnProcsLayers is what makes LoRA params trainable in diffusers 0.36
    lora_layers = AttnProcsLayers(unet.attn_processors)
    return lora_layers


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

    # Inject LoRA + get trainable module
    lora_layers = add_lora_to_unet(pipe.unet).to(DEVICE)
    trainable_params = list(lora_layers.parameters())

    if len(trainable_params) == 0:
        raise RuntimeError("LoRA injection happened but produced 0 trainable params. (Unexpected on diffusers 0.36.0)")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

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
    lora_layers.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images in pbar:
            images = images.to(DEVICE, dtype=DTYPE)

            # 1) images -> latents (B,4,64,64)
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # 2) add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # 3) predict noise with UNet (LoRA layers are active inside attn processors)
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


    os.makedirs(OUTPUT_DIR, exist_ok=True)


    pipe.unet.save_attn_procs(OUTPUT_DIR)

    print(f"LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
