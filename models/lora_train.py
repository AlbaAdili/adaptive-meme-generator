# models/lora_train.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig

# ============================
# CONFIG
# ============================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 1          # keep 1 for your deadline
BATCH_SIZE = 1
LR = 1e-4
RANK = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ============================
# DATASET (images only)
# ============================
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
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze base parts
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # ============================
    # REAL LoRA via PEFT adapter
    # ============================
    print("Adding PEFT LoRA adapter to UNet...")

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    # IMPORTANT: positional/normal call (no rank= keyword)
    pipe.unet.add_adapter(lora_config)

    # Enable adapter if API exists (version-safe)
    try:
        pipe.unet.set_adapter("default")
    except Exception:
        pass

    # Collect trainable params (now LoRA params exist)
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("Still no trainable params. LoRA adapter did not attach.")

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"✅ Trainable LoRA params: {total_trainable}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # Empty prompt embedding (image-only training)
    with torch.no_grad():
        tokens = pipe.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        empty_emb = pipe.text_encoder(tokens)[0]

    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training on {len(dataset)} images...")
    pipe.unet.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                latents = pipe.vae.encode(batch).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = empty_emb.repeat(latents.shape[0], 1, 1)

            # Normal UNet forward (NO input_ids anywhere)
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    # ============================
    # SAVE LoRA weights
    # ============================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Version-safe saving:
    # - If diffusers exposes save_lora_weights -> best
    # - else save UNet adapters
    if hasattr(pipe, "save_lora_weights"):
        pipe.save_lora_weights(OUTPUT_DIR)
    else:
        # Saves adapter weights/config
        pipe.unet.save_pretrained(OUTPUT_DIR)

    print(f" LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
