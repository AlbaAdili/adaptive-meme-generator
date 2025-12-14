# models/lora_train.py
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import StableDiffusionPipeline, DDPMScheduler

from peft import LoraConfig, get_peft_model

# ======================
# CONFIG
# ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 1         
BATCH_SIZE = 1
LR = 1e-4
RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ======================
# DATASET (images only)
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


def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    # Load SD1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except UNet LoRA
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # ---- PEFT LoRA on UNet attention projections ----
    lora_cfg = LoraConfig(
        r=RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        # SD UNet attention projections typically have these names
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        task_type="FEATURE_EXTRACTION",
    )

    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    pipe.unet.train()

    # Trainable params check
    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    # Empty prompt embedding (fast style adaptation)
    with torch.no_grad():
        tokens = pipe.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        empty_emb = pipe.text_encoder(tokens)[0]  # (1,77,hidden)

    # Data
    ds = MemeDataset(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(ds)} images...")

    for epoch in range(EPOCHS):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images in pbar:
            images = images.to(DEVICE, dtype=DTYPE)

            # images -> latents (B,4,64,64)
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

    # Save PEFT adapter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_pretrained(OUTPUT_DIR)
    print(f"Saved PEFT LoRA adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
