import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler


# ======================
# CONFIG
# ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 2          
LR = 1e-4
RANK = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ======================
# DATASET (images only)
# ======================
class MemeDataset(Dataset):
    def __init__(self, folder):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


# ======================
# MAIN
# ======================
def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")
    print("Loading Stable Diffusion pipeline...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.unet.requires_grad_(False)


    pipe.unet.add_adapter(
        adapter_name="meme_lora",
        rank=RANK,
    )

    pipe.unet.train()

    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=LR,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader) * EPOCHS,
    )

    print(f"Training LoRA on {len(dataset)} meme images")

    for epoch in range(EPOCHS):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(DEVICE, dtype=DTYPE)

            # Encode images → latents
            latents = pipe.vae.encode(batch).latent_dist.sample()
            latents = latents * 0.18215

            # Noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Dummy text embeddings (style-only training)
            encoder_hidden_states = torch.zeros(
                (latents.shape[0], 77, pipe.text_encoder.config.hidden_size),
                device=DEVICE,
                dtype=DTYPE,
            )

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} finished")

    # ======================
    # SAVE LoRA
    # ======================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_attn_procs(OUTPUT_DIR)

    print(f" LoRA weights saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
