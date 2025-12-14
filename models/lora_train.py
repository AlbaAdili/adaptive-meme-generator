# models/lora_train.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor


# ======================
# CONFIG
# ======================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-4

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
            raise RuntimeError("No images found in data/meme_train")

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
# MAIN
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

    # ======================
    # Inject LoRA (SAFE VERSION)
    # ======================
    print("Injecting LoRA attention processors...")
    lora_attn_procs = {}

    for name in pipe.unet.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor()

    pipe.unet.set_attn_processor(lora_attn_procs)

    # Enable training only for LoRA
    trainable_params = []
    for p in pipe.unet.parameters():
        if p.requires_grad:
            trainable_params.append(p)

    if len(trainable_params) == 0:
        raise RuntimeError("LoRA injection failed")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # ======================
    # Empty text embedding
    # ======================
    with torch.no_grad():
        tokens = pipe.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        empty_emb = pipe.text_encoder(tokens)[0]

    # ======================
    # Training
    # ======================
    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training LoRA on {len(dataset)} meme images")
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

            encoder_hidden_states = empty_emb.repeat(latents.shape[0], 1, 1)

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

        print(f"Epoch {epoch+1} finished")

    # ======================
    # Save LoRA
    # ======================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_attn_procs(OUTPUT_DIR)

    print(f" LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
