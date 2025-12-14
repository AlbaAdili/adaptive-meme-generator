# models/lora_train.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0

# ----------------------------
# CONFIG
# ----------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-4
RANK = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ----------------------------
# DATASET (IMAGES ONLY)
# ----------------------------
class MemeDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
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

# ----------------------------
# MAIN
# ----------------------------
def main():
    print(f"Device: {DEVICE}")

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

    # ----------------------------
    # Inject LoRA 
    # ----------------------------
    lora_attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor2_0(rank=RANK)

    pipe.unet.set_attn_processor(lora_attn_procs)

    # Optimizer (ONLY LoRA params)
    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)

    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training LoRA on {len(dataset)} images")

    for epoch in range(EPOCHS):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(DEVICE, dtype=DTYPE)

            noise = torch.randn_like(batch)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (batch.shape[0],),
                device=DEVICE,
            ).long()

            latents = pipe.vae.encode(batch).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} done")


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_attn_procs(OUTPUT_DIR)

    print(f" LoRA saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
