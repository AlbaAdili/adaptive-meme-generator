# models/lora_train.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

BATCH_SIZE = 1
EPOCHS = 2          # keep small (demo-quality)
LR = 1e-4
IMAGE_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ------------------------------------------------------------
# DATASET (images only, captions intentionally ignored)
# ------------------------------------------------------------
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
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image)

# ------------------------------------------------------------
# LOAD PIPELINE
# ------------------------------------------------------------
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
pipe.to(device)
pipe.safety_checker = None

pipe.unet.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# ------------------------------------------------------------
# APPLY LoRA TO UNET (CORRECT WAY)
# ------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

# ------------------------------------------------------------
# TRAINING LOOP (REAL DIFFUSION LOSS)
# ------------------------------------------------------------
dataset = MemeDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training LoRA on {len(dataset)} meme images")

for epoch in range(EPOCHS):
    for images in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device, dtype=dtype)

        # Sample noise
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (images.shape[0],),
            device=device,
        ).long()

        # Add noise
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Predict noise
        noise_pred = pipe.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=None,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} completed")

# ------------------------------------------------------------
# SAVE LoRA WEIGHTS
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
pipe.unet.save_pretrained(OUTPUT_DIR)

print(f"LoRA training finished. Saved to {OUTPUT_DIR}")
