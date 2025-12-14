import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

IMAGE_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 2
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# DATASET (images only)
# ------------------------
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

# ------------------------
# LOAD PIPELINE
# ------------------------
print(" Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to(DEVICE)

pipe.safety_checker = None
pipe.enable_attention_slicing()

noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# ------------------------
# ADD LoRA TO UNET
# ------------------------
print(" Injecting LoRA layers...")
lora_attn_procs = {}

for name, attn in pipe.unet.attn_processors.items():
    lora_attn_procs[name] = LoRAAttnProcessor(
        hidden_size=attn.hidden_size,
        cross_attention_dim=attn.cross_attention_dim,
        rank=8,
    )

pipe.unet.set_attn_processor(lora_attn_procs)
pipe.unet.train()

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

# ------------------------
# TRAINING LOOP (LATENT SPACE)
# ------------------------
dataset = MemeDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f" Training LoRA on {len(dataset)} meme images")

for epoch in range(EPOCHS):
    pbar = tqdm(loader)
    for images in pbar:
        images = images.to(DEVICE, dtype=torch.float16)

        # Encode images → latents (4 channels!)
        latents = pipe.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=DEVICE,
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Dummy text conditioning (we do NOT train text)
        encoder_hidden_states = pipe.text_encoder(
            pipe.tokenizer(
                [""] * latents.shape[0],
                padding="max_length",
                return_tensors="pt",
            ).input_ids.to(DEVICE)
        )[0]

        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} | loss={loss.item():.4f}")

print(" Training finished")

# ------------------------
# SAVE LoRA
# ------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
pipe.unet.save_attn_procs(OUTPUT_DIR)

print(f" LoRA weights saved to: {OUTPUT_DIR}")
