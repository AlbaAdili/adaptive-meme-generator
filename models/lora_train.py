import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

PROMPT = "a meme-style image, internet meme, humorous"
IMAGE_SIZE = 512

BATCH_SIZE = 1
MAX_STEPS = 300      
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class MemeDataset(Dataset):
    def __init__(self, image_dir):
        self.images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)

# --------------------------------------------------
# LOAD PIPELINE
# --------------------------------------------------
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
).to(DEVICE)

pipe.enable_attention_slicing()

# Freeze everything
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)

# --------------------------------------------------
# APPLY LoRA TO UNET
# --------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.05,
    bias="none",
    task_type="UNET",
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------
dataset = MemeDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------------------------------
# TRAINING LOOP (REAL DIFFUSION TRAINING)
# --------------------------------------------------
print("🚀 Starting LoRA fine-tuning...")
step = 0

for epoch in range(1000):  # loop until MAX_STEPS
    for images in loader:
        images = images.to(DEVICE, dtype=DTYPE)

        # Encode images
        latents = pipe.vae.encode(images).latent_dist.sample()
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

        # Text embeddings
        text_inputs = pipe.tokenizer(
            [PROMPT],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(
            text_inputs.input_ids.to(DEVICE)
        )[0]

        # Predict noise
        noise_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        if step % 25 == 0:
            print(f"Step {step}/{MAX_STEPS} | loss={loss.item():.4f}")

        if step >= MAX_STEPS:
            break

    if step >= MAX_STEPS:
        break

# --------------------------------------------------
# SAVE LORA WEIGHTS
# --------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
pipe.unet.save_pretrained(OUTPUT_DIR)

print("LoRA training finished.")
print(f"Weights saved to: {OUTPUT_DIR}")
