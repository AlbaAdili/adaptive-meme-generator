
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor2_0

# ---------------- CONFIG ----------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 1
BATCH_SIZE = 1
LR = 1e-4
RANK = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------- DATASET ----------------
class MemeDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        if not self.files:
            raise RuntimeError("No images found")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB"))


def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None
    ).to(DEVICE)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze base
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # -------- Inject LoRA (diffusers-native) --------
    print("Injecting LoRA attention processors...")
    lora_procs = {
        name: LoRAAttnProcessor2_0(rank=RANK)
        for name in pipe.unet.attn_processors.keys()
    }
    pipe.unet.set_attn_processor(lora_procs)

    # Collect trainable params
    trainable_params = []
    for _, p in pipe.unet.named_parameters():
        if p.requires_grad:
            trainable_params.append(p)

    print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # Empty prompt embedding
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
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = batch.to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                latents = pipe.vae.encode(batch).latent_dist.sample()
                latents *= pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.size(0),),
                device=DEVICE
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = empty_emb.repeat(latents.size(0), 1, 1)

           
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipe.unet.save_attn_procs(OUTPUT_DIR)
    print(f" LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
