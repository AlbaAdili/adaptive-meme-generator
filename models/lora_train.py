# models/lora_train.py
import os
import inspect
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
# DATASET (images only)
# ----------------------------
class MemeDataset(Dataset):
    def __init__(self, root: str):
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


# ----------------------------
# LoRA processor factory 
# ----------------------------
def make_lora_proc(rank: int):
    """
    Diffusers versions differ:
      - some use LoRAAttnProcessor2_0(rank=...)
      - some use LoRAAttnProcessor2_0(lora_rank=...)
      - some accept no rank arg
    This function adapts automatically.
    """
    sig = inspect.signature(LoRAAttnProcessor2_0.__init__)
    params = sig.parameters

    if "rank" in params:
        return LoRAAttnProcessor2_0(rank=rank)
    if "lora_rank" in params:
        return LoRAAttnProcessor2_0(lora_rank=rank)

    # fallback: no rank argument supported in this build
    return LoRAAttnProcessor2_0()


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

    # ----------------------------
    # Inject LoRA into UNet attention processors
    # ----------------------------
    print("Injecting LoRA into UNet attention processors...")
    lora_attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        lora_attn_procs[name] = make_lora_proc(RANK)
    pipe.unet.set_attn_processor(lora_attn_procs)

trainable_params = []
for name, module in pipe.unet.named_modules():
    if "lora" in name.lower():
        for p in module.parameters():
            p.requires_grad = True
            trainable_params.append(p)

if len(trainable_params) == 0:
    raise RuntimeError("LoRA layers injected but no trainable params found")


    optimizer = torch.optim.AdamW(trainable_params, lr=LR)

    # ----------------------------
    # Prepare fixed text embeddings (empty prompt) for conditioning
    # ----------------------------
    with torch.no_grad():
        tokens = pipe.tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(DEVICE)

        empty_emb = pipe.text_encoder(tokens)[0]  # (1, 77, hidden)

    # ----------------------------
    # Data
    # ----------------------------
    dataset = MemeDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training LoRA on {len(dataset)} images")
    pipe.unet.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE, dtype=DTYPE)

            # Encode images to latents (B, 4, 64, 64)
            with torch.no_grad():
                latents = pipe.vae.encode(batch).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            # Sample noise + timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Repeat empty embedding for batch
            encoder_hidden_states = empty_emb.repeat(latents.shape[0], 1, 1)

            # Predict noise
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

        print(f"Epoch {epoch+1} done")

    # ----------------------------
    # Save LoRA weights
    # ----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Different diffusers versions expose different save APIs
    if hasattr(pipe.unet, "save_attn_procs"):
        pipe.unet.save_attn_procs(OUTPUT_DIR)
    elif hasattr(pipe, "save_lora_weights"):
        pipe.save_lora_weights(OUTPUT_DIR)
    else:
        # last-resort: save full unet (not ideal but won't crash)
        pipe.unet.save_pretrained(OUTPUT_DIR)

    print(f"LoRA saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
