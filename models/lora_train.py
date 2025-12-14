# models/lora_train.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model_state_dict
import safetensors.torch as st

# =========================
# CONFIG
# =========================
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/meme_train"
OUTPUT_DIR = "lora_weights/meme_lora"

EPOCHS = 2
BATCH_SIZE = 1
LR = 1e-5
RANK = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPE_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# =========================
# DATASET
# =========================
class MemeDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        if not self.files:
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
    print(f"Device: {DEVICE} | pipeline dtype: {PIPE_DTYPE}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=PIPE_DTYPE,
        safety_checker=None,
    ).to(DEVICE)

    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze base model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # =========================
    # ADD PEFT LoRA
    # =========================
    print("Adding PEFT LoRA adapter to UNet...")

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    pipe.unet.add_adapter(lora_config)
    pipe.unet.set_adapter("default")
    pipe.unet.train()

    trainable = []
    for p in pipe.unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
            trainable.append(p)

    print(f"Trainable LoRA params: {sum(p.numel() for p in trainable)}")

    optimizer = torch.optim.AdamW(trainable, lr=LR)

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

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            batch = batch.to(DEVICE, dtype=PIPE_DTYPE)

            with torch.no_grad():
                latents = pipe.vae.encode(batch).latent_dist.sample()
                latents *= pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = empty_emb.repeat(latents.shape[0], 1, 1)

            optimizer.zero_grad(set_to_none=True)

            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    # =========================
    # SAVE LoRA (CORRECT)
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    st.save_file(
        lora_state_dict,
        os.path.join(OUTPUT_DIR, "adapter_model.safetensors"),
    )

    pipe.unet.peft_config["default"].save_pretrained(OUTPUT_DIR)

    print(" LoRA training finished")
    print(f"Saved to: {OUTPUT_DIR}")
    print("Files:")
    print(" - adapter_model.safetensors")
    print(" - adapter_config.json")


if __name__ == "__main__":
    main()
