import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "lora_weights/meme_lora"

PROMPTS = [
    "When your code works on first try",
    "POV: You forgot to save your weights",
    "Debugging at 3am",
    "GPU be like: not enough memory",
    "When deadline is tomorrow",
]

OUTPUT_DIR = "results/lora_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --------------------------------------------------
# LOAD CLIP
# --------------------------------------------------
print(" Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --------------------------------------------------
# LOAD BASE PIPELINE
# --------------------------------------------------
print("Loading base SD model...")
base_pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None,
).to(DEVICE)

# --------------------------------------------------
# LOAD LORA PIPELINE
# --------------------------------------------------
print("Loading LoRA SD model...")
lora_pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    safety_checker=None,
).to(DEVICE)

from peft import PeftModel

print("Attaching PEFT LoRA to UNet...")
lora_pipe.unet = PeftModel.from_pretrained(
    lora_pipe.unet,
    LORA_PATH,
)
lora_pipe.unet.eval()


# --------------------------------------------------
# CLIP SCORE FUNCTION
# --------------------------------------------------
def clip_score(image: Image.Image, text: str) -> float:
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()

    return score

# --------------------------------------------------
# RUN COMPARISON
# --------------------------------------------------
results = []

print("Running base vs LoRA comparison...")

for prompt in tqdm(PROMPTS):
    # BASE
    base_img = base_pipe(prompt, num_inference_steps=30).images[0]
    base_path = os.path.join(
        OUTPUT_DIR, f"base_{prompt.replace(' ', '_')}.png"
    )
    base_img.save(base_path)

    base_clip = clip_score(base_img, prompt)

    # LORA
    lora_img = lora_pipe(prompt, num_inference_steps=30).images[0]
    lora_path = os.path.join(
        OUTPUT_DIR, f"lora_{prompt.replace(' ', '_')}.png"
    )
    lora_img.save(lora_path)

    lora_clip = clip_score(lora_img, prompt)

    results.append({
        "prompt": prompt,
        "base_clip": base_clip,
        "lora_clip": lora_clip,
        "delta_clip": lora_clip - base_clip,
    })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "clip_comparison.csv")
df.to_csv(csv_path, index=False)

print("\nComparison complete")
print(df)
print(f"\nResults saved to {csv_path}")
