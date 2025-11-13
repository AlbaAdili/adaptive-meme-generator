import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from time import time
import os

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# dtype
if DEVICE in ("cuda", "mps"):
    PIPE_DTYPE = torch.float16
else:
    PIPE_DTYPE = torch.float32


class DiffusionGenerator:
    """
    Wrapper for SD1.5 or SDXL image generation.
    """

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5",
                 steps=30, device=DEVICE, size=(512, 512)):

        self.model_name = model_name
        self.steps = steps
        self.device = device
        self.size = size

        print(f"Loading model: {model_name} on {device}")

        # Detect SDXL and load the correct pipeline
        if "xl" in model_name.lower() or "sdxl" in model_name.lower():
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=PIPE_DTYPE,
            ).to(device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=PIPE_DTYPE,
            ).to(device)

        os.makedirs("results/images", exist_ok=True)

    def generate(self, prompt, outname="generated.png"):
        start = time()

        result = self.pipe(
            prompt,
            num_inference_steps=self.steps,
            height=self.size[1],
            width=self.size[0],
        )

        image = result.images[0]
        outpath = f"results/images/{outname}"
        image.save(outpath)

        return outpath, time() - start
