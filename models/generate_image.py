import torch
from diffusers import StableDiffusionPipeline
from time import time
import os

class DiffusionGenerator:
    """
    Wrapper for SD1.5 or SDXL image generation.
    """

    def __init__(self, model_name="runwayml/stable-diffusion-v1-5",
                 steps=30, device="cuda", size=(512, 512)):

        self.model_name = model_name
        self.steps = steps
        self.device = device
        self.size = size

        print(f"Loading model: {model_name}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)

        os.makedirs("results/images", exist_ok=True)

    def generate(self, prompt, outname="generated.png"):
        start = time()

        image = self.pipe(
            prompt,
            num_inference_steps=self.steps,
            height=self.size[1],
            width=self.size[0],
        ).images[0]

        outpath = f"results/images/{outname}"
        image.save(outpath)

        return outpath, time() - start
