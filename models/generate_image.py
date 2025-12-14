import os
from time import time

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# ------------------------------------------------------------
# Device + dtype
# ------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

if DEVICE in ("cuda", "mps"):
    PIPE_DTYPE = torch.float16
else:
    PIPE_DTYPE = torch.float32


class DiffusionGenerator:
    """
    Wrapper around SD1.5 / SDXL with optional LoRA weights.
    """

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        steps: int = 30,
        device: str = DEVICE,
        size: tuple[int, int] = (512, 512),
        lora_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.steps = steps
        self.device = device
        self.size = size

        print(f"Loading model: {model_name} on {device}")

        # SDXL vs SD1.5
        if "xl" in model_name.lower() or "sdxl" in model_name.lower():
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_name,
                torch_dtype=PIPE_DTYPE,
            )
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=PIPE_DTYPE,
            )

        self.pipe.to(device)

        # Optional LoRA weights (fine-tuned on meme dataset)
        if lora_path:
            try:
           
                self.pipe.load_lora_weights(lora_path)
                print(f"Loaded LoRA weights from {lora_path}")
            except Exception as e:
                print(f"Could not load LoRA weights from {lora_path}: {e}")


        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

        # Small memory optimisation
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        os.makedirs("results/images", exist_ok=True)

    # --------------------------------------------------------
    # Generate one image
    # --------------------------------------------------------
    def generate(self, prompt: str, outname: str = "generated.png"):
        start = time()

        result = self.pipe(
            prompt,
            num_inference_steps=self.steps,
            height=self.size[1],
            width=self.size[0],
        )

        image = result.images[0]
        outpath = os.path.join("results", "images", outname)
        image.save(outpath)

        latency = time() - start
        return outpath, latency
