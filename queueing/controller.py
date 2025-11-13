import asyncio
import time
import torch

from models.generate_image import DiffusionGenerator
from models.gif_creator import create_gif
from evaluation.measure_metrics import log_request

# Device detection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[AdaptiveController] Using device: {DEVICE}")

HIGH_LOAD = 5
LOW_LOAD = 2


class AdaptiveController:

    def __init__(self):

        self.queue = asyncio.Queue()

        # Fast model (SD1.5)
        self.fast_model = DiffusionGenerator(
            "runwayml/stable-diffusion-v1-5",
            steps=20,
            device=DEVICE,
            size=(512, 512)
        )

        if DEVICE == "mps":
            print("SDXL disabled on Mac MPS — using SD1.5 for both modes")
            self.quality_model = DiffusionGenerator(
                "runwayml/stable-diffusion-v1-5",
                steps=30,
                device=DEVICE,
                size=(512, 512)
            )
        else:
            self.quality_model = DiffusionGenerator(
                "stabilityai/stable-diffusion-xl-base-1.0",
                steps=50,
                device=DEVICE,
                size=(768, 768)
            )

        self.current_mode = "quality"

    async def choose_model(self):
        qsize = self.queue.qsize()

        if qsize >= HIGH_LOAD:
            self.current_mode = "fast"
        elif qsize <= LOW_LOAD:
            self.current_mode = "quality"

        return self.fast_model if self.current_mode == "fast" else self.quality_model

    async def worker(self):
        while True:
            prompt = await self.queue.get()

            model = await self.choose_model()
            mode = "fast" if model == self.fast_model else "quality"

            filename = f"{int(time.time()*1000)}.png"
            path, latency = model.generate(prompt, filename)

            # Adaptive GIF length
            frames = 1 if mode == "fast" else 8

            if frames > 1:
                gif_path = create_gif(
                    [path] * frames,
                    outpath=path.replace(".png", ".gif")
                )
            else:
                gif_path = path

            log_request(prompt, mode, latency, frames)

            print(f"[{mode}] {prompt} — {latency:.2f}s | frames={frames}")

            self.queue.task_done()
