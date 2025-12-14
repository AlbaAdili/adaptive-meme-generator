import asyncio
import time

import torch

from models.generate_image import DiffusionGenerator
from models.gif_creator import create_gif
from evaluation.measure_metrics import log_request

# ------------------------------------------------------------
# Device detection
# ------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[AdaptiveController] Using device: {DEVICE}")

HIGH_LOAD = 5   # queue length at which we switch to fast mode
LOW_LOAD = 2    # queue length below which we go back to quality mode


class AdaptiveController:
    """Queue-based controller that switches models based on load."""

    def __init__(self, fast_lora: str | None = None, quality_lora: str | None = None):
        self.queue: asyncio.Queue = asyncio.Queue()

        # Fast model (SD 1.5)
        self.fast_model = DiffusionGenerator(
            "runwayml/stable-diffusion-v1-5",
            steps=20,
            device=DEVICE,
            size=(512, 512),
            lora_path=fast_lora,
        )

        # Quality model (SDXL on CUDA, SD1.5 on Mac)
        if DEVICE == "mps":
            print("⚠️ SDXL disabled on Mac MPS — using SD1.5 for both modes")
            self.quality_model = DiffusionGenerator(
                "runwayml/stable-diffusion-v1-5",
                steps=30,  # more steps → higher quality
                device=DEVICE,
                size=(512, 512),
                lora_path=quality_lora or fast_lora,
            )
        else:
            self.quality_model = DiffusionGenerator(
                "stabilityai/stable-diffusion-xl-base-1.0",
                steps=50,
                device=DEVICE,
                size=(768, 768),
                lora_path=quality_lora,
            )

        self.current_mode: str = "quality"

    # --------------------------------------------------------
    # Model selection logic
    # --------------------------------------------------------
    async def choose_model(self) -> DiffusionGenerator:
        qsize = self.queue.qsize()

        if qsize >= HIGH_LOAD:
            self.current_mode = "fast"
        elif qsize <= LOW_LOAD:
            self.current_mode = "quality"

        return self.fast_model if self.current_mode == "fast" else self.quality_model

    # --------------------------------------------------------
    # Worker loop
    # --------------------------------------------------------
    async def worker(self):
        while True:
            job = await self.queue.get()

            # Jobs come from load_simulator as dicts
            if isinstance(job, dict):
                prompt = job.get("prompt", "")
                arrival_ts = job.get("arrival_ts", None)
                job_id = job.get("id", None)
            else:
                # Backwards compatibility for plain strings
                prompt = str(job)
                arrival_ts = None
                job_id = None

            # Number of jobs in system (this one + ones still waiting)
            queue_len = self.queue.qsize() + 1

            start_ts = time.time()
            model = await self.choose_model()
            mode = "fast" if model is self.fast_model else "quality"

            filename = f"{int(start_ts * 1000)}.png"
            image_path, model_latency = model.generate(prompt, filename)

            # Adaptive GIF length
            frames = 1 if mode == "fast" else 8

            if frames > 1:
                gif_path = create_gif(
                    [image_path] * frames,
                    outpath=image_path.replace(".png", ".gif"),
                )
            else:
                gif_path = image_path

            total_latency = model_latency

            # Log metrics (including CLIP quality & queue stats)
            try:
                log_request(
                    prompt=prompt,
                    mode=mode,
                    latency=total_latency,
                    gif_frames=frames,
                    image_path=gif_path,
                    queue_len=queue_len,
                    arrival_ts=arrival_ts,
                    start_ts=start_ts,
                )
            except TypeError:
                # Fallback if an older log_request signature is used
                log_request(prompt, mode, total_latency, frames)

            jid = f" job={job_id}" if job_id is not None else ""
            print(
                f"[{mode}] prompt='{prompt}'{jid} — "
                f"{total_latency:.2f}s | frames={frames} | queue={queue_len}"
            )

            self.queue.task_done()
