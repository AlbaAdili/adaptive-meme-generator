import csv
import os
from datetime import datetime
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
os.makedirs("results/logs", exist_ok=True)
LOGFILE = "results/logs/requests.csv"

# ------------------------------------------------------------------
# CLIP model (for image–text quality)
# ------------------------------------------------------------------

if torch.cuda.is_available():
    _CLIP_DEVICE = "cuda"
else:
    _CLIP_DEVICE = "cpu"

_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROCESSOR: Optional[CLIPProcessor] = None


def _load_clip():
    """Lazy-load CLIP once and reuse."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL.to(_CLIP_DEVICE)
        _CLIP_MODEL.eval()


def compute_clip_score(prompt: str, image_path: str) -> float:
    """Return CLIP image–text similarity (higher ≈ better alignment)."""
    if not image_path or not os.path.exists(image_path):
        return float("nan")

    _load_clip()

    image = Image.open(image_path).convert("RGB")
    inputs = _CLIP_PROCESSOR(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(_CLIP_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _CLIP_MODEL(**inputs)
        logits_per_image = outputs.logits_per_image  # shape [1, 1]
        score = logits_per_image[0, 0].item()
    return float(score)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------


def log_request(
    prompt: str,
    mode: str,
    latency: float,
    gif_frames: int,
    image_path: Optional[str] = None,
    queue_len: Optional[int] = None,
    arrival_ts: Optional[float] = None,
    start_ts: Optional[float] = None,
) -> None:
    """Append one row to results/logs/requests.csv.

    Columns:
        timestamp   – wall-clock time when logging happens
        arrival_ts  – when the job was enqueued (simulation time, seconds)
        start_ts    – when generation started
        prompt      – text prompt
        mode        – "fast" or "quality"
        latency     – generation latency (seconds)
        gif_frames  – number of frames in the GIF
        queue_len   – #jobs in system when this job started
        clip_score  – CLIP image–text similarity (quality metric)
        image_path  – where the image/GIF is stored
    """
    file_exists = os.path.isfile(LOGFILE)

    clip_score = compute_clip_score(prompt, image_path) if image_path else float("nan")

    with open(LOGFILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "arrival_ts",
                    "start_ts",
                    "prompt",
                    "mode",
                    "latency",
                    "gif_frames",
                    "queue_len",
                    "clip_score",
                    "image_path",
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                arrival_ts if arrival_ts is not None else "",
                start_ts if start_ts is not None else "",
                prompt,
                mode,
                latency,
                gif_frames,
                queue_len if queue_len is not None else "",
                clip_score,
                image_path or "",
            ]
        )
