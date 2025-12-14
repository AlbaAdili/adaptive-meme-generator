import asyncio
import random
import time


async def simulate_load(
    controller,
    prompts,
    rate: float = 0.5,
    max_jobs: int | None = None,
):
    """Generate jobs with exponential inter-arrival times.

    Args:
        controller: AdaptiveController instance (must expose .queue).
        prompts: list of prompt strings.
        rate: lambda for the exponential distribution (jobs per second).
        max_jobs: optional cap on number of jobs to send (None = infinite).
    """
    job_id = 0

    while True:
        if max_jobs is not None and job_id >= max_jobs:
            break

        prompt = random.choice(prompts)
        arrival_ts = time.time()

        job = {
            "id": job_id,
            "prompt": prompt,
            "arrival_ts": arrival_ts,
        }
        await controller.queue.put(job)
        job_id += 1

        # Exponential inter-arrival time (what your prof asked)
        dt = random.expovariate(rate)  # mean 1/rate seconds
        await asyncio.sleep(dt)
