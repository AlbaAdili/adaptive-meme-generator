import asyncio
import random

async def simulate_load(controller, prompts, rate=0.6):
    """
    Simulates random users sending meme requests.
    """
    while True:
        await controller.queue.put(random.choice(prompts))
        await asyncio.sleep(random.expovariate(rate))
