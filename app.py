import asyncio
from queue.controller import AdaptiveController
from queue.load_simulator import simulate_load

prompts = [
    "When your code works on first try",
    "POV: You forgot to save your weights",
    "Debugging at 3am",
    "GPU be like: not enough memory",
    "When deadline is tomorrow"
]

async def main():
    controller = AdaptiveController()

    # Start worker
    asyncio.create_task(controller.worker())

    # Start load simulation
    asyncio.create_task(simulate_load(controller, prompts, rate=0.4))

    print("Adaptive Meme Generator Running...")
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
