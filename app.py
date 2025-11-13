import asyncio
from queueing.controller import AdaptiveController
from queueing.load_simulator import simulate_load

prompts = [
    "When your code works on first try",
    "POV: You forgot to save your weights",
    "Debugging at 3am",
    "GPU be like: not enough memory",
    "When deadline is tomorrow"
]

async def main():
    controller = AdaptiveController()

    # Start the worker (handles queued meme requests)
    asyncio.create_task(controller.worker())

    # Start load simulation (pushes prompts into the queue)
    asyncio.create_task(simulate_load(controller, prompts, rate=0.4))

    print("Adaptive Meme Generator Running...")
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
