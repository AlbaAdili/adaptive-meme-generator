from PIL import Image
import os

def create_gif(frame_paths, outpath="results/gifs/out.gif", duration=150):

    os.makedirs("results/gifs", exist_ok=True)
    
    frames = [Image.open(p) for p in frame_paths]

    frames[0].save(
        outpath,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    return outpath
