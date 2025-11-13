import csv
import os
from datetime import datetime

os.makedirs("results/logs", exist_ok=True)
LOGFILE = "results/logs/requests.csv"

def log_request(prompt, mode, latency, gif_frames):
    file_exists = os.path.isfile(LOGFILE)
    with open(LOGFILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "prompt", "mode", "latency", "gif_frames"])
        w.writerow([datetime.now(), prompt, mode, latency, gif_frames])
