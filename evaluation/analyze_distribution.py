import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, expon

LOGFILE = "results/logs/requests.csv"


def main():
    if not os.path.exists(LOGFILE):
        raise SystemExit(f"No log file found at {LOGFILE}. Run app.py first.")

    df = pd.read_csv(LOGFILE)

    # ------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------
    print("\n=== Overall summary ===")
    print(df[["latency", "gif_frames", "queue_len", "clip_score"]].describe())

    print("\n=== Per-mode latency (s) ===")
    print(df.groupby("mode")["latency"].describe())

    if "clip_score" in df.columns:
        print("\n=== Per-mode CLIP score ===")
        print(df.groupby("mode")["clip_score"].describe())

    os.makedirs("results/plots", exist_ok=True)

    # ------------------------------------------------------------
    # Inter-arrival time analysis (online part)
    # ------------------------------------------------------------
    if "arrival_ts" in df.columns and df["arrival_ts"].notna().sum() > 1:
        ts = df["arrival_ts"].dropna().values
        ts = np.sort(ts)
        inter = np.diff(ts)
        print(f"\nMean inter-arrival time: {inter.mean():.3f} s  (N={len(inter)})")

        # KS test vs exponential (what she mentioned in the meeting)
        if len(inter) > 5:
            loc = 0.0
            scale = inter.mean()
            ks_stat, pval = kstest(inter, "expon", args=(loc, scale))
            print(f"KS test vs Exp(mean={scale:.3f}): stat={ks_stat:.3f}, p={pval:.3f}")

        plt.figure()
        sns.histplot(inter, bins=20, kde=True)
        plt.xlabel("Inter-arrival time (s)")
        plt.ylabel("Count")
        plt.title("Inter-arrival time distribution")
        plt.tight_layout()
        plt.savefig("results/plots/inter_arrival_hist.png")

    # ------------------------------------------------------------
    # Latency distribution
    # ------------------------------------------------------------
    plt.figure()
    sns.histplot(
        df,
        x="latency",
        hue="mode",
        bins=20,
        kde=True,
        stat="density",
        common_norm=False,
    )
    plt.xlabel("Latency (s)")
    plt.title("Latency distribution by mode")
    plt.tight_layout()
    plt.savefig("results/plots/latency_hist.png")

    # ------------------------------------------------------------
    # Quality vs latency
    # ------------------------------------------------------------
    if "clip_score" in df.columns:
        plt.figure()
        sns.scatterplot(data=df, x="latency", y="clip_score", hue="mode")
        plt.xlabel("Latency (s)")
        plt.ylabel("CLIP score")
        plt.title("Quality vs. latency")
        plt.tight_layout()
        plt.savefig("results/plots/latency_vs_clip.png")

    print("\nSaved plots to results/plots/.")


if __name__ == "__main__":
    main()
