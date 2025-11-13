import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest

df = pd.read_csv("results/logs/requests.csv")

sns.histplot(df, x="latency", hue="mode", bins=40, kde=True)
plt.title("Latency Distribution by Mode")
plt.savefig("results/histograms/latency_hist.png")

stat, p = kstest(df["latency"], "norm")
print("KS test:", stat, p)
