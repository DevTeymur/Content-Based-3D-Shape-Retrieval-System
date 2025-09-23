import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def read_stats(stats_path="stats",files=["original_stats.csv","resampled_stats.csv"]):
    stats_path = Path(stats_path)
    for f in files:
        f = Path(f)
        df = pd.read_csv(stats_path / f)
        s = df['num_vertices']

        s.plot(kind="hist", bins=100, edgecolor="black")
        plt.savefig(stats_path / f"histogram_{f.name}.png", dpi=300, bbox_inches="tight")

        plt.clf()

if __name__=="__main__":
    read_stats("stats")
