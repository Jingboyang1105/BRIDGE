from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "03_How_DORIAN_Works"
    / "data"
    / "source"
    / "adaptive_prediction_mixer_behavior_known_count_source.csv"
)
OUT = ROOT / "03_How_DORIAN_Works" / "figures" / "final"


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    data = pd.read_csv(SOURCE).sort_values("query_known_count")
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(
        data["query_known_count"],
        data["mean_cml_weight"],
        color="#14706E",
        marker="o",
        markersize=3.0,
        linewidth=1.4,
    )
    ax.set_xlabel("Number of observed tissues")
    ax.set_ylabel("Mean memory-retrieval expert weight")
    ax.set_ylim(0.40, 0.54)
    ax.set_yticks(np.arange(0.40, 0.541, 0.02))
    ax.grid(axis="both", color="#E8E8E8", linewidth=0.6)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.96, bottom=0.14)

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "expert2_weight_by_observed_tissues"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
