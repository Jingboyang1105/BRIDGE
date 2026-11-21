from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "results" / "main_efficiency"
SRC_FIG = PROJECT / "results" / "main_figures"
OUT = SRC_FIG / "paper_clean_revisions"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#222222",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

TEAL = "#167573"
RUST = "#BA7666"
BLUE = "#4E78A0"
RED = "#C95847"
GREY = "#C9C6BD"
DARK = "#1E252B"
GRID = "#E9E5DD"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def best_so_far(y: pd.Series) -> pd.Series:
    return y.cummin()


def short_tissue(name: str) -> str:
    repl = {
        "Brain - ": "",
        "Esophagus - Gastroesophageal Junction": "Esoph. GEJ",
        "Esophagus - Muscularis": "Esoph. muscularis",
        "Esophagus - Mucosa": "Esoph. mucosa",
        "Small Intestine - Terminal Ileum": "Small intestine",
        "Adipose - Subcutaneous": "Adipose subcut.",
        "Adipose - Visceral (Omentum)": "Adipose visceral",
        "Skin - Sun Exposed (Lower leg)": "Skin sun",
        "Skin - Not Sun Exposed (Suprapubic)": "Skin not sun",
        "Heart - Left Ventricle": "Heart L. vent.",
        "Heart - Atrial Appendage": "Heart atrial app.",
        "Breast - Mammary Tissue": "Breast mammary",
        "Nucleus accumbens (basal ganglia)": "Nucleus accumbens",
        "Caudate (basal ganglia)": "Caudate",
        "Putamen (basal ganglia)": "Putamen",
        "Spinal cord (cervical c-1)": "Spinal cord",
        "Frontal Cortex (BA9)": "Frontal cortex",
        "Anterior cingulate cortex (BA24)": "Ant. cingulate",
        "Cells - Cultured fibroblasts": "Fibroblasts",
        "Cells - EBV-transformed lymphocytes": "EBV cells",
        "Liver - Hepatocyte": "Liver hepatocyte",
        "Pancreas - Mixed Cell": "Pancreas mixed",
        "Pancreas - Acini": "Pancreas acini",
        "Cervix - Ectocervix": "Cervix ecto.",
        "Cervix - Endocervix": "Cervix endo.",
    }
    out = name
    for a, b in repl.items():
        out = out.replace(a, b)
    out = out.replace(" - ", " ")
    return out


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.03, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def training_efficiency() -> None:
    dorian = pd.read_csv(FINAL / "dorian_validation_curve.csv")
    hyfa = pd.read_csv(FINAL / "hyfa_validation_curve.csv")
    summary = pd.read_csv(FINAL / "training_efficiency_summary.csv")
    cml_summary = summary[summary["model"].eq("CML_two_expert_gate")].set_index("metric")["value"]
    hyfa_summary = summary[summary["model"].eq("HYFA_official_b16")].set_index("metric")["value"]
    thresholds = ["0.270", "0.265", "0.260", "0.258", "0.256"]
    mile = pd.DataFrame(
        {
            "milestone": [f"<={t}" for t in thresholds],
            "CML_epoch": [float(cml_summary[f"first_epoch_val_mae_le_{t}"]) for t in thresholds],
            "HYFA_epoch": [float(hyfa_summary[f"first_epoch_val_mae_le_{t}"]) for t in thresholds],
        }
    )
    mile["HYFA_div_CML"] = mile["HYFA_epoch"] / mile["CML_epoch"]
    dorian["best"] = best_so_far(dorian["val_mae"])
    hyfa["best"] = best_so_far(hyfa["val_mae"])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.55})

    ax = axes[0]
    ax.plot(dorian["epoch"], dorian["best"], color=TEAL, lw=1.8, label="DORIAN")
    ax.plot(hyfa["epoch"], hyfa["best"], color=RUST, lw=1.8, label="HYFA")
    ax.scatter([int(dorian.loc[dorian["val_mae"].idxmin(), "epoch"])], [dorian["val_mae"].min()], s=22, color=TEAL, zorder=3)
    ax.scatter([int(hyfa.loc[hyfa["val_mae"].idxmin(), "epoch"])], [hyfa["val_mae"].min()], s=22, color=RUST, zorder=3)
    ax.set_xlim(0, 200)
    ax.set_ylim(0.252, 0.316)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAE")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    panel(ax, "a")

    ax = axes[1]
    y = np.arange(len(mile))[::-1]
    ax.hlines(y, mile["CML_epoch"], mile["HYFA_epoch"], color=GREY, lw=2.2, zorder=1)
    ax.scatter(mile["CML_epoch"], y, s=34, color=TEAL, edgecolor="white", lw=0.6, zorder=2)
    ax.scatter(mile["HYFA_epoch"], y, s=34, color=RUST, edgecolor="white", lw=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(mile["milestone"])
    ax.set_xlabel("first epoch reached")
    ax.set_xlim(0, max(130, mile["HYFA_epoch"].max() + 10))
    ax.grid(axis="x", color=GRID, lw=0.6)
    for yy, ratio, x in zip(y, mile["HYFA_div_CML"], mile["HYFA_epoch"]):
        ax.text(x + 3, yy, f"{ratio:.1f}x", va="center", ha="left", fontsize=6, color="#68727D")
    panel(ax, "b")

    save(fig, "training_efficiency_clean")


def memory_vs_mae() -> None:
    df = pd.read_csv(SRC_FIG / "paper_main_clean" / "main_tissue_memory_vs_mae_source.csv")
    df = df.sort_values("training_memory_records").copy()
    df.to_csv(OUT / "memory_vs_mae_clean_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.8, 3.25))
    x = df["training_memory_records"].clip(lower=1)
    y = df["mae"]
    colors = np.where(df["low_memory"], RED, BLUE)
    sizes = 18 + 38 * (df["n_samples"] / df["n_samples"].max())
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.88, edgecolor="white", lw=0.45, zorder=2)
    ax.axhline(y.mean(), color="#3A3A3A", lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("training-memory records")
    ax.set_ylabel("mean MAE")
    ax.grid(color=GRID, lw=0.55)

    labels = [
        "Liver - Hepatocyte",
        "Pancreas - Acini",
        "Small Intestine - Terminal Ileum - Lymphoid Aggregate",
        "Cervix - Ectocervix",
        "Nerve - Tibial",
        "Esophagus - Muscularis",
        "Cortex",
    ]
    offsets = {
        "Liver - Hepatocyte": (7, 2),
        "Pancreas - Acini": (6, 2),
        "Small Intestine - Terminal Ileum - Lymphoid Aggregate": (5, -8),
        "Cervix - Ectocervix": (5, 5),
        "Nerve - Tibial": (4, -9),
        "Esophagus - Muscularis": (5, -8),
        "Cortex": (5, 4),
    }
    for _, row in df[df["target_tissue_name"].isin(labels)].iterrows():
        dx, dy = offsets[row["target_tissue_name"]]
        ax.annotate(
            short_tissue(row["target_tissue_name"]),
            xy=(row["training_memory_records"], row["mae"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.6,
            color="#2E3439",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#B8B8B8", "shrinkA": 2, "shrinkB": 2},
        )

    ax.scatter([], [], s=28, c=BLUE, edgecolor="white", lw=0.45, label="higher memory")
    ax.scatter([], [], s=28, c=RED, edgecolor="white", lw=0.45, label="lower memory")
    ax.legend(frameon=False, loc="upper right", fontsize=6, handletextpad=0.3)
    save(fig, "memory_vs_mae_clean")


def memory_ranked_bar() -> None:
    df = pd.read_csv(SRC_FIG / "paper_main_clean" / "main_tissue_memory_vs_mae_source.csv")
    df = df.sort_values("mae", ascending=True).copy()
    df.to_csv(OUT / "memory_ranked_bar_clean_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 7.4))
    y = np.arange(len(df))
    colors = np.where(df["low_memory"], RED, BLUE)
    ax.barh(y, df["mae"], color=colors, height=0.68, edgecolor="none")
    ax.axvline(df["mae"].mean(), color="#242424", lw=0.75, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([short_tissue(x) for x in df["target_tissue_name"]], fontsize=4.8)
    ax.set_xlabel("mean MAE")
    ax.set_xlim(0, max(0.58, df["mae"].max() * 1.05))
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.tick_params(axis="y", length=0)

    # Small inline legend, outside the data region.
    ax.scatter([], [], s=24, c=BLUE, label="higher memory")
    ax.scatter([], [], s=24, c=RED, label="lower memory")
    ax.legend(frameon=False, loc="lower right", fontsize=6, handletextpad=0.25)
    save(fig, "memory_ranked_bar_clean")


def main() -> None:
    training_efficiency()
    memory_vs_mae()
    memory_ranked_bar()


if __name__ == "__main__":
    main()
