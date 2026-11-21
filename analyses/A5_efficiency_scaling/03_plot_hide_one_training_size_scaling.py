"""Build the paper-ready formal hide-one training-size scaling figure and table."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hide_one_training_size_scaling_corrected"
FIGURES = ROOT / "figures" / "final"
INPUT = RESULTS / "hide_one_training_size_scaling_overall.csv"
EXPECTED_HASH = "7b6e0fcfce336574b329b5455fac190666bbd3b03c9d9838e3d7550a9dd76746"

METHOD_NAMES = {
    "two_expert_cml_gate": "DORIAN",
    "same_donor_residual": "Same-donor expert",
    "cml_retrieval": "Memory expert",
}
METHOD_ORDER = ["DORIAN", "Same-donor expert", "Memory expert"]
COLORS = {
    "DORIAN": "#167A78",
    "Same-donor expert": "#BC7565",
    "Memory expert": "#8172CF",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    expected_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    if sorted(data["fraction"].unique().tolist()) != expected_fractions:
        raise ValueError("Expected exactly the five locked training fractions")
    if len(data) != 15 or set(data["method"]) != set(METHOD_NAMES):
        raise ValueError("Expected three methods at each of five fractions")
    if not data.groupby("fraction").size().eq(3).all():
        raise ValueError("Each fraction must contain exactly three methods")
    if not data["n_samples"].eq(3922).all() or not data["test_records"].eq(3922).all():
        raise ValueError("Every row must use the locked 3,922 test records")
    if not data["n_entries"].eq(78_440_000).all():
        raise ValueError("Every row must contain 3,922 x 20,000 evaluated entries")
    if data["test_record_hash"].nunique() != 1 or data["test_record_hash"].iloc[0] != EXPECTED_HASH:
        raise ValueError("Test-record identity does not match the formal Table 1 task")
    if not data["evaluation_protocol"].eq("hide_one").all():
        raise ValueError("Only the formal hide-one protocol is allowed")
    numeric = data.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("All numerical outputs must be finite")
    data = data.copy()
    data["Method"] = data["method"].map(METHOD_NAMES)
    data["Training fraction"] = (100 * data["fraction"]).round().astype(int)
    return data


def export_complete_table(data: pd.DataFrame) -> None:
    table = data.rename(columns={
        "train_donors": "Training donors",
        "train_records": "Training hide-one records",
        "test_records": "Test records",
        "n_entries": "Evaluated entries",
        "residual_sample_pearson_mean": "Residual sample Pearson mean",
        "residual_sample_pearson_median": "Residual sample Pearson median",
        "mae": "MAE",
        "rmse": "RMSE",
        "hyfa_sample_pearson_mean": "Sample Pearson mean",
        "hyfa_sample_pearson_median": "Sample Pearson median",
        "hyfa_gene_pearson_mean": "Gene Pearson mean",
        "hyfa_gene_pearson_median": "Gene Pearson median",
        "evaluation_protocol": "Evaluation protocol",
        "test_record_hash": "Test-record SHA-256",
        "epochs": "Encoder epochs",
        "batch": "Encoder batch",
        "mixer_epochs": "Mixer epochs",
        "mixer_batch": "Mixer batch",
    })
    columns = [
        "Training fraction", "Method", "Training donors", "Training hide-one records",
        "Test records", "Evaluated entries", "Residual sample Pearson mean",
        "Residual sample Pearson median", "MAE", "RMSE", "Sample Pearson mean",
        "Sample Pearson median", "Gene Pearson mean", "Gene Pearson median",
        "Evaluation protocol", "Test-record SHA-256", "Encoder epochs", "Encoder batch",
        "Mixer epochs", "Mixer batch",
    ]
    table = table[columns].sort_values(
        ["Training fraction", "Method"],
        key=lambda series: series.map({name: index for index, name in enumerate(METHOD_ORDER)})
        if series.name == "Method" else series,
    )
    table.to_csv(RESULTS / "Table_hide_one_training_size_scaling_complete.csv", index=False)
    table.to_csv(RESULTS / "Figure_hide_one_training_size_scaling_source_data.csv", index=False)

    paper = table.pivot(
        index=["Training fraction", "Training donors", "Training hide-one records"],
        columns="Method",
        values=["MAE", "Residual sample Pearson mean"],
    )
    paper.columns = [f"{method} {metric}" for metric, method in paper.columns]
    paper = paper.reset_index()
    paper = paper[[
        "Training fraction", "Training donors", "Training hide-one records",
        "Same-donor expert MAE", "Same-donor expert Residual sample Pearson mean",
        "Memory expert MAE", "Memory expert Residual sample Pearson mean",
        "DORIAN MAE", "DORIAN Residual sample Pearson mean",
    ]]
    paper.to_csv(RESULTS / "Table_hide_one_training_size_scaling_paper_ready.csv", index=False)


def draw_figure(data: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    panels = [
        ("residual_sample_pearson_mean", "Residual Pearson", (0.20, 0.46)),
        ("mae", "MAE", (0.24, 0.285)),
    ]
    for ax, (metric, ylabel, ylim), panel in zip(axes, panels, ["a", "b"]):
        for method in METHOD_ORDER:
            subset = data.loc[data["Method"] == method].sort_values("Training fraction")
            ax.plot(
                subset["Training fraction"], subset[metric],
                color=COLORS[method], marker="o", markersize=4.2,
                linewidth=1.8, label=method,
            )
        ax.set_xlabel("Training donors used (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_xlim(7, 103)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#E7E3DE", linewidth=0.6)
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=3, handlelength=2.0, columnspacing=1.4)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.84, wspace=0.34)
    stem = FIGURES / "hide_one_training_size_scaling"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    data = load_and_validate()
    export_complete_table(data)
    draw_figure(data)
    print(FIGURES / "hide_one_training_size_scaling.svg")


if __name__ == "__main__":
    main()
