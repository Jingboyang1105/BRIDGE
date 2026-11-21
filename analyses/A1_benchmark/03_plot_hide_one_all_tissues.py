"""Full 65-tissue GTEx v11 hide-one bar plots for the manuscript."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
METHODS = ["DORIAN", "HYFA", "PMM"]
COLORS = {"DORIAN": "#6C5CE7", "HYFA": "#F2B84B", "PMM": "#38B2A3"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def short_name(name: str) -> str:
    replacements = {
        "Brain - ": "Brain: ",
        "Cells - ": "Cells: ",
        "Adipose - ": "Adipose: ",
        "Artery - ": "Artery: ",
        "Breast - ": "Breast: ",
        "Colon - ": "Colon: ",
        "Esophagus - ": "Esophagus: ",
        "Heart - ": "Heart: ",
        "Kidney - ": "Kidney: ",
        "Skin - ": "Skin: ",
        "Small Intestine - ": "Small intestine: ",
        "Cervix - ": "Cervix: ",
        "Stomach - ": "Stomach: ",
        "Liver - ": "Liver: ",
        "Pancreas - ": "Pancreas: ",
    }
    out = str(name)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def load_records() -> pd.DataFrame:
    specs = [
        (
            "DORIAN",
            ROOT / "02_Benchmark_vs_HYFA_PMM/results/hide_one/hide_one_test_200e_cml_record_scores_for_boxplots.csv",
        ),
        (
            "HYFA",
            ROOT / "02_Benchmark_vs_HYFA_PMM/results/hide_one/hide_one_test_200e_hyfa_record_scores_for_boxplots.csv",
        ),
        (
            "PMM",
            ROOT / "02_Benchmark_vs_HYFA_PMM/results/pmm_v11_seed42/formal_olga/hide_one/pmm_record_scores.csv",
        ),
    ]
    frames = []
    for method, path in specs:
        df = pd.read_csv(path)
        if method == "DORIAN":
            df = df[
                (df["method"] == "CML gate")
                & (df["prediction_key"] == "pred_two_expert_cml_gate")
            ].copy()
        df["method"] = method
        frames.append(
            df[
                [
                    "record_id",
                    "target_tissue",
                    "target_tissue_name",
                    "method",
                    "mae",
                    "residual_sample_pearson",
                ]
            ].copy()
        )
    records = pd.concat(frames, ignore_index=True)
    records["method"] = pd.Categorical(records["method"], METHODS, ordered=True)
    records["target_tissue"] = records["target_tissue"].astype(int)
    records["mae"] = pd.to_numeric(records["mae"], errors="raise")
    records["residual_sample_pearson"] = pd.to_numeric(
        records["residual_sample_pearson"], errors="raise"
    )
    if records.groupby("method", observed=True)["record_id"].nunique().to_dict() != {
        "DORIAN": 3922,
        "HYFA": 3922,
        "PMM": 3922,
    }:
        raise ValueError("Record coverage differs across methods")
    if records.groupby("method", observed=True)["target_tissue"].nunique().nunique() != 1:
        raise ValueError("Tissue coverage differs across methods")
    return records


def build_source(records: pd.DataFrame) -> pd.DataFrame:
    source = (
        records.groupby(
            ["target_tissue", "target_tissue_name", "method"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_records=("record_id", "nunique"),
            mae=("mae", "mean"),
            residual_sample_pearson=("residual_sample_pearson", "mean"),
        )
    )
    order = (
        source[source["method"] == "DORIAN"]
        .sort_values("residual_sample_pearson", ascending=False)["target_tissue"]
        .tolist()
    )
    if len(order) != 65:
        raise ValueError(f"Expected 65 tissues, found {len(order)}")
    order_map = {tissue: rank for rank, tissue in enumerate(order)}
    source["tissue_order"] = source["target_tissue"].map(order_map)
    source["display_tissue"] = source["target_tissue_name"].map(short_name)
    source = source.sort_values(["tissue_order", "method"]).reset_index(drop=True)
    if len(source) != 65 * 3:
        raise ValueError("Incomplete tissue-method grid")
    if not np.isfinite(source[["mae", "residual_sample_pearson"]].to_numpy()).all():
        raise ValueError("Non-finite values in plot source")
    return source


def save(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".png": {"dpi": 300},
        ".svg": {},
        ".pdf": {},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT / f"{stem}{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_metric(source: pd.DataFrame, metric: str, ylabel: str, stem: str) -> None:
    tissue = (
        source[["tissue_order", "display_tissue"]]
        .drop_duplicates()
        .sort_values("tissue_order")
    )
    labels = tissue["display_tissue"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15.8, 5.7))
    for idx, method in enumerate(METHODS):
        block = source[source["method"] == method].sort_values("tissue_order")
        ax.bar(
            x + (idx - 1) * width,
            block[metric].to_numpy(),
            width=width,
            color=COLORS[method],
            edgecolor="none",
            label=method,
            zorder=3,
        )

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=5.1)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#E6E6EE", linewidth=0.55, zorder=0)
    ax.margins(x=0.002)
    if metric == "residual_sample_pearson":
        low = min(-0.1, float(source[metric].min()) - 0.03)
        high = max(0.55, float(source[metric].max()) + 0.03)
        ax.set_ylim(low, high)
        ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", zorder=2)
    else:
        ax.set_ylim(0, max(0.95, float(source[metric].max()) * 1.06))

    handles = [Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=7.5,
        handlelength=1.8,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.39, top=0.91)
    save(fig, stem)


def plot_violin_metric(records: pd.DataFrame, source: pd.DataFrame, metric: str, ylabel: str, stem: str) -> None:
    tissue = (
        source[["target_tissue", "tissue_order", "display_tissue"]]
        .drop_duplicates()
        .sort_values("tissue_order")
    )
    tissues = tissue["target_tissue"].tolist()
    labels = tissue["display_tissue"].tolist()
    order_map = dict(zip(tissues, tissue["tissue_order"]))
    records = records.copy()
    records["tissue_order"] = records["target_tissue"].map(order_map)
    y = np.arange(len(tissues))
    offsets = np.array([-0.27, 0.0, 0.27])
    fig, ax = plt.subplots(figsize=(9.2, 18.0))
    for method_index, method in enumerate(METHODS):
        distributions = [
            records[(records["target_tissue"] == t) & (records["method"] == method)][metric].to_numpy()
            for t in tissues
        ]
        if any(len(values) == 0 for values in distributions):
            raise ValueError(f"Empty {method} distribution for {metric}")
        violin = ax.violinplot(
            distributions, positions=y + offsets[method_index], widths=0.25,
            vert=False, showmeans=False, showmedians=False, showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_linewidth(0.25)
            body.set_alpha(0.88)
        medians = [np.median(values) for values in distributions]
        ax.scatter(
            medians, y + offsets[method_index], s=7, color="white",
            edgecolor=COLORS[method], linewidth=0.45, zorder=3,
        )
    ax.set_xlabel(ylabel, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.7)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#E6E6EE", linewidth=0.55, zorder=0)
    ax.margins(y=0.003)
    if metric == "residual_sample_pearson":
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--", zorder=2)
    else:
        ax.set_xlim(0, max(1.0, float(records[metric].max()) * 1.03))
    ax.legend(
        handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005), frameon=False,
        fontsize=7.5, handlelength=1.8, columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.33, right=0.99, bottom=0.045, top=0.975)
    save(fig, stem)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    source = build_source(records)
    source.to_csv(OUT / "hide_one_full_65_tissue_bar_source.csv", index=False)
    records.to_csv(OUT / "hide_one_full_65_tissue_violin_source.csv", index=False)
    plot_metric(
        source,
        "residual_sample_pearson",
        "Residual Pearson correlation",
        "hide_one_full_65_tissue_residual_bar",
    )
    plot_metric(source, "mae", "MAE", "hide_one_full_65_tissue_mae_bar")
    plot_violin_metric(
        records, source, "residual_sample_pearson", "Residual Pearson correlation",
        "hide_one_full_65_tissue_residual_violin",
    )
    plot_violin_metric(records, source, "mae", "MAE", "hide_one_full_65_tissue_mae_violin")
    print(
        f"Complete: {source['target_tissue'].nunique()} tissues, "
        f"{source['method'].nunique()} methods, {len(source)} source rows"
    )


if __name__ == "__main__":
    main()
