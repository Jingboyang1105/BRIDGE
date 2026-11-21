from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parents[2]
OLD_RECORDS = ROOT / "02_Benchmark_vs_HYFA_PMM" / "figures" / "brain_tissues" / "tissue_violin_source_mae_residual.csv"
NEW_RECORDS = ROOT / "09_Accessible_TEEBoT_Benchmark" / "results" / "tables" / "Table_accessible_record_metrics.csv"
OUT = ROOT / "09_Accessible_TEEBoT_Benchmark" / "results" / "figures"

METHODS = ["DORIAN", "HYFA", "PMM", "TEEBoT"]
COLORS = {
    "DORIAN": "#6C5CE7",
    "HYFA": "#F2B84B",
    "PMM": "#36B8A5",
    "TEEBoT": "#4C78A8",
}


def build_source() -> pd.DataFrame:
    old = pd.read_csv(OLD_RECORDS)
    old = old.loc[
        old["protocol"].eq("accessible") & old["method"].isin(METHODS[:3])
    ].copy()

    new = pd.read_csv(NEW_RECORDS)
    teebot = new.loc[new["method"].eq("TEEBoT_accessible_age_sex")].copy()
    teebot = teebot.rename(columns={
        "target_tissue_id": "target_tissue",
        "residual_pearson": "residual_sample_pearson",
    })
    teebot["protocol"] = "accessible"
    teebot["method"] = "TEEBoT"

    meta = (
        old.loc[old["method"].eq("DORIAN"), [
            "record_id", "target_tissue", "target_tissue_name", "display_tissue", "tissue_order"
        ]]
        .drop_duplicates("record_id")
    )
    teebot = teebot.merge(
        meta,
        on=["record_id", "target_tissue", "target_tissue_name"],
        how="inner",
        validate="one_to_one",
    )
    teebot = teebot[[
        "protocol", "method", "record_id", "target_tissue", "target_tissue_name",
        "mae", "residual_sample_pearson", "display_tissue", "tissue_order",
    ]]

    source = pd.concat([old, teebot], ignore_index=True)
    source["method"] = pd.Categorical(source["method"], METHODS, ordered=True)
    source = source.sort_values(["tissue_order", "method", "record_id"]).reset_index(drop=True)

    expected_records = set(meta["record_id"].astype(str))
    for method in METHODS:
        observed = set(source.loc[source["method"].eq(method), "record_id"].astype(str))
        if observed != expected_records:
            raise ValueError(f"{method} does not use the exact shared accessible records")
    if len(source) != 1270 * 4:
        raise ValueError(f"Expected 5,080 method-record rows, found {len(source):,}")
    if not np.isfinite(source[["mae", "residual_sample_pearson"]].to_numpy()).all():
        raise ValueError("Non-finite record metric detected")
    return source


def draw_violin(ax: plt.Axes, source: pd.DataFrame, metric: str, ylabel: str) -> None:
    tissue_rows = (
        source.loc[source["method"].eq("DORIAN")]
        .sort_values("tissue_order")
        .drop_duplicates("target_tissue")
    )
    tissues = tissue_rows["target_tissue"].tolist()
    labels = tissue_rows["display_tissue"].tolist()
    x = np.arange(len(tissues))
    offsets = np.array([-0.30, -0.10, 0.10, 0.30])

    for method_index, method in enumerate(METHODS):
        distributions = [
            source.loc[
                source["target_tissue"].eq(tissue) & source["method"].eq(method), metric
            ].to_numpy()
            for tissue in tissues
        ]
        violin = ax.violinplot(
            distributions,
            positions=x + offsets[method_index],
            vert=True,
            widths=0.19,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_linewidth(0.4)
            body.set_alpha(0.88)
        medians = [np.median(values) for values in distributions]
        ax.scatter(
            x + offsets[method_index], medians, s=12, color="white",
            edgecolor=COLORS[method], linewidth=0.8, zorder=3,
        )

    values = source[metric].to_numpy()
    padding = max((values.max() - values.min()) * 0.06, 0.02)
    lower, upper = values.min() - padding, values.max() + padding
    if metric == "residual_sample_pearson":
        lower, upper = max(-1, lower), min(1, upper)
        if lower < 0 < upper:
            ax.axhline(0, color="#777777", linewidth=0.65, linestyle="--")
    ax.set_ylim(lower, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=62, ha="right", rotation_mode="anchor", fontsize=6.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", color="#E9E7EE", linewidth=0.6)
    ax.set_axisbelow(True)


def save_all(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")


def draw_horizontal_violin(
    ax: plt.Axes,
    source: pd.DataFrame,
    tissue_order: list[int],
    metric: str,
    xlabel: str,
    show_tissue_labels: bool,
) -> None:
    meta = (
        source.loc[source["method"].eq("DORIAN"), ["target_tissue", "display_tissue"]]
        .drop_duplicates("target_tissue")
        .set_index("target_tissue")
    )
    labels = [meta.loc[tissue, "display_tissue"] for tissue in tissue_order]
    y = np.arange(len(tissue_order))
    offsets = np.array([-0.30, -0.10, 0.10, 0.30])

    for method_index, method in enumerate(METHODS):
        distributions = [
            source.loc[
                source["target_tissue"].eq(tissue) & source["method"].eq(method), metric
            ].to_numpy()
            for tissue in tissue_order
        ]
        violin = ax.violinplot(
            distributions,
            positions=y + offsets[method_index],
            vert=False,
            widths=0.19,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_linewidth(0.4)
            body.set_alpha(0.88)
        medians = [np.median(values) for values in distributions]
        ax.scatter(
            medians, y + offsets[method_index], s=11, color="white",
            edgecolor=COLORS[method], linewidth=0.75, zorder=3,
        )

    values = source[metric].to_numpy()
    padding = max((values.max() - values.min()) * 0.05, 0.02)
    lower, upper = values.min() - padding, values.max() + padding
    if metric == "residual_sample_pearson":
        lower, upper = max(-1, lower), min(1, upper)
        if lower < 0 < upper:
            ax.axvline(0, color="#777777", linewidth=0.65, linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(len(tissue_order) - 0.45, -0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_tissue_labels else [], fontsize=7.1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E9E7EE", linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = build_source()
    source.to_csv(OUT / "SourceData_accessible_DORIAN_HYFA_PMM_TEEBoT_violins.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(18.0, 10.0))
    draw_violin(axes[0], source, "residual_sample_pearson", "Residual Pearson correlation")
    draw_violin(axes[1], source, "mae", "MAE")
    axes[0].text(-0.035, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=12)
    axes[1].text(-0.035, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=12)
    fig.legend(
        handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
        loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=4,
        frameon=False, fontsize=9, handlelength=1.5, columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.94, bottom=0.20, hspace=0.54)
    save_all(fig, OUT / "accessible_DORIAN_HYFA_PMM_TEEBoT_violin_panels")
    plt.close(fig)

    # Requested paper layout: tissues run vertically, two metrics are side by side.
    dorian_rank = (
        source.loc[source["method"].eq("DORIAN")]
        .groupby("target_tissue", as_index=False)["residual_sample_pearson"]
        .mean()
        .sort_values("residual_sample_pearson", ascending=False)
    )
    ranked_tissues = dorian_rank["target_tissue"].astype(int).tolist()
    rank_map = {tissue: rank + 1 for rank, tissue in enumerate(ranked_tissues)}
    source["dorian_residual_rank"] = source["target_tissue"].map(rank_map)
    source.to_csv(OUT / "SourceData_accessible_DORIAN_HYFA_PMM_TEEBoT_violins.csv", index=False)

    # Standalone paper panels: tissues on the left, one metric per file.
    for metric, xlabel in [
        ("residual_sample_pearson", "Residual Pearson correlation"),
        ("mae", "MAE"),
    ]:
        standalone, ax = plt.subplots(figsize=(7.5, 11.0))
        draw_horizontal_violin(
            ax, source, ranked_tissues, metric, xlabel, True,
        )
        ax.legend(
            handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
            loc="lower center", bbox_to_anchor=(0.5, 1.002), ncol=4,
            frameon=False, fontsize=8.0, handlelength=1.3, columnspacing=1.3,
        )
        standalone.subplots_adjust(left=0.37, right=0.985, top=0.955, bottom=0.065)
        save_all(
            standalone,
            OUT / f"accessible_DORIAN_HYFA_PMM_TEEBoT_horizontal_violin_{metric}",
        )
        plt.close(standalone)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 11.0), gridspec_kw={"wspace": 0.10})
    draw_horizontal_violin(
        axes[0], source, ranked_tissues, "residual_sample_pearson",
        "Residual Pearson correlation", True,
    )
    draw_horizontal_violin(
        axes[1], source, ranked_tissues, "mae", "MAE", False,
    )
    axes[0].text(-0.28, 1.012, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=11)
    axes[1].text(-0.035, 1.012, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.legend(
        handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
        loc="upper center", bbox_to_anchor=(0.57, 0.995), ncol=4,
        frameon=False, fontsize=8.5, handlelength=1.4, columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.29, right=0.99, top=0.955, bottom=0.065)
    save_all(fig, OUT / "accessible_DORIAN_HYFA_PMM_TEEBoT_violin_horizontal")
    plt.close(fig)


if __name__ == "__main__":
    main()
