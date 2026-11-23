"""Publication figures for formal GTEx v11 tissue comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

METHODS = ["BRIDGE", "HYFA", "PMM"]
COLORS = {"BRIDGE": "#6C5CE7", "HYFA": "#F2B84B", "PMM": "#36B8A5"}
METRICS = {
    "mae": {"label": "MAE", "better": "lower"},
    "residual_sample_pearson": {"label": "Residual Pearson correlation", "better": "higher"},
}


def _read(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required formal result is missing: {path}")
    return pd.read_csv(path)


def _short_name(name: str) -> str:
    replacements = {
        "Brain - ": "",
        "Cells - ": "Cells: ",
        "Esophagus - ": "Esophagus: ",
        "Artery - ": "Artery: ",
        "Heart - ": "Heart: ",
        "Colon - ": "Colon: ",
    }
    result = str(name)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _load_records(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    hide_specs = [
        ("BRIDGE", "02_Benchmark_vs_HYFA_PMM/results/hide_one/hide_one_test_200e_cml_record_scores_for_boxplots.csv"),
        ("HYFA", "02_Benchmark_vs_HYFA_PMM/results/hide_one/hide_one_test_200e_hyfa_record_scores_for_boxplots.csv"),
        ("PMM", "02_Benchmark_vs_HYFA_PMM/results/pmm_v11_seed42/formal_olga/hide_one/pmm_record_scores.csv"),
    ]
    for method, path in hide_specs:
        df = _read(root, path)
        if method == "BRIDGE":
            df = df[
                (df["method"] == "CML gate")
                & (df["prediction_key"] == "pred_two_expert_cml_gate")
            ].copy()
        df = df[df["target_tissue_name"].astype(str).str.startswith("Brain - ")].copy()
        df["method"] = method
        df["protocol"] = "hide_one"
        frames.append(df)

    accessible = _read(
        root,
        "02_Benchmark_vs_HYFA_PMM/results/accessible/accessible_20000genes_seed42_record_scores_for_boxplots.csv",
    )
    accessible["method"] = accessible["method"].replace(
        {"CML_accessible_hide_one_gate": "BRIDGE", "HYFA_accessible": "HYFA"}
    )
    accessible["protocol"] = "accessible"
    frames.append(accessible)

    pmm_accessible = _read(
        root, "02_Benchmark_vs_HYFA_PMM/results/pmm_v11_seed42/formal_olga/accessible/pmm_record_scores.csv"
    )
    pmm_accessible["method"] = "PMM"
    pmm_accessible["protocol"] = "accessible"
    frames.append(pmm_accessible)

    records = pd.concat(frames, ignore_index=True)
    records = records[
        [
            "protocol",
            "method",
            "record_id",
            "target_tissue",
            "target_tissue_name",
            "mae",
            "residual_sample_pearson",
        ]
    ].copy()
    records["target_tissue"] = records["target_tissue"].astype(int)
    records["mae"] = pd.to_numeric(records["mae"], errors="raise")
    records["residual_sample_pearson"] = pd.to_numeric(
        records["residual_sample_pearson"], errors="raise"
    )
    records["display_tissue"] = records["target_tissue_name"].map(_short_name)
    records["method"] = pd.Categorical(records["method"], METHODS, ordered=True)
    return records


def build_sources(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _load_records(root)
    bars = (
        records.groupby(
            ["protocol", "method", "target_tissue", "target_tissue_name", "display_tissue"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_records=("record_id", "size"),
            mae=("mae", "mean"),
            residual_sample_pearson=("residual_sample_pearson", "mean"),
        )
    )

    expected = {"hide_one": 13, "accessible": 28}
    ordered_bars: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []
    for protocol, expected_n in expected.items():
        block = bars[bars["protocol"] == protocol].copy()
        record_block = records[records["protocol"] == protocol]
        if block["target_tissue"].nunique() != expected_n:
            raise ValueError(f"Unexpected {protocol} tissue coverage")
        if record_block["target_tissue"].nunique() != expected_n:
            raise ValueError(f"Unexpected {protocol} record coverage")
        for frame in (block, record_block):
            if set(frame["method"].dropna().astype(str)) != set(METHODS):
                raise ValueError(f"Incomplete method coverage for {protocol}")
        order = (
            block[block["method"] == "BRIDGE"]
            .sort_values("residual_sample_pearson", ascending=False)["target_tissue"]
            .tolist()
        )
        order_frame = pd.DataFrame(
            {"protocol": protocol, "target_tissue": order, "tissue_order": range(len(order))}
        )
        order_frames.append(order_frame)
        ordered_bars.append(block.merge(order_frame, on=["protocol", "target_tissue"], validate="many_to_one"))

    bars = pd.concat(ordered_bars, ignore_index=True)
    order_table = pd.concat(order_frames, ignore_index=True)
    records = records.merge(order_table, on=["protocol", "target_tissue"], validate="many_to_one")
    if not np.isfinite(bars[["mae", "residual_sample_pearson"]].to_numpy()).all():
        raise ValueError("Non-finite tissue-level metrics")
    if not np.isfinite(records[["mae", "residual_sample_pearson"]].to_numpy()).all():
        raise ValueError("Non-finite record-level metrics")
    bars = bars.sort_values(["protocol", "tissue_order", "method"]).reset_index(drop=True)
    records = records.sort_values(["protocol", "tissue_order", "method", "record_id"]).reset_index(drop=True)
    return bars, records


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_vertical_bars(
    bars: pd.DataFrame, protocol: str, metric: str, output_base: Path
) -> None:
    data = bars[bars["protocol"] == protocol].copy()
    tissue_rows = data.sort_values("tissue_order").drop_duplicates("target_tissue")
    tissues = tissue_rows["target_tissue"].tolist()
    labels = tissue_rows["display_tissue"].tolist()
    n = len(tissues)
    width = 9.0 if n == 13 else 14.5
    fig, ax = plt.subplots(figsize=(width, 4.8))
    x = np.arange(n)
    bar_width = 0.25
    for index, method in enumerate(METHODS):
        values = data[data["method"] == method].set_index("target_tissue").loc[tissues, metric]
        ax.bar(
            x + (index - 1) * bar_width,
            values,
            width=bar_width * 0.92,
            color=COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=6.4 if n == 13 else 5.7)
    ax.set_ylabel(METRICS[metric]["label"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#E9E7EE", linewidth=0.6)
    ax.set_axisbelow(True)
    if metric == "residual_sample_pearson":
        ax.axhline(0, color="#777777", linewidth=0.65, linestyle="--")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.34 if n == 13 else 0.40)
    _save(fig, output_base)


def plot_grouped_violins(
    records: pd.DataFrame, protocol: str, metric: str, output_base: Path
) -> None:
    data = records[records["protocol"] == protocol].copy()
    tissue_rows = data.sort_values("tissue_order").drop_duplicates("target_tissue")
    tissues = tissue_rows["target_tissue"].tolist()
    labels = tissue_rows["display_tissue"].tolist()
    n = len(tissues)
    fig_width = 11.0 if n == 13 else 18.0
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    x = np.arange(n)
    metric_min = data[metric].min()
    metric_max = data[metric].max()
    padding = max((metric_max - metric_min) * 0.06, 0.02)
    ylim = (metric_min - padding, metric_max + padding)
    if metric == "residual_sample_pearson":
        ylim = (max(-1, ylim[0]), min(1, ylim[1]))

    offsets = np.array([-0.27, 0.0, 0.27])
    for method_index, method in enumerate(METHODS):
        distributions = [
            data[(data["target_tissue"] == tissue) & (data["method"] == method)][metric].to_numpy()
            for tissue in tissues
        ]
        violin = ax.violinplot(
            distributions,
            positions=x + offsets[method_index],
            vert=True,
            widths=0.25,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_linewidth(0.4)
            body.set_alpha(0.86)
        medians = [np.median(values) for values in distributions]
        ax.scatter(
            x + offsets[method_index], medians, s=11, color="white",
            edgecolor=COLORS[method], linewidth=0.8, zorder=3,
        )
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=62, ha="right", fontsize=6.3)
    ax.set_ylabel(METRICS[metric]["label"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7, labelleft=True)
    ax.grid(axis="y", color="#E9E7EE", linewidth=0.6)
    ax.set_axisbelow(True)
    if metric == "residual_sample_pearson" and ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color="#777777", linewidth=0.65, linestyle="--")
    ax.legend(
        handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=7.5,
        handlelength=1.5, columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.91, bottom=0.35)
    _save(fig, output_base)


def plot_horizontal_grouped_violins(
    records: pd.DataFrame, protocol: str, metric: str, output_base: Path
) -> None:
    """Paper layout with tissues on the y-axis and the metric on the x-axis."""
    data = records[records["protocol"] == protocol].copy()
    tissue_rows = data.sort_values("tissue_order").drop_duplicates("target_tissue")
    tissues = tissue_rows["target_tissue"].tolist()
    labels = tissue_rows["display_tissue"].tolist()
    n = len(tissues)
    fig_height = 6.2 if n == 13 else 11.0
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    y = np.arange(n)

    values = data[metric].to_numpy()
    padding = max((values.max() - values.min()) * 0.06, 0.02)
    xlim = (values.min() - padding, values.max() + padding)
    if metric == "residual_sample_pearson":
        xlim = (max(-1, xlim[0]), min(1, xlim[1]))

    offsets = np.array([-0.27, 0.0, 0.27])
    for method_index, method in enumerate(METHODS):
        distributions = [
            data[(data["target_tissue"] == tissue) & (data["method"] == method)][metric].to_numpy()
            for tissue in tissues
        ]
        violin = ax.violinplot(
            distributions,
            positions=y + offsets[method_index],
            vert=False,
            widths=0.25,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[method])
            body.set_edgecolor("white")
            body.set_linewidth(0.4)
            body.set_alpha(0.88)
        medians = [np.median(values_) for values_ in distributions]
        ax.scatter(
            medians, y + offsets[method_index], s=11, color="white",
            edgecolor=COLORS[method], linewidth=0.8, zorder=3,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(n - 0.55, -0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.set_xlabel(METRICS[metric]["label"], fontsize=8.5)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E9E7EE", linewidth=0.6)
    ax.set_axisbelow(True)
    if metric == "residual_sample_pearson" and xlim[0] < 0 < xlim[1]:
        ax.axvline(0, color="#777777", linewidth=0.65, linestyle="--")
    ax.legend(
        handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.002), fontsize=7.5,
        handlelength=1.4, columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.31 if n == 13 else 0.34, right=0.98, top=0.94, bottom=0.08)
    _save(fig, output_base)


def plot_main_figure_two_by_two(records: pd.DataFrame, output_base: Path) -> None:
    """Main-text four-panel figure with typography sized at final page width."""
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 9.2))
    panel_specs = [
        ("hide_one", "residual_sample_pearson", "a"),
        ("accessible", "residual_sample_pearson", "b"),
        ("hide_one", "mae", "c"),
        ("accessible", "mae", "d"),
    ]

    for ax, (protocol, metric, panel_label) in zip(axes.flat, panel_specs):
        data = records[records["protocol"] == protocol].copy()
        tissue_rows = data.sort_values("tissue_order").drop_duplicates("target_tissue")
        tissues = tissue_rows["target_tissue"].tolist()
        labels = tissue_rows["display_tissue"].tolist()
        n_tissues = len(tissues)
        x = np.arange(n_tissues)

        metric_min = data[metric].min()
        metric_max = data[metric].max()
        padding = max((metric_max - metric_min) * 0.06, 0.02)
        ylim = (metric_min - padding, metric_max + padding)
        if metric == "residual_sample_pearson":
            ylim = (max(-1, ylim[0]), min(1, ylim[1]))

        offsets = np.array([-0.27, 0.0, 0.27])
        for method_index, method in enumerate(METHODS):
            distributions = [
                data[(data["target_tissue"] == tissue) & (data["method"] == method)][metric].to_numpy()
                for tissue in tissues
            ]
            violin = ax.violinplot(
                distributions,
                positions=x + offsets[method_index],
                vert=True,
                widths=0.25,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[method])
                body.set_edgecolor("white")
                body.set_linewidth(0.45)
                body.set_alpha(0.88)
            medians = [np.median(values) for values in distributions]
            ax.scatter(
                x + offsets[method_index], medians, s=15, color="white",
                edgecolor=COLORS[method], linewidth=0.8, zorder=3,
            )

        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=58 if protocol == "hide_one" else 62,
            ha="right",
            rotation_mode="anchor",
            fontsize=7.3 if protocol == "hide_one" else 6.5,
        )
        ax.set_ylabel(METRICS[metric]["label"], fontsize=9.5, labelpad=6)
        ax.tick_params(axis="y", labelsize=8.2, width=0.7, length=3)
        ax.tick_params(axis="x", width=0.7, length=3, pad=2)
        ax.grid(axis="y", color="#E9E7EE", linewidth=0.65)
        ax.set_axisbelow(True)
        if metric == "residual_sample_pearson" and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
        ax.legend(
            handles=[Patch(facecolor=COLORS[m], edgecolor="none", label=m) for m in METHODS],
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=8.5,
            handlelength=1.4,
            columnspacing=1.6,
        )
        ax.text(
            -0.105,
            1.105,
            panel_label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.subplots_adjust(left=0.07, right=0.992, top=0.94, bottom=0.13, wspace=0.19, hspace=0.72)
    _save(fig, output_base)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output_dir = Path(__file__).resolve().parent
    bars, records = build_sources(root)
    bars.to_csv(output_dir / "tissue_bar_source_mae_residual.csv", index=False)
    records.to_csv(output_dir / "tissue_violin_source_mae_residual.csv", index=False)
    for protocol in ("hide_one", "accessible"):
        for metric in METRICS:
            plot_vertical_bars(
                bars,
                protocol,
                metric,
                output_dir / f"{protocol}_vertical_bar_{metric}",
            )
            plot_grouped_violins(
                records,
                protocol,
                metric,
                output_dir / f"{protocol}_grouped_violin_{metric}",
            )
            plot_horizontal_grouped_violins(
                records,
                protocol,
                metric,
                output_dir / f"{protocol}_horizontal_violin_{metric}",
            )
    plot_main_figure_two_by_two(
        records,
        output_dir / "combined_final" / "figure2_hide_one_accessible_four_panel_large_type",
    )


if __name__ == "__main__":
    main()
