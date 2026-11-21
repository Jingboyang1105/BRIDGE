from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
SRC = OUT / "source_data"
FIG.mkdir(parents=True, exist_ok=True)
SRC.mkdir(parents=True, exist_ok=True)

COLORS = {"DORIAN": "#6C5CE7", "HYFA": "#F2B84B", "PMM": "#38B2A3"}
METHOD_ORDER = ["DORIAN", "HYFA", "PMM"]
HALLMARK_GROUP_COLORS = {
    "Immune / inflammation": "#4C78A8",
    "Metabolic / stress": "#72A36A",
    "Growth / cell cycle": "#B279A2",
    "Hormone / secretory": "#8B73E6",
    "ECM / stromal / muscle": "#E3A33E",
    "Other": "#9A9A9A",
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


def clean_hallmark_label(value: str) -> str:
    replacements = {
        "Tnfa": "TNFA", "Nfkb": "NF-κB", "Il6": "IL6", "Jak": "JAK",
        "Stat3": "STAT3", "Myc": "MYC", "Mtorc1": "mTORC1", "Kras": "KRAS",
    }
    text = str(value)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def save_all(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def add_median_points(ax: plt.Axes, data: pd.DataFrame, tissue_order: list[str], metric: str) -> None:
    offsets = dict(zip(METHOD_ORDER, [-0.2667, 0.0, 0.2667]))
    med = data.groupby(["target_tissue_name", "method"], observed=True)[metric].median()
    for yi, tissue in enumerate(tissue_order):
        for method in METHOD_ORDER:
            key = (tissue, method)
            if key in med.index and np.isfinite(med.loc[key]):
                ax.scatter(med.loc[key], yi + offsets[method], s=5.5, color="white",
                           edgecolor="#333333", linewidth=0.25, zorder=5)


def horizontal_violin(data: pd.DataFrame, metric: str, xlabel: str, stem: Path,
                      sort_ascending: bool, xlim: tuple[float, float] | None = None) -> None:
    dorian = data.loc[data["method"] == "DORIAN"]
    tissue_order = (dorian.groupby("target_tissue_name")[metric].mean()
                    .sort_values(ascending=sort_ascending).index.tolist())
    plot_data = data.copy()
    plot_data["target_tissue_name"] = pd.Categorical(
        plot_data["target_tissue_name"], categories=tissue_order, ordered=True)
    height = max(9.0, 0.31 * len(tissue_order) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, height))
    sns.violinplot(
        data=plot_data, y="target_tissue_name", x=metric, hue="method",
        order=tissue_order, hue_order=METHOD_ORDER, palette=COLORS,
        orient="h", dodge=True, cut=0, inner=None, linewidth=0.35,
        density_norm="width", bw_adjust=0.75, saturation=1, ax=ax,
    )
    add_median_points(ax, plot_data, tissue_order, metric)
    if metric == "residual_sample_pearson":
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.65, zorder=0)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=5.2, pad=1.5)
    ax.tick_params(axis="x", labelsize=6.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="lower center", bbox_to_anchor=(0.5, 1.002),
              ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=1.3)
    fig.tight_layout()
    save_all(fig, stem)
    ordered = plot_data.sort_values(["target_tissue_name", "method", "record_id"])
    ordered.to_csv(SRC / f"{stem.name}_source.csv", index=False)


def build_hallmark() -> None:
    clean_master = SRC / "hallmark_dorian_vs_hyfa_all50_source.csv"
    if clean_master.exists():
        master = pd.read_csv(clean_master)
    else:
        master = pd.read_csv(
            ROOT / "_Archive_Do_Not_Use" / "Legacy_Mechanism_and_Hallmark_Outputs" /
            "results" / "gene_hallmark_biology_all" / "hallmark_biology_master_table_20k.csv"
        )
    master["label"] = master["hallmark_label"].map(clean_hallmark_label)
    master.to_csv(clean_master, index=False)

    # One paper-facing table covering both Hallmark claims.
    table = master.copy()
    table["recovery_rank"] = table["mean_residual_gene_pearson"].rank(
        method="first", ascending=False
    ).astype(int)
    table["dorian_vs_hyfa_rank"] = table["mean_cml_vs_hyfa_mae_improvement"].rank(
        method="first", ascending=False
    ).astype(int)
    table = table.sort_values("recovery_rank")[[
        "recovery_rank", "dorian_vs_hyfa_rank", "hallmark", "label", "hallmark_group",
        "n_genes_in_20k", "mean_residual_gene_pearson", "mean_gene_pearson", "mean_mae",
        "mean_cml_vs_hyfa_mae_improvement", "delta_improvement_vs_background",
        "top_improved_overlap", "top_improved_odds_ratio", "top_improved_fisher_p",
        "top_improved_fisher_fdr", "member_genes_in_20k",
    ]]
    table_dir = ROOT / "Results_for_Paper" / "Supplementary_Tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "Table_S_Hallmark_Recovery_and_DORIAN_vs_HYFA.csv", index=False)

    legend_handles = [
        Patch(facecolor=color, label=group)
        for group, color in HALLMARK_GROUP_COLORS.items()
    ]

    # A: recovery across all 50 Hallmark sets. Categories and colors are shared
    # exactly with the DORIAN-vs-HYFA enrichment panel below.
    recovery = master.sort_values("mean_residual_gene_pearson", ascending=True).reset_index(drop=True)
    recovery["rank"] = np.arange(1, len(recovery) + 1)
    recovery_colors = recovery["hallmark_group"].map(HALLMARK_GROUP_COLORS).fillna("#9A9A9A")
    fig, ax = plt.subplots(figsize=(7.4, 11.0))
    ax.barh(
        recovery["label"], recovery["mean_residual_gene_pearson"],
        color=recovery_colors, height=0.72,
    )
    recovery_mean = recovery["mean_residual_gene_pearson"].mean()
    ax.axvline(recovery_mean, color="#555555", linestyle="--", linewidth=0.75)
    ax.text(
        recovery_mean + 0.002, 0.99, f"mean = {recovery_mean:.3f}",
        transform=ax.get_xaxis_transform(), ha="left", va="top", color="#555555",
    )
    ax.set_xlabel("Mean residual gene Pearson")
    ax.set_ylabel("")
    ax.set_xlim(0, max(0.46, recovery["mean_residual_gene_pearson"].max() + 0.015))
    ax.tick_params(axis="y", labelsize=5.8)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.legend(handles=legend_handles, loc="lower right", ncol=2, fontsize=6.0,
              handlelength=1.2, columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()
    save_all(fig, FIG / "hallmark_residual_recovery_all50_unified_colors")
    recovery.to_csv(SRC / "hallmark_residual_recovery_all50_unified_colors_source.csv", index=False)

    # A: pathways with the largest average per-gene MAE reduction by DORIAN.
    top = master.nlargest(15, "mean_cml_vs_hyfa_mae_improvement").sort_values(
        "mean_cml_vs_hyfa_mae_improvement")
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    y = np.arange(len(top))
    group_colors = top["hallmark_group"].map(HALLMARK_GROUP_COLORS).fillna("#9A9A9A")
    ax.hlines(
        y, 0, top["mean_cml_vs_hyfa_mae_improvement"],
        color=group_colors, alpha=0.30, linewidth=2.1,
    )
    ax.scatter(
        top["mean_cml_vs_hyfa_mae_improvement"], y,
        color=group_colors, s=35, zorder=3,
    )
    ax.set_yticks(y, top["label"])
    ax.set_xlabel("Mean MAE reduction: HYFA − DORIAN")
    ax.set_ylabel("")
    ax.axvline(0, color="#777777", linewidth=0.7)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    shown_groups = [group for group in HALLMARK_GROUP_COLORS if group in set(top["hallmark_group"])]
    ax.legend(
        handles=[Patch(facecolor=HALLMARK_GROUP_COLORS[group], label=group) for group in shown_groups],
        loc="lower right", frameon=False, fontsize=6.3,
    )
    fig.tight_layout()
    save_all(fig, FIG / "hallmark_top15_mean_mae_improvement")
    top.to_csv(SRC / "hallmark_top15_mean_mae_improvement_source.csv", index=False)

    # B: pathways enriched among the genes most improved by DORIAN.
    sig = master.loc[master["top_improved_fisher_fdr"] < 0.05].copy()
    sig = sig.sort_values("top_improved_odds_ratio")
    fig, ax = plt.subplots(figsize=(7.0, max(4.7, 0.26 * len(sig) + 1.2)))
    sig_colors = sig["hallmark_group"].map(HALLMARK_GROUP_COLORS).fillna("#9A9A9A")
    ax.barh(sig["label"], sig["top_improved_odds_ratio"], color=sig_colors, height=0.7)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Odds ratio among genes most improved by DORIAN over HYFA")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.5)
    ax.text(0.99, 1.01, "FDR < 0.05", transform=ax.transAxes, ha="right", va="bottom", color="#555555")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.0,
              handlelength=1.2, labelspacing=0.35)
    fig.tight_layout()
    save_all(fig, FIG / "hallmark_significant_odds_ratios")
    sig.to_csv(SRC / "hallmark_significant_odds_ratios_source.csv", index=False)

    # C: compact alternative combining effect size, enrichment and overlap.
    bubble = sig.copy()
    bubble["minus_log10_fdr"] = -np.log10(bubble["top_improved_fisher_fdr"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(7.0, 5.3))
    sizes = 18 + 2.2 * bubble["top_improved_overlap"]
    sc = ax.scatter(
        bubble["mean_cml_vs_hyfa_mae_improvement"], bubble["top_improved_odds_ratio"],
        s=sizes, c=bubble["minus_log10_fdr"], cmap="Purples", edgecolor="#4A3EA0", linewidth=0.45,
        alpha=0.9)
    label_rows = bubble.nlargest(min(6, len(bubble)), "top_improved_odds_ratio")
    for _, row in label_rows.iterrows():
        right_side = row["mean_cml_vs_hyfa_mae_improvement"] > 0.087
        ax.annotate(row["label"], (row["mean_cml_vs_hyfa_mae_improvement"], row["top_improved_odds_ratio"]),
                    xytext=(-4 if right_side else 4, 4), textcoords="offset points", fontsize=5.2,
                    ha="right" if right_side else "left")
    ax.axhline(1, color="#777777", linestyle="--", linewidth=0.65)
    ax.set_xlabel("Mean MAE reduction: HYFA − DORIAN")
    ax.set_ylabel("Odds ratio among most-improved genes")
    ax.grid(color="#ECECEC", linewidth=0.45)
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("−log10(FDR)")
    fig.tight_layout()
    save_all(fig, FIG / "hallmark_effect_enrichment_bubble")
    bubble.to_csv(SRC / "hallmark_effect_enrichment_bubble_source.csv", index=False)


def build_violins() -> None:
    hide = pd.read_csv(
        ROOT / "02_Benchmark_vs_HYFA_PMM" / "figures" / "all_tissues" /
        "hide_one_full_65_tissue_violin_source.csv")
    hide["method"] = hide["method"].replace({"CML": "DORIAN"})
    hide = hide.loc[hide["method"].isin(METHOD_ORDER)].copy()
    horizontal_violin(hide, "residual_sample_pearson", "Residual Pearson correlation",
                      FIG / "hide_one_all65_residual_pearson_horizontal_violin",
                      sort_ascending=False, xlim=(-0.72, 1.02))
    horizontal_violin(hide, "mae", "MAE",
                      FIG / "hide_one_all65_mae_horizontal_violin",
                      sort_ascending=True, xlim=(0, max(1.6, hide["mae"].quantile(0.999))))

    combined = pd.read_csv(ROOT / "Results_for_Paper" / "Source_Data" / "Figure_2_violin_source.csv")
    acc = combined.loc[combined["protocol"] == "accessible"].copy()
    acc = acc.loc[acc["method"].isin(METHOD_ORDER)]
    horizontal_violin(acc, "residual_sample_pearson", "Residual Pearson correlation",
                      FIG / "accessible_all28_residual_pearson_horizontal_violin",
                      sort_ascending=False, xlim=(-0.72, 1.02))
    horizontal_violin(acc, "mae", "MAE",
                      FIG / "accessible_all28_mae_horizontal_violin",
                      sort_ascending=True, xlim=(0, max(1.6, acc["mae"].quantile(0.999))))


def build_retrieval_analysis() -> None:
    neighbors = pd.read_csv(
        ROOT / "03_How_DORIAN_Works" / "data" / "source" / "retrieval_neighbors_long.csv")
    support = pd.read_csv(ROOT / "07_GTEx_Completed_Resource" / "audit" / "tissue_support.csv")
    support = support[["target_tissue_name", "n_hide_one_training_records"]].rename(
        columns={"target_tissue_name": "retrieved_tissue_name", "n_hide_one_training_records": "support_proxy"})
    for col in ["is_same_tissue", "is_same_donor", "is_known_tissue"]:
        neighbors[col] = neighbors[col].astype(str).str.lower().eq("true")
    off = neighbors.loc[~neighbors["is_same_tissue"]].copy()
    off["same_system"] = off["target_tissue_system"] == off["retrieved_tissue_system"]

    pair = (off.groupby(["target_tissue_name", "target_tissue_system", "retrieved_tissue_name",
                         "retrieved_tissue_system"], as_index=False)
            .size().rename(columns={"size": "n_retrievals"}))
    pair["same_system"] = pair["target_tissue_system"] == pair["retrieved_tissue_system"]
    total_by_target = off.groupby("target_tissue_name").size().rename("n_offdiag_total")
    pair = pair.merge(total_by_target, on="target_tissue_name")
    pair["offdiag_fraction_within_target"] = pair["n_retrievals"] / pair["n_offdiag_total"]
    pair = pair.merge(support, on="retrieved_tissue_name", how="left")
    pair.to_csv(SRC / "retrieval_all_off_diagonal_pairs.csv", index=False)

    target_total = neighbors.groupby("target_tissue_name").size().rename("n_all")
    diag = neighbors.groupby("target_tissue_name")["is_same_tissue"].mean().rename("diagonal_fraction")
    same_system = off.groupby("target_tissue_name")["same_system"].mean().rename("same_system_fraction_offdiag")
    top_pair = (pair.sort_values(["target_tissue_name", "n_retrievals"], ascending=[True, False])
                .drop_duplicates("target_tissue_name")
                [["target_tissue_name", "retrieved_tissue_name", "retrieved_tissue_system",
                  "n_retrievals", "offdiag_fraction_within_target", "same_system", "support_proxy"]]
                .rename(columns={"retrieved_tissue_name": "top_alternative_tissue",
                                 "retrieved_tissue_system": "top_alternative_system",
                                 "n_retrievals": "top_alternative_n",
                                 "offdiag_fraction_within_target": "top_alternative_fraction_of_offdiag",
                                 "same_system": "top_alternative_same_system",
                                 "support_proxy": "top_alternative_support_proxy"}))
    target_support = support.rename(columns={"retrieved_tissue_name": "target_tissue_name",
                                             "support_proxy": "target_support_proxy"})
    summary = pd.concat([target_total, diag, same_system], axis=1).reset_index().merge(
        top_pair, on="target_tissue_name", how="left")
    summary = summary.merge(target_support, on="target_tissue_name", how="left")
    summary.to_csv(SRC / "retrieval_target_off_diagonal_summary.csv", index=False)

    retrieved_counts = off.groupby("retrieved_tissue_name").size().rename("offdiag_retrieval_count").reset_index()
    retrieved_counts = retrieved_counts.merge(support, on="retrieved_tissue_name", how="left")
    valid = retrieved_counts.dropna(subset=["support_proxy"])
    rho, pval = spearmanr(valid["support_proxy"], valid["offdiag_retrieval_count"])
    diag_valid = summary.dropna(subset=["target_support_proxy", "diagonal_fraction"])
    diag_rho, diag_pval = spearmanr(diag_valid["target_support_proxy"], diag_valid["diagonal_fraction"])
    retrieved_counts.to_csv(SRC / "retrieval_offdiag_frequency_vs_support.csv", index=False)

    observed = float(off["same_system"].mean())
    rng = np.random.default_rng(42)
    target_systems = off["target_tissue_system"].to_numpy()
    retrieved_systems = off["retrieved_tissue_system"].to_numpy()
    null = np.empty(2000)
    for i in range(len(null)):
        null[i] = np.mean(target_systems == rng.permutation(retrieved_systems))
    perm_p = (1 + np.sum(null >= observed)) / (len(null) + 1)
    diagnostics = pd.DataFrame({
        "metric": ["observed_offdiag_same_system_fraction", "permutation_null_mean",
                   "permutation_p_one_sided", "spearman_support_vs_offdiag_frequency_rho",
                   "spearman_support_vs_offdiag_frequency_p", "spearman_target_support_vs_diagonal_fraction_rho",
                   "spearman_target_support_vs_diagonal_fraction_p",
                   "targets_with_top_alternative_same_system_fraction"],
        "value": [observed, null.mean(), perm_p, rho, pval,
                  diag_rho, diag_pval,
                  summary["top_alternative_same_system"].dropna().astype(bool).mean()],
    })
    diagnostics.to_csv(SRC / "retrieval_diagnostic_statistics.csv", index=False)

    top20 = pair.nlargest(20, "n_retrievals").sort_values("n_retrievals")
    labels = top20["target_tissue_name"].str.replace("Brain - ", "", regex=False) + " → " + \
             top20["retrieved_tissue_name"].str.replace("Brain - ", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].barh(np.arange(len(top20)), top20["n_retrievals"],
                 color=np.where(top20["same_system"], "#6C5CE7", "#B8B8B8"))
    axes[0].set_yticks(np.arange(len(top20)), labels, fontsize=5.2)
    axes[0].set_xlabel("Off-diagonal top-k retrieval count")
    axes[0].set_title("Largest cross-tissue borrowing pairs", loc="left", fontweight="bold")
    axes[0].text(0.99, 0.01, "Purple: same tissue system", transform=axes[0].transAxes,
                 ha="right", va="bottom", color="#5C4BCB", fontsize=6)
    axes[0].grid(axis="x", color="#E8E8E8", linewidth=0.5)

    axes[1].scatter(valid["support_proxy"], valid["offdiag_retrieval_count"],
                    color="#197B78", s=25, alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training-record support proxy (log scale)")
    axes[1].set_ylabel("Off-diagonal retrieval count")
    axes[1].set_title("Availability contribution", loc="left", fontweight="bold")
    axes[1].text(0.03, 0.97, f"Spearman ρ = {rho:.2f}\nP = {pval:.2g}",
                 transform=axes[1].transAxes, va="top")
    axes[1].grid(color="#E8E8E8", linewidth=0.5)
    fig.tight_layout()
    save_all(fig, FIG / "retrieval_off_diagonal_diagnostic")


def validate_inputs_and_outputs() -> None:
    hide = pd.read_csv(SRC / "hide_one_all65_residual_pearson_horizontal_violin_source.csv")
    acc = pd.read_csv(SRC / "accessible_all28_residual_pearson_horizontal_violin_source.csv")
    assert hide["target_tissue_name"].nunique() == 65
    assert acc["target_tissue_name"].nunique() == 28
    assert set(hide["method"].unique()) == set(METHOD_ORDER)
    assert set(acc["method"].unique()) == set(METHOD_ORDER)
    for stem in [
        "hallmark_top15_mean_mae_improvement", "hallmark_significant_odds_ratios",
        "hallmark_effect_enrichment_bubble", "hide_one_all65_residual_pearson_horizontal_violin",
        "hide_one_all65_mae_horizontal_violin", "accessible_all28_residual_pearson_horizontal_violin",
        "accessible_all28_mae_horizontal_violin", "retrieval_off_diagonal_diagnostic",
    ]:
        for ext in [".png", ".svg", ".pdf", ".tiff"]:
            path = FIG / f"{stem}{ext}"
            assert path.exists() and path.stat().st_size > 0, path


if __name__ == "__main__":
    build_hallmark()
    build_violins()
    build_retrieval_analysis()
    validate_inputs_and_outputs()
    print(OUT)
