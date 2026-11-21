"""GTEx v11 donor- and tissue-level missingness marginals."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

OBSERVED = "#6757D9"
MISSING = "#E9E6F7"

def build_sources(root: Path):
    records = pd.read_csv(root / "records/donor_query_dictionary_20000genes_seed42.csv")
    records = records[(records["split"] == "train") & (records["source"] == "hide_one")].copy()
    names = pd.read_csv(root / "results/main_pmm_v11_seed42/formal_olga/hide_one/pmm_per_tissue_metrics.csv")
    name_map = names.drop_duplicates("target_tissue").set_index("target_tissue")["target_tissue_name"].to_dict()
    matrix = pd.crosstab(records["donor"], records["target_tissue"]).reindex(columns=sorted(name_map), fill_value=0).gt(0)
    donor = pd.DataFrame({"donor": matrix.index, "observed_count": matrix.sum(axis=1).to_numpy()})
    donor["total_tissues"] = matrix.shape[1]
    donor["missing_rate"] = 1 - donor["observed_count"] / donor["total_tissues"]
    donor = donor.sort_values(["missing_rate", "donor"], ascending=[False, True]).reset_index(drop=True)
    donor["plot_order"] = np.arange(1, len(donor) + 1)
    tissue = pd.DataFrame({"target_tissue": matrix.columns, "observed_count": matrix.sum(axis=0).to_numpy()})
    tissue["tissue"] = tissue["target_tissue"].map(name_map)
    tissue["total_donors"] = matrix.shape[0]
    tissue["missing_rate"] = 1 - tissue["observed_count"] / tissue["total_donors"]
    tissue = tissue.sort_values(["missing_rate", "target_tissue"], ascending=[True, True]).reset_index(drop=True)
    tissue["plot_order"] = np.arange(1, len(tissue) + 1)
    return donor, tissue

def _style(ax):
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)

def _save(fig, base):
    for ext in ("png", "svg", "pdf", "tiff"):
        fig.savefig(base.with_suffix("." + ext), dpi=600 if ext == "tiff" else 250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_donor(donor, out):
    fig, ax = plt.subplots(figsize=(5.0, 7.0))
    y = np.arange(len(donor))
    observed = donor["observed_count"].to_numpy()
    total_tissues = int(donor["total_tissues"].iloc[0])
    mean_observed = float(observed.mean())
    ax.barh(y, observed, color=OBSERVED, height=1.0, linewidth=0)
    ax.invert_yaxis(); ax.set_yticks([])
    ax.set_xlim(0, total_tissues)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 65])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.set_xlabel("Observed tissues per donor (of 65)")
    ax.set_ylabel(f"Training donors ranked by tissue coverage (n={len(donor)})")
    ax.axvline(mean_observed, color="#555555", ls="--", lw=1.2)
    ax.text(mean_observed + 1.0, 0.985, f"Mean = {mean_observed:.1f}",
            transform=ax.get_xaxis_transform(), ha="left", va="top", color="#555555")
    _save(fig, out / "donor_missingness_descending")

def plot_tissue(tissue, out):
    fig, ax = plt.subplots(figsize=(7.2, 10.5))
    y = np.arange(len(tissue)); observed = 1-tissue["missing_rate"].to_numpy()
    ax.barh(y, observed, color=OBSERVED, height=.82, linewidth=0)
    ax.barh(y, tissue["missing_rate"], left=observed, color=MISSING, height=.82, linewidth=0)
    ax.set_yticks(y, tissue["tissue"], fontsize=5.6); ax.invert_yaxis(); _style(ax)
    ax.set_xlabel("Proportion of donors"); ax.set_ylabel(f"Tissues (n={len(tissue)})")
    ax.legend(handles=[Patch(color=OBSERVED,label="Observed"),Patch(color=MISSING,label="Missing")], ncol=2, loc="lower center", bbox_to_anchor=(.5,1.01), frameon=False)
    _save(fig, out / "tissue_missingness_ascending")

def _coverage_stats(donor):
    values = donor["observed_count"].to_numpy()
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return values.mean(), median, q1, q3

def plot_donor_coverage_histogram(donor, out):
    """Distribution view: directly answers how many tissues a typical donor has."""
    values = donor["observed_count"].to_numpy()
    mean, median, q1, q3 = _coverage_stats(donor)
    bins = np.arange(-0.5, donor["total_tissues"].iloc[0] + 1.5, 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(values, bins=bins, color=OBSERVED, edgecolor="white", linewidth=.35)
    ax.axvline(mean, color="#D8902F", lw=1.5, label=f"Mean = {mean:.1f}")
    ax.axvline(median, color="#2F3349", lw=1.5, ls="--", label=f"Median = {median:.0f}")
    ax.axvspan(q1, q3, color="#D8902F", alpha=.13, lw=0, label=f"IQR = {q1:.0f}–{q3:.0f}")
    ax.set_xlim(-.5, values.max() + 2.5)
    ax.set_xticks(np.arange(0, values.max() + 2, 5))
    ax.set_xlabel("Observed tissues per donor")
    ax.set_ylabel("Number of donors")
    ax.text(.01, .97, f"{donor['total_tissues'].iloc[0]} possible tissues", transform=ax.transAxes,
            ha="left", va="top", color="#2F3349")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out / "donor_observed_tissue_count_histogram")

def plot_donor_coverage_ranked(donor, out):
    """Rank view: shows the full donor coverage range without hiding individuals in bins."""
    ranked = donor.sort_values(["observed_count", "donor"]).reset_index(drop=True)
    values = ranked["observed_count"].to_numpy()
    mean, median, q1, q3 = _coverage_stats(ranked)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(1, len(ranked) + 1)
    ax.fill_between(x, 0, values, color=MISSING, linewidth=0)
    ax.plot(x, values, color=OBSERVED, lw=1.7)
    ax.axhline(mean, color="#D8902F", lw=1.5, label=f"Mean = {mean:.1f}")
    ax.axhline(median, color="#2F3349", lw=1.5, ls="--", label=f"Median = {median:.0f}")
    ax.text(.01, .97, f"IQR {q1:.0f}–{q3:.0f}", transform=ax.transAxes,
            ha="left", va="top", color="#2F3349")
    ax.set_xlim(1, len(ranked)); ax.set_ylim(0, donor["total_tissues"].iloc[0])
    ax.set_xticks([1, 100, 200, 300, 400, 500, 600, len(ranked)])
    ax.set_yticks(np.arange(0, donor["total_tissues"].iloc[0] + 1, 10))
    ax.set_xlabel("Donors ordered by observed tissue count")
    ax.set_ylabel("Observed tissues per donor")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out / "donor_observed_tissue_count_ranked")

def main():
    root = Path(__file__).resolve().parents[3]; out = Path(__file__).resolve().parent
    records_path = root / "records" / "donor_query_dictionary_20000genes_seed42.csv"
    names_path = root / "results" / "main_pmm_v11_seed42" / "formal_olga" / "hide_one" / "pmm_per_tissue_metrics.csv"
    if records_path.exists() and names_path.exists():
        donor, tissue = build_sources(root)
        donor.to_csv(out/"donor_missingness_descending_source.csv", index=False)
        tissue.to_csv(out/"tissue_missingness_ascending_source.csv", index=False)
    else:
        donor = pd.read_csv(out/"donor_missingness_descending_source.csv")
        tissue = pd.read_csv(out/"tissue_missingness_ascending_source.csv")
    plot_donor(donor,out); plot_tissue(tissue,out)
    plot_donor_coverage_histogram(donor, out)
    plot_donor_coverage_ranked(donor, out)

if __name__ == "__main__": main()
