from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "main_figures" / "paper_clean_revisions"

KNOWN_SOURCE = OUT_DIR / "main_gate_known_count_no_titles_gate_source.csv"
TARGET_SOURCE = ROOT / "results" / "main_figures" / "paper_selected_final" / "main_gate_strategy_by_tissue_selected_hires_source.csv"


SYSTEM_COLORS = {
    "CNS": "#A9A711",
    "GI": "#B9855A",
    "Cardiovascular": "#6B4C7A",
    "Metabolic/Muscle": "#D2A800",
    "Skin": "#3F98C5",
    "Immune/Blood": "#C33B94",
    "Endocrine": "#4AA36D",
    "Reproductive/Breast": "#D5798C",
    "Urinary": "#8B7766",
    "Respiratory": "#6FA83E",
    "Other": "#8EB8C0",
}


def setup() -> None:
    mpl.use("Agg")
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
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def short_label(name: str) -> str:
    replacements = {
        "Brain - Anterior cingulate cortex (BA24)": "Ant. cingulate",
        "Brain - Caudate (basal ganglia)": "Caudate",
        "Brain - Nucleus accumbens (basal ganglia)": "Nucleus accumbens",
        "Brain - Putamen (basal ganglia)": "Putamen",
        "Brain - Frontal Cortex (BA9)": "Frontal cortex",
        "Brain - Cerebellar Hemisphere": "Cerebellar hemi.",
        "Brain - Spinal cord (cervical c-1)": "Spinal cord",
        "Esophagus - Gastroesophageal Junction": "Esoph. GEJ",
        "Esophagus - Muscularis": "Esoph. muscularis",
        "Esophagus - Mucosa": "Esoph. mucosa",
        "Small Intestine - Terminal Ileum": "Small intestine",
        "Small Intestine - Terminal Ileum - Mixed Cell": "Small int. mixed",
        "Small Intestine - Terminal Ileum - Lymphoid Aggregate": "Small int. lymphoid",
        "Colon - Transverse - Mixed Cell": "Colon trans. mixed",
        "Colon - Transverse - Muscularis": "Colon trans. musc.",
        "Adipose - Subcutaneous": "Adipose subcut.",
        "Adipose - Visceral (Omentum)": "Adipose visceral",
        "Skin - Sun Exposed (Lower leg)": "Skin sun",
        "Skin - Not Sun Exposed (Suprapubic)": "Skin not sun",
        "Cells - EBV-transformed lymphocytes": "Cells EBV",
        "Cells - Cultured fibroblasts": "Cultured fibro.",
        "Breast - Mammary Tissue": "Breast mammary",
        "Heart - Atrial Appendage": "Heart atrial app.",
        "Heart - Left Ventricle": "Heart L. vent.",
        "Artery - Coronary": "Artery coronary",
        "Artery - Tibial": "Artery tibial",
        "Liver - Mixed Cell": "Liver mixed",
        "Liver Hepatocyte": "Liver hepatocyte",
    }
    if name in replacements:
        return replacements[name]
    return name.replace("Brain - ", "").replace(" - ", " ")[:26]


def save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    setup()
    known = pd.read_csv(KNOWN_SOURCE).rename(columns={"actual_known_count": "query_known_count"})
    known = known.sort_values("query_known_count")
    target = pd.read_csv(TARGET_SOURCE).copy()
    target = target.rename(
        columns={
            "system": "target_tissue_system",
            "retrieval_weight": "mean_cml_weight",
            "n": "n_records",
        }
    )
    target = target[target["n_records"] >= 10].sort_values("mean_cml_weight", ascending=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    known.to_csv(OUT_DIR / "adaptive_prediction_mixer_behavior_known_count_source.csv", index=False)
    target.to_csv(OUT_DIR / "adaptive_prediction_mixer_behavior_target_tissue_source.csv", index=False)

    teal = "#16736F"
    brown = "#B07A5A"
    grid = "#E7E7E7"

    fig = plt.figure(figsize=(9.2, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.72], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_box_aspect(1)
    ax1.plot(
        known["query_known_count"],
        known["mean_donor_level_average_weight"],
        color=brown,
        marker="o",
        markersize=3.2,
        linewidth=1.6,
        label="donor-level average residual expert",
    )
    ax1.plot(
        known["query_known_count"],
        known["mean_cml_weight"],
        color=teal,
        marker="o",
        markersize=3.2,
        linewidth=1.6,
        label="sample matching expert",
    )
    ax1.set_xlabel("number of observed tissues")
    ax1.set_ylabel("mean mixer weight")
    ax1.set_ylim(0.40, 0.62)
    ax1.set_yticks(np.arange(0.40, 0.621, 0.05))
    ax1.grid(axis="both", color=grid, linewidth=0.65)
    ax1.legend(loc="upper right", fontsize=6.2, handlelength=1.8)
    ax1.text(-0.18, 1.03, "c", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(target))
    colors = [SYSTEM_COLORS.get(x, "#9AA3A7") for x in target["target_tissue_system"]]
    ax2.axvline(0.5, color="#989898", linewidth=0.75, linestyle="--", zorder=0)
    ax2.hlines(y, 0.5, target["mean_cml_weight"], color=colors, alpha=0.70, linewidth=1.15)
    ax2.scatter(
        target["mean_cml_weight"],
        y,
        s=12,
        c=colors,
        edgecolor="#293238",
        linewidth=0.25,
        zorder=3,
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels([short_label(x) for x in target["target_tissue_name"]], fontsize=4.8)
    ax2.set_xlabel("mean sample matching weight")
    ax2.set_xlim(0.28, 0.58)
    ax2.set_xticks([0.3, 0.4, 0.5])
    ax2.grid(axis="x", color=grid, linewidth=0.65)
    ax2.tick_params(axis="y", length=2.5, pad=1.5)
    ax2.text(-0.20, 1.03, "d", transform=ax2.transAxes, fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.09)
    base = OUT_DIR / "adaptive_prediction_mixer_behavior"
    save_all(fig, base)
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
