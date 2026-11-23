"""Plot the formal hide-one retrieval composition in the established heatmap style."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
MECH = ROOT / "03_How_DORIAN_Works"
SOURCE = MECH / "data" / "hide_one_retrieval" / "hide_one_retrieval_composition_matrix.csv"
ORDER_TEMPLATE = MECH / "data" / "source" / "global_retrieval_composition_matrix.csv"
OUT = MECH / "figures" / "final" / "hide_one_retrieval_composition_heatmap_full"
PAPER_OUT = ROOT / "Results_for_Paper" / "All_Final_Figures_By_Experiment" / "03_Model_Behavior"


def short_label(name: str) -> str:
    replacements = {
        "Brain - Amygdala": "Amygdala",
        "Brain - Anterior cingulate cortex (BA24)": "Ant. cingulate",
        "Brain - Caudate (basal ganglia)": "Caudate",
        "Brain - Cerebellar Hemisphere": "Cerebellar hemi.",
        "Brain - Cerebellum": "Cerebellum",
        "Brain - Cortex": "Cortex",
        "Brain - Frontal Cortex (BA9)": "Frontal cortex",
        "Brain - Hippocampus": "Hippocampus",
        "Brain - Hypothalamus": "Hypothalamus",
        "Brain - Nucleus accumbens (basal ganglia)": "Nucleus accumbens",
        "Brain - Putamen (basal ganglia)": "Putamen",
        "Brain - Spinal cord (cervical c-1)": "Spinal cord",
        "Brain - Substantia nigra": "Substantia nigra",
        "Colon - Sigmoid": "Colon sigmoid",
        "Colon - Transverse": "Colon transverse",
        "Colon - Transverse - Mucosa": "Colon transverse mucosa",
        "Colon - Transverse - Mixed Cell": "Colon transverse mixed",
        "Colon - Transverse - Muscularis": "Colon transverse muscularis",
        "Esophagus - Gastroesophageal Junction": "Esoph. GEJ",
        "Esophagus - Mucosa": "Esoph. mucosa",
        "Esophagus - Muscularis": "Esoph. muscularis",
        "Small Intestine - Terminal Ileum": "Small int.",
        "Small Intestine - Terminal Ileum - Lymphoid Aggregate": "Small int. lymphoid",
        "Small Intestine - Terminal Ileum - Mixed Cell": "Small int. mixed",
        "Stomach - Mixed Cell": "Stomach mixed",
        "Stomach - Mucosa": "Stomach mucosa",
        "Stomach - Muscularis": "Stomach muscularis",
        "Pancreas - Acini": "Pancreas acini",
        "Pancreas - Islets": "Pancreas islets",
        "Pancreas - Mixed Cell": "Pancreas mixed",
        "Liver - Hepatocyte": "Liver hepatocyte",
        "Liver - Mixed Cell": "Liver mixed cell",
        "Liver - Portal Tract": "Liver portal tract",
        "Artery - Aorta": "Artery Aorta",
        "Artery - Coronary": "Artery Coronary",
        "Artery - Tibial": "Artery Tibial",
        "Heart - Atrial Appendage": "Heart atrial app.",
        "Heart - Left Ventricle": "Heart L. vent.",
        "Adipose - Subcutaneous": "Adipose subcut.",
        "Adipose - Visceral (Omentum)": "Adipose visceral",
        "Muscle - Skeletal": "Muscle skeletal",
        "Skin - Not Sun Exposed (Suprapubic)": "Skin not sun",
        "Skin - Sun Exposed (Lower leg)": "Skin sun",
        "Cells - EBV-transformed lymphocytes": "Cells EBV",
        "Cells - Cultured fibroblasts": "Cells cultured fibro.",
        "Breast - Mammary Tissue": "Breast mammary",
        "Kidney - Cortex": "Kidney Cortex",
        "Kidney - Medulla": "Kidney Medulla",
        "Cervix - Ectocervix": "Cervix Ectocervix",
        "Cervix - Endocervix": "Cervix Endocervix",
        "Minor Salivary Gland": "Minor Salivary Gland",
        "Nerve - Tibial": "Nerve Tibial",
    }
    return replacements.get(str(name), str(name))


def save_all(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})


def main() -> int:
    matrix = pd.read_csv(SOURCE, index_col=0)
    template = pd.read_csv(ORDER_TEMPLATE, index_col=0)
    if set(matrix.index) != set(template.index):
        raise ValueError("hide-one target tissues do not match the established 65-tissue order")
    if set(matrix.columns) != set(template.columns):
        raise ValueError("hide-one retrieved tissues do not match the established 68-tissue order")
    matrix = matrix.reindex(index=template.index, columns=template.columns)
    if matrix.shape != (65, 68) or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("invalid hide-one composition matrix")
    if not np.allclose(matrix.sum(axis=1).to_numpy(), 1.0):
        raise ValueError("hide-one composition rows do not sum to one")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })
    cmap = LinearSegmentedColormap.from_list(
        "bridge_black_teal_mint",
        ["#080304", "#1c1324", "#332758", "#385f9f", "#38a6a6", "#62d0ad", "#e1f6df"],
    )
    fig, ax = plt.subplots(figsize=(13.8, 12.2))
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, vmin=0, vmax=1, square=False, linewidths=0,
        cbar_kws={"label": "top-k fraction", "shrink": 0.94}, rasterized=True,
    )
    ax.set_xlabel("retrieved tissue", fontsize=8)
    ax.set_ylabel("target tissue", fontsize=8)
    ax.set_xticklabels([short_label(x) for x in matrix.columns], rotation=90, fontsize=4.4)
    ax.set_yticklabels([short_label(x) for x in matrix.index], rotation=0, fontsize=4.4)
    ax.tick_params(length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#222222")
    save_all(fig, OUT)
    plt.close(fig)

    ordered_source = MECH / "data" / "hide_one_retrieval" / "hide_one_retrieval_composition_matrix_ordered.csv"
    matrix.to_csv(ordered_source)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    for extension in [".png", ".svg", ".pdf", ".tiff"]:
        shutil.copy2(OUT.with_suffix(extension), PAPER_OUT / OUT.with_suffix(extension).name)
    shutil.copy2(ordered_source, PAPER_OUT / ordered_source.name)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
