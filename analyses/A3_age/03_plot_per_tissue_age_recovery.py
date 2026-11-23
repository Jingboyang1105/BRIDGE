from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
RESULTS = EXPERIMENT / "results" / "all_eligible_formal_seed42"
INPUT = RESULTS / "age_recovery_endpoints.csv"
OUT = RESULTS / "figures"

COLORS = {"BRIDGE": "#6C5CE7", "HYFA": "#F2B84B"}


def shorten(name: str) -> str:
    replacements = {
        "Brain - ": "Brain – ",
        "Cells - ": "Cells – ",
        "Esophagus - ": "Esophagus – ",
        "Adipose - ": "Adipose – ",
        "Artery - ": "Artery – ",
        "Heart - ": "Heart – ",
        "Colon - ": "Colon – ",
        "Skin - ": "Skin – ",
        "Cervix - ": "Cervix – ",
        "Kidney - ": "Kidney – ",
        "Small Intestine - ": "Small intestine – ",
    }
    result = str(name)
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = result.replace("EBV-transformed lymphocytes", "EBV lymphocytes")
    result = result.replace("Cultured fibroblasts", "Fibroblasts")
    result = result.replace("Gastroesophageal Junction", "GE junction")
    result = result.replace("Anterior cingulate cortex (BA24)", "Anterior cingulate (BA24)")
    result = result.replace("Frontal Cortex (BA9)", "Frontal cortex (BA9)")
    result = result.replace("Spinal cord (cervical c-1)", "Spinal cord (C1)")
    result = result.replace("Cerebellar Hemisphere", "Cerebellar hemisphere")
    result = result.replace("Nucleus accumbens (basal ganglia)", "Nucleus accumbens")
    result = result.replace("Substantia nigra", "Substantia nigra")
    result = result.replace("Suprapubic (not sun exposed)", "Suprapubic")
    result = result.replace("Lower leg (sun exposed)", "Lower leg")
    result = result.replace("Atrial Appendage", "Atrial appendage")
    result = result.replace("Left Ventricle", "Left ventricle")
    result = result.replace("Muscularis", "muscularis")
    result = result.replace("Mucosa", "mucosa")
    return result


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = {"tissue", "tissue_name", "missing_rate", "repeat", "arm", "beta_pearson"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = frame.loc[frame["arm"].isin(COLORS)].copy()
    if frame["tissue"].nunique() != 49 or set(frame["arm"]) != set(COLORS):
        raise ValueError("Expected exactly 49 tissues and BRIDGE/HYFA")
    if not np.isfinite(frame["beta_pearson"]).all():
        raise ValueError("Non-finite beta Pearson values")

    cell_counts = frame.groupby(["tissue", "arm"]).size()
    if not cell_counts.eq(20).all():
        raise ValueError(f"Expected 20 cells per tissue-method, got {cell_counts.value_counts().to_dict()}")

    summary = (
        frame.groupby(["tissue", "tissue_name", "arm"], as_index=False)["beta_pearson"]
        .agg(mean="mean", sd="std", n_cells="size")
    )
    wide = summary.pivot(index=["tissue", "tissue_name"], columns="arm", values="mean").reset_index()
    wide["bridge_minus_hyfa"] = wide["BRIDGE"] - wide["HYFA"]
    wide = wide.sort_values("BRIDGE", ascending=False, kind="stable").reset_index(drop=True)
    wins = int((wide["bridge_minus_hyfa"] > 0).sum())
    if wins != 43:
        raise ValueError(f"Expected 43 BRIDGE tissue wins, got {wins}")

    order = wide["tissue_name"].tolist()
    rank = {name: i for i, name in enumerate(order)}
    summary["display_order"] = summary["tissue_name"].map(rank)
    summary = summary.sort_values(["display_order", "arm"], key=lambda s: s.map({"BRIDGE": 0, "HYFA": 1}) if s.name == "arm" else s)
    source = summary.merge(wide[["tissue", "bridge_minus_hyfa"]], on="tissue", how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    source.to_csv(OUT / "age_beta_pearson_per_tissue_source_data.csv", index=False)
    (OUT / "age_beta_pearson_per_tissue_QA.json").write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "tissues": 49,
                "methods": list(COLORS),
                "cells_per_tissue_method": 20,
                "bridge_tissue_wins": wins,
                "sort": "BRIDGE mean beta_pearson descending",
                "colors": COLORS,
                "all_finite": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
        }
    )

    y = np.arange(len(wide))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 12.6), constrained_layout=False)
    ax.barh(y - height / 2, wide["BRIDGE"], height, color=COLORS["BRIDGE"], label="BRIDGE", linewidth=0)
    ax.barh(y + height / 2, wide["HYFA"], height, color=COLORS["HYFA"], label="HYFA", linewidth=0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.65, len(wide) - 0.35)
    ax.set_xlabel("Mean age-beta Pearson", fontsize=9, labelpad=7)
    ax.set_ylabel("Tissue", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([shorten(v) for v in wide["tissue_name"]], fontsize=6.6)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.1)], fontsize=7.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.018), ncol=2, fontsize=8, handlelength=1.5)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.975, bottom=0.055)

    base = OUT / "age_beta_pearson_per_tissue"
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
