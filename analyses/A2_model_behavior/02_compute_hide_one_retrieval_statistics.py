"""Compute manuscript-facing retrieval statistics from formal hide-one data only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "source_data" / "hide_one_retrieval_composition_long.csv"
OUTPUT = HERE / "outputs" / "tables" / "hide_one_retrieval_statistics.json"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    diagonal = frame.loc[frame["target_tissue"] == frame["retrieved_tissue"]].copy()
    if diagonal["target_tissue"].nunique() != frame["target_tissue"].nunique():
        raise ValueError("Every hide-one target tissue must have a diagonal entry")
    total_topk = int(frame["n_topk"].sum())
    diagonal_topk = int(diagonal["n_topk"].sum())
    top_rows = frame.loc[frame.groupby("target_tissue")["n_topk"].idxmax()]
    rho, p_value = spearmanr(diagonal["n_target_records"], diagonal["topk_fraction"])
    result = {
        "source": str(SOURCE.relative_to(HERE)),
        "n_target_tissues": int(diagonal.shape[0]),
        "n_test_records": int(diagonal["n_target_records"].sum()),
        "top_k": int(total_topk // diagonal["n_target_records"].sum()),
        "total_retrieval_slots": total_topk,
        "same_tissue_retrieval_slots": diagonal_topk,
        "weighted_same_tissue_fraction": diagonal_topk / total_topk,
        "mean_per_tissue_same_tissue_fraction": float(diagonal["topk_fraction"].mean()),
        "median_per_tissue_same_tissue_fraction": float(diagonal["topk_fraction"].median()),
        "targets_with_same_tissue_as_most_retrieved": int(
            (top_rows["target_tissue"] == top_rows["retrieved_tissue"]).sum()
        ),
        "spearman_target_support_vs_same_tissue_fraction": float(rho),
        "spearman_p_value": float(p_value),
        "finite": bool(np.isfinite(frame["topk_fraction"]).all()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
