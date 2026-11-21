"""Compute publication metrics from explicitly aligned frozen arrays.

Inputs must share record and gene order. Tissue means must be estimated from
training donors only. This script performs no data discovery and has no legacy
runtime dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT))
from dorian.metrics import hyfa_pearson_scores, overall_metrics, residual_sample_pearson  # noqa: E402


def column_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.square(x).sum(axis=0) * np.square(y).sum(axis=0))
    result = np.full(x.shape[1], np.nan)
    valid = denominator > 1e-12
    result[valid] = (x[:, valid] * y[:, valid]).sum(axis=0) / denominator[valid]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--tissue-means", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = np.load(args.truth, mmap_mode="r")
    prediction = np.load(args.prediction, mmap_mode="r")
    records = pd.read_csv(args.records)
    tissue_means = np.load(args.tissue_means)
    if truth.shape != prediction.shape:
        raise ValueError(f"truth/prediction shape mismatch: {truth.shape} vs {prediction.shape}")
    if truth.shape[0] != len(records):
        raise ValueError("record count does not match prediction rows")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("truth and predictions must be finite")
    target = records["target_tissue"].to_numpy(dtype=np.int64)
    if target.min() < 0 or target.max() >= tissue_means.shape[0]:
        raise ValueError("target tissue index is outside tissue-mean matrix")

    overall = overall_metrics(args.method, truth, prediction)
    residual_truth = np.asarray(truth) - tissue_means[target]
    residual_prediction = np.asarray(prediction) - tissue_means[target]
    residual_sample = residual_sample_pearson(truth, prediction, target, tissue_means)
    residual_gene = column_pearson(residual_truth, residual_prediction)
    normal_gene = hyfa_pearson_scores(truth, prediction)["gene_pearson"]
    overall.update(
        {
            "residual_sample_pearson_mean": float(np.nanmean(residual_sample)),
            "residual_sample_pearson_median": float(np.nanmedian(residual_sample)),
            "gene_pearson_mean": float(np.nanmean(normal_gene)),
            "residual_gene_pearson_mean": float(np.nanmean(residual_gene)),
        }
    )

    per_tissue = []
    for tissue in np.sort(np.unique(target)):
        keep = target == tissue
        row = overall_metrics(args.method, truth[keep], prediction[keep])
        row.update(
            {
                "target_tissue": int(tissue),
                "n_records": int(keep.sum()),
                "residual_sample_pearson_mean": float(np.nanmean(residual_sample[keep])),
                "residual_gene_pearson_mean": float(
                    np.nanmean(column_pearson(residual_truth[keep], residual_prediction[keep]))
                ),
            }
        )
        per_tissue.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).to_csv(args.output_dir / f"{args.method}_overall_metrics.csv", index=False)
    pd.DataFrame(per_tissue).to_csv(
        args.output_dir / f"{args.method}_per_tissue_metrics.csv", index=False
    )
    print(pd.DataFrame([overall]).to_string(index=False))


if __name__ == "__main__":
    main()
