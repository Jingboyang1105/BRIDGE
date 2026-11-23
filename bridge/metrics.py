"""Evaluation indicators for BRIDGE gene-imputation experiments.

The publication metrics are mean sample-wise, mean gene-wise, and residual
sample-wise Pearson correlations.  A flattened matrix correlation is retained
only as an explicitly named internal diagnostic.
"""

import numpy as np
import pandas as pd


def _flat_pearson(true, pred):
    a = np.asarray(true, dtype=np.float64).reshape(-1)
    b = np.asarray(pred, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if den <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / den)


def _column_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y shape mismatch: {x.shape} vs {y.shape}")
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0)
    y_std = np.std(y, axis=0)
    den = x_std * y_std
    corr = np.full(x.shape[1], np.nan, dtype=np.float64)
    valid = den > 1e-12
    if np.any(valid):
        corr[valid] = np.sum(x[:, valid] * y[:, valid], axis=0) / (x.shape[0] * den[valid])
    return corr


def hyfa_pearson_scores(true, pred):
    """Return HYFA-style sample-wise and gene-wise Pearson score vectors.

    This mirrors HYFA's pearson_correlation_score:
    sample_corr=True computes one Pearson per sample across genes.
    sample_corr=False computes one Pearson per gene across samples.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"true and pred shape mismatch: {true.shape} vs {pred.shape}")
    return {
        "sample_pearson": _column_pearson(true.T, pred.T),
        "gene_pearson": _column_pearson(true, pred),
    }


def residual_sample_pearson(true, pred, target_tissue, tissue_means):
    """Sample-wise Pearson after subtracting each record's target tissue mean."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    target_tissue = np.asarray(target_tissue, dtype=np.int64)
    tissue_means = np.asarray(tissue_means, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"true and pred shape mismatch: {true.shape} vs {pred.shape}")
    if true.shape[0] != target_tissue.shape[0]:
        raise ValueError("target_tissue length must match the number of records")
    residual_true = true - tissue_means[target_tissue]
    residual_pred = pred - tissue_means[target_tissue]
    return hyfa_pearson_scores(residual_true, residual_pred)["sample_pearson"]


def _nan_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanmedian(values))


def overall_metrics(method, true, pred, extra=None):
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if true.shape != pred.shape:
        raise ValueError(f"true and pred shape mismatch: {true.shape} vs {pred.shape}")
    err = pred - true
    row = {
        "method": method,
        "n_samples": int(true.shape[0]),
        "n_entries": int(true.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.square(err).mean())),
        "flat_pearson_internal": _flat_pearson(true, pred),
    }
    hyfa = hyfa_pearson_scores(true, pred)
    sample_mean, sample_median = _nan_summary(hyfa["sample_pearson"])
    gene_mean, gene_median = _nan_summary(hyfa["gene_pearson"])
    row.update(
        {
            "sample_pearson_mean": sample_mean,
            "sample_pearson_median": sample_median,
            "gene_pearson_mean": gene_mean,
            "gene_pearson_median": gene_median,
            # Backward-compatible aliases for older result readers. New code
            # should use the unprefixed publication metric names above.
            "hyfa_sample_pearson_mean": sample_mean,
            "hyfa_sample_pearson_median": sample_median,
            "hyfa_gene_pearson_mean": gene_mean,
            "hyfa_gene_pearson_median": gene_median,
        }
    )
    if extra:
        row.update(extra)
    return row


def per_tissue_metrics(method, true, pred, target_tissue, tissue_names=None, extra=None):
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    target_tissue = np.asarray(target_tissue)
    tissue_names = tissue_names or {}
    rows = []
    for tissue in np.sort(np.unique(target_tissue)):
        mask = target_tissue == tissue
        row = overall_metrics(method, true[mask], pred[mask], extra=extra)
        row["target_tissue"] = int(tissue)
        row["target_tissue_name"] = tissue_names.get(int(tissue), str(int(tissue)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_tissue_name").reset_index(drop=True)


