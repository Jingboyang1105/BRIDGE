"""Shared prediction-matrix adapter for our model, HYFA, and TEEBoT."""

from pathlib import Path

import numpy as np
import pandas as pd

from dorian.metrics import overall_metrics, per_tissue_metrics


def _pred_key(method):
    return f"pred_{method}"


def save_prediction_bundle(path, record_ids, gene_ids, y_true, predictions):
    """Save one benchmark prediction bundle in record_id x gene matrix format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.float32)
    arrays = {
        "record_ids": np.asarray(record_ids, dtype=object),
        "gene_ids": np.asarray(gene_ids, dtype=object),
        "y_true": y_true,
    }
    if y_true.shape != (len(arrays["record_ids"]), len(arrays["gene_ids"])):
        raise ValueError("y_true shape must equal len(record_ids) x len(gene_ids)")
    for method, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != y_true.shape:
            raise ValueError(f"prediction shape mismatch for {method}: {pred.shape} vs {y_true.shape}")
        arrays[_pred_key(method)] = pred
    np.savez_compressed(path, **arrays)


def load_prediction_bundle(path):
    loaded = np.load(Path(path), allow_pickle=True)
    methods = sorted(k[len("pred_") :] for k in loaded.files if k.startswith("pred_"))
    return {
        "record_ids": loaded["record_ids"].astype(str),
        "gene_ids": loaded["gene_ids"].astype(str),
        "y_true": loaded["y_true"].astype(np.float32),
        "predictions": {method: loaded[_pred_key(method)].astype(np.float32) for method in methods},
    }


def _aligned_record_metadata(bundle_record_ids, records):
    records = records.copy()
    records["record_id"] = records["record_id"].astype(str)
    indexed = records.set_index("record_id", drop=False)
    missing = [rid for rid in bundle_record_ids if rid not in indexed.index]
    if missing:
        raise ValueError(f"record_id mismatch: {missing[:3]}")
    aligned = indexed.loc[list(bundle_record_ids)].reset_index(drop=True)
    if aligned["record_id"].tolist() != list(bundle_record_ids):
        raise ValueError("record_id mismatch after alignment")
    return aligned


def evaluate_prediction_bundle(bundle_path, records, extra=None):
    """Evaluate every method in a saved prediction bundle with aligned metrics."""
    bundle = load_prediction_bundle(bundle_path)
    meta = _aligned_record_metadata(bundle["record_ids"], records)
    target_tissue = meta["target_tissue"].to_numpy()
    tissue_names = {}
    if "target_tissue_name" in meta.columns:
        for tissue, name in zip(meta["target_tissue"], meta["target_tissue_name"]):
            tissue_names[int(tissue)] = str(name)
    extra = dict(extra or {})
    overall_rows = []
    per_tissue_rows = []
    known_rows = []
    for method, pred in bundle["predictions"].items():
        overall_rows.append(overall_metrics(method, bundle["y_true"], pred, extra=extra))
        per_tissue_rows.append(
            per_tissue_metrics(method, bundle["y_true"], pred, target_tissue, tissue_names=tissue_names, extra=extra)
        )
        if "actual_known_count" in meta.columns:
            for count in np.sort(meta["actual_known_count"].unique()):
                mask = meta["actual_known_count"].to_numpy() == count
                count_extra = dict(extra)
                count_extra["actual_known_count"] = int(count)
                known_rows.append(overall_metrics(method, bundle["y_true"][mask], pred[mask], extra=count_extra))
    overall = pd.DataFrame(overall_rows).sort_values("mae").reset_index(drop=True)
    per_tissue = pd.concat(per_tissue_rows, ignore_index=True).sort_values(["method", "target_tissue"]).reset_index(
        drop=True
    )
    by_known = pd.DataFrame(known_rows)
    if not by_known.empty:
        by_known = by_known.sort_values(["actual_known_count", "mae"]).reset_index(drop=True)
    return overall, per_tissue, by_known


def write_evaluation_tables(bundle_path, records, out_dir, prefix, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall, per_tissue, by_known = evaluate_prediction_bundle(bundle_path, records, extra=extra)
    overall.to_csv(out_dir / f"{prefix}_overall_metrics.csv", index=False)
    per_tissue.to_csv(out_dir / f"{prefix}_per_tissue_metrics.csv", index=False)
    if not by_known.empty:
        by_known.to_csv(out_dir / f"{prefix}_by_known_count_metrics.csv", index=False)
    return overall, per_tissue, by_known


