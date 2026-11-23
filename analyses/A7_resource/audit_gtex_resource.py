"""Read-only GTEx v11 inventory for the BRIDGE completed resource."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def donor_id(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive GTEx donor from sample ID: {sample_id}")
    return "-".join(parts[:2])


def _normalise_meta(meta: pd.DataFrame, tissue_encoder: Dict[str, int]) -> pd.DataFrame:
    frame = meta.copy()
    if "tissue_idx" not in frame.columns:
        raise ValueError("Metadata must contain tissue_idx")
    frame["tissue_idx"] = pd.to_numeric(frame["tissue_idx"], errors="raise").astype(int)
    inverse = {int(index): str(name) for name, index in tissue_encoder.items()}
    unknown = sorted(set(frame["tissue_idx"]) - set(inverse))
    if unknown:
        raise ValueError(f"Metadata contains tissue indices absent from encoder: {unknown}")
    frame["tissue_name"] = frame["tissue_idx"].map(inverse)
    frame["donor"] = [donor_id(sample_id) for sample_id in frame.index]
    return frame


def build_observed_grid(meta: pd.DataFrame, tissue_encoder: Dict[str, int]) -> pd.DataFrame:
    frame = _normalise_meta(meta, tissue_encoder)
    donors = sorted(frame["donor"].unique())
    tissues = sorted((int(index), str(name)) for name, index in tissue_encoder.items())
    measured = set(zip(frame["donor"], frame["tissue_idx"]))
    rows = []
    for donor in donors:
        for tissue_idx, tissue_name in tissues:
            rows.append(
                {
                    "donor": donor,
                    "target_tissue": tissue_idx,
                    "target_tissue_name": tissue_name,
                    "observed": (donor, tissue_idx) in measured,
                }
            )
    return pd.DataFrame(rows)


def classify_context(n_observed: int, classes: dict) -> str:
    for label, bounds in classes.items():
        lower, upper = bounds
        if int(n_observed) >= int(lower) and (upper is None or int(n_observed) <= int(upper)):
            return str(label)
    raise ValueError(f"Observed-tissue count {n_observed} does not match a context class")


def donor_manifest(
    meta: pd.DataFrame, tissue_encoder: Dict[str, int], classes: dict
) -> pd.DataFrame:
    frame = _normalise_meta(meta, tissue_encoder)
    counts = frame.groupby("donor")["tissue_idx"].nunique().sort_index()
    total_tissues = len(tissue_encoder)
    out = counts.rename("n_observed_tissues").reset_index()
    out["n_missing_tissues"] = total_tissues - out["n_observed_tissues"]
    out["context_class"] = [classify_context(value, classes) for value in out["n_observed_tissues"]]
    return out


def tissue_manifest(
    meta: pd.DataFrame,
    tissue_encoder: Dict[str, int],
    prior_eval: Optional[pd.DataFrame],
    rules: dict,
) -> pd.DataFrame:
    frame = _normalise_meta(meta, tissue_encoder)
    donor_counts = frame.groupby("donor")["tissue_idx"].nunique()
    frame["donor_is_multitissue"] = frame["donor"].map(donor_counts).ge(2)
    measured = frame.drop_duplicates(["donor", "tissue_idx"])
    rows = []
    prior_tissues: set[int] = set()
    if prior_eval is not None:
        column = "target_tissue" if "target_tissue" in prior_eval.columns else "tissue"
        prior_tissues = set(pd.to_numeric(prior_eval[column], errors="raise").astype(int))
    min_donors = int(rules["minimum_measured_donors_for_eligible_tissue"])
    min_multi = int(rules["minimum_multitissue_donors_for_eligible_tissue"])
    for tissue_name, tissue_idx_raw in sorted(tissue_encoder.items(), key=lambda item: int(item[1])):
        tissue_idx = int(tissue_idx_raw)
        block = measured[measured["tissue_idx"] == tissue_idx]
        n_donors = int(block["donor"].nunique())
        n_multi = int(block.loc[block["donor_is_multitissue"], "donor"].nunique())
        status = "eligible" if n_donors >= min_donors and n_multi >= min_multi else "low-support"
        reasons = []
        if n_donors < min_donors:
            reasons.append(f"measured_donors<{min_donors}")
        if n_multi < min_multi:
            reasons.append(f"multitissue_donors<{min_multi}")
        rows.append(
            {
                "target_tissue": tissue_idx,
                "target_tissue_name": str(tissue_name),
                "n_measured_samples": int((frame["tissue_idx"] == tissue_idx).sum()),
                "n_measured_donors": n_donors,
                "n_multitissue_donors": n_multi,
                "n_hide_one_training_records": n_multi,
                "tissue_mean_available": n_donors > 0,
                "in_locked_encoder": True,
                "in_prior_evaluation": tissue_idx in prior_tissues if prior_eval is not None else False,
                "production_status": status,
                "status_reason": ";".join(reasons) if reasons else "support_thresholds_passed",
            }
        )
    return pd.DataFrame(rows)


def estimate_storage(n_profiles: int, n_genes: int) -> dict[str, int]:
    float32_bytes = int(n_profiles) * int(n_genes) * 4
    return {
        "n_profiles": int(n_profiles),
        "n_genes": int(n_genes),
        "float32_bytes": float32_bytes,
        "float32_gib": float32_bytes / (1024**3),
    }


def validate_inputs(
    expr: np.ndarray,
    meta: pd.DataFrame,
    tissue_encoder: Dict[str, int],
    gene_ids: List[str],
    expected_genes: int,
) -> None:
    if len(meta) != int(expr.shape[0]):
        raise ValueError(f"Expression rows ({expr.shape[0]}) differ from metadata rows ({len(meta)})")
    if int(expr.shape[1]) != int(expected_genes):
        raise ValueError(f"Expected {expected_genes} expression genes, found {expr.shape[1]}")
    if meta.index.has_duplicates:
        raise ValueError("Metadata sample IDs are duplicated")
    if len(gene_ids) != int(expected_genes) or len(set(map(str, gene_ids))) != int(expected_genes):
        raise ValueError("Gene manifest must contain the expected number of unique genes")
    _normalise_meta(meta, tissue_encoder)
    rows_per_chunk = max(1, min(256, int(expr.shape[0])))
    for start in range(0, int(expr.shape[0]), rows_per_chunk):
        if not np.isfinite(np.asarray(expr[start : start + rows_per_chunk])).all():
            raise ValueError("Expression matrix contains non-finite values")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_gene_ids(path: Path) -> List[str]:
    frame = pd.read_csv(path)
    preferred = [column for column in ("gene_id", "gene", "Name") if column in frame.columns]
    column = preferred[0] if preferred else frame.columns[0]
    return frame[column].astype(str).tolist()


def run_audit(
    compat_dir: Path,
    gene_order_path: Path,
    config_path: Path,
    out_dir: Path,
    prior_evaluation_path: Optional[Path] = None,
) -> dict:
    compat_dir = Path(compat_dir)
    expr_path = compat_dir / "expr.npy"
    meta_path = compat_dir / "meta.csv"
    encoder_path = compat_dir / "tissue_encoder.json"
    required = [expr_path, meta_path, encoder_path, Path(gene_order_path), Path(config_path)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required audit inputs: {missing}")

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with encoder_path.open("r", encoding="utf-8") as handle:
        tissue_encoder = {str(name): int(index) for name, index in json.load(handle).items()}
    meta = pd.read_csv(meta_path, index_col=0)
    expr = np.load(expr_path, mmap_mode="r")
    gene_ids = load_gene_ids(Path(gene_order_path))
    validate_inputs(expr, meta, tissue_encoder, gene_ids, int(config["gene_count"]))

    prior_eval = pd.read_csv(prior_evaluation_path) if prior_evaluation_path else None
    donors = donor_manifest(meta, tissue_encoder, config["context_classes"])
    tissues = tissue_manifest(meta, tissue_encoder, prior_eval, config)
    grid = build_observed_grid(meta, tissue_encoder)
    grid = grid.merge(
        donors[["donor", "n_observed_tissues", "context_class"]], on="donor", validate="many_to_one"
    ).merge(
        tissues[["target_tissue", "production_status"]], on="target_tissue", validate="many_to_one"
    )
    observed = grid[grid["observed"]].copy()
    candidates = grid[(~grid["observed"]) & grid["production_status"].eq("eligible")].copy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    donors.to_csv(out_dir / "donor_coverage.csv", index=False)
    tissues.to_csv(out_dir / "tissue_support.csv", index=False)
    observed.to_csv(out_dir / "observed_donor_tissue_cells.csv", index=False)
    candidates.to_csv(out_dir / "candidate_missing_donor_tissue_cells.csv", index=False)

    storage = {
        "imputed_only": estimate_storage(len(candidates), int(config["gene_count"])),
        "completed_rectangular_all_tissues": estimate_storage(len(grid), int(config["gene_count"])),
    }
    with (out_dir / "storage_projection.json").open("w", encoding="utf-8") as handle:
        json.dump(storage, handle, indent=2)

    summary = {
        "dataset": config["dataset"],
        "n_samples": int(len(meta)),
        "n_donors": int(donors["donor"].nunique()),
        "n_tissues": int(len(tissue_encoder)),
        "n_genes": int(config["gene_count"]),
        "n_observed_donor_tissue_cells": int(len(observed)),
        "n_candidate_missing_cells": int(len(candidates)),
        "n_eligible_tissues": int(tissues["production_status"].eq("eligible").sum()),
        "n_low_support_tissues": int(tissues["production_status"].eq("low-support").sum()),
        "context_class_counts": {str(k): int(v) for k, v in donors["context_class"].value_counts().items()},
    }
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    inputs = {}
    for path in required + ([Path(prior_evaluation_path)] if prior_evaluation_path else []):
        inputs[str(path.resolve())] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    outputs = {}
    for name in (
        "audit_summary.json",
        "donor_coverage.csv",
        "tissue_support.csv",
        "observed_donor_tissue_cells.csv",
        "candidate_missing_donor_tissue_cells.csv",
        "storage_projection.json",
    ):
        path = out_dir / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    with (out_dir / "input_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"inputs": inputs, "outputs": outputs}, handle, indent=2)
    (out_dir / "PRELIMINARY_AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compat-dir", type=Path, required=True)
    parser.add_argument("--gene-order", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prior-evaluation", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        args.compat_dir, args.gene_order, args.config, args.out_dir, args.prior_evaluation
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
