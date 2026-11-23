"""Run formal GTEx hide-one training-size scaling for BRIDGE."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from CML_Gene_Imputation.src.data import load_gene_set
from CML_Gene_Imputation.src.metrics import overall_metrics, residual_sample_pearson
from CML_Gene_Imputation.src.models.cml import CrossTissueCML, RecordDataset
from CML_Gene_Imputation.src.models.two_expert_gate import predict_gate, train_gate
from CML_Gene_Imputation.src.records import build_hide_one_records, donor_to_indices
from CML_Gene_Imputation.src.splits import donor_id, split_donors
from CML_Gene_Imputation.src.train_hide_one import (
    encode_target_bank,
    limit_records,
    predict_retrieval,
    records_to_arrays,
    tissue_mean_matrix,
    train_cml_model,
)
try:
    from CML_Gene_Imputation.src.hide_one_scaling_protocol import (
        build_hide_one_protocol_records,
        ordered_record_hash,
        validate_formal_test_records,
    )
except ImportError:
    from hide_one_scaling_protocol import (
        build_hide_one_protocol_records,
        ordered_record_hash,
        validate_formal_test_records,
    )


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "hide_one_training_size_scaling"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--gate_epochs", type=int, default=50)
    p.add_argument("--gate_batch", type=int, default=4096)
    p.add_argument("--max_records", type=int, default=0, help="Smoke/debug limit per split; 0 means full.")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


def parse_fractions(value):
    fractions = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if not fractions:
        raise ValueError("--fractions must contain at least one value")
    bad = [f for f in fractions if f <= 0 or f > 1]
    if bad:
        raise ValueError(f"fractions must be in (0, 1], got {bad}")
    return fractions


def subset_train_indices_by_donor_fraction(meta, train_idx, val_idx, test_idx, fraction, seed):
    """Select a seeded subset of train donors while keeping val/test untouched."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if float(fraction) >= 1.0:
        return train_idx.copy()

    donors = _donors_for_indices(meta, train_idx)
    train_donors = pd.Index(donors).unique().to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = train_donors.copy()
    rng.shuffle(shuffled)
    n_select = max(1, int(np.ceil(len(shuffled) * float(fraction))))
    selected = set(shuffled[:n_select])
    subset = np.array([int(i) for i, donor in zip(train_idx, donors) if str(donor) in selected], dtype=np.int64)

    blocked = set(np.asarray(val_idx, dtype=np.int64)) | set(np.asarray(test_idx, dtype=np.int64))
    overlap = blocked & set(subset)
    if overlap:
        raise ValueError(f"train subset overlaps val/test indices: {sorted(overlap)[:5]}")
    return subset


def _donors_for_indices(meta, indices):
    if "donor" in meta.columns:
        return meta.iloc[np.asarray(indices, dtype=np.int64)]["donor"].astype(str).to_numpy()
    return np.array([donor_id(meta.index[int(i)]) for i in indices], dtype=object)


def _fraction_tag(fraction):
    pct = int(round(float(fraction) * 100))
    return f"train_{pct:03d}pct"


def _row_with_residual(method, true, pred, arrays, tissue_means, extra):
    row = overall_metrics(method, true, pred, extra=extra)
    residual = residual_sample_pearson(true, pred, arrays["target_tissue"], tissue_means)
    row["residual_sample_pearson_mean"] = float(np.nanmean(residual))
    row["residual_sample_pearson_median"] = float(np.nanmedian(residual))
    return row


def _expert_stack(arrays, cml_pred):
    return np.stack([arrays["donor_level_average"], cml_pred], axis=1).astype(np.float32)


def predict_training_retrieval(model, train_z, train_expr, train_ds, train_records, batch, k, device):
    """Exclude every training query's own target from its retrieval memory.

    New runtimes mask by target identity. Older Olga runtimes mask the aligned
    row only; that fallback is equivalent here because hide-one training has
    exactly one record per target.
    """
    target_indices = np.asarray([int(record["idx"]) for record in train_records], dtype=np.int64)
    if len(target_indices) != len(set(target_indices.tolist())):
        raise ValueError("hide-one training must contain exactly one record per target")
    try:
        return predict_retrieval(
            model,
            train_z,
            train_expr,
            train_ds,
            batch,
            k,
            device,
            exclude_self=True,
            train_target_indices=target_indices,
        )
    except TypeError as error:
        if "train_target_indices" not in str(error):
            raise
        return predict_retrieval(
            model, train_z, train_expr, train_ds, batch, k, device, exclude_self=True
        )


def run_fraction(data, split, fraction, args, out_dir):
    # Every fraction is a separate formal experiment. Reset all RNGs so its
    # initialization is seed 42 rather than the state left by the prior run.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    expr = data.expr
    meta = data.meta
    tissue = meta["tissue_idx"].to_numpy()
    subset_idx = subset_train_indices_by_donor_fraction(
        meta, split.train_idx, split.val_idx, split.test_idx, fraction, args.seed
    )
    maps = {
        "train": donor_to_indices(meta, subset_idx),
        "val": donor_to_indices(meta, split.val_idx),
        "test": donor_to_indices(meta, split.test_idx),
    }
    tissue_means = tissue_mean_matrix(expr, tissue, subset_idx)
    train_records, val_records, test_records = build_hide_one_protocol_records(
        meta,
        subset_idx,
        split.val_idx,
        split.test_idx,
        maps["train"],
        maps["val"],
        maps["test"],
        build_hide_one_records,
    )
    train_records = limit_records(train_records, args.max_records, args.seed)
    val_records = limit_records(val_records, args.max_records, args.seed + 1)
    test_records = limit_records(test_records, args.max_records, args.seed + 2)
    if int(args.max_records) == 0:
        test_record_hash = validate_formal_test_records(test_records)
    else:
        test_record_hash = ordered_record_hash(test_records)
    print(
        f"fraction={fraction:.3f} train_samples={len(subset_idx)} "
        f"train_donors={pd.Index(_donors_for_indices(meta, subset_idx)).nunique()} "
        f"records train={len(train_records)} val={len(val_records)} test={len(test_records)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossTissueCML(args.genes, int(tissue.max()) + 1, args.emb_dim).to(device)
    train_args = SimpleNamespace(
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        k=args.k,
    )
    val_cml_mae = train_cml_model(model, train_records, val_records, expr, meta, tissue_means, train_args, device)

    train_ds = RecordDataset(train_records, expr, meta, tissue_means)
    train_z, train_expr = encode_target_bank(model, train_ds, args.batch, device)

    train_arrays = records_to_arrays(train_records, expr, meta, tissue_means)
    _, train_cml = predict_training_retrieval(
        model,
        train_z,
        train_expr,
        train_ds,
        train_records,
        args.batch,
        args.k,
        device,
    )
    val_ds = RecordDataset(val_records, expr, meta, tissue_means)
    val_true, val_cml = predict_retrieval(model, train_z, train_expr, val_ds, args.batch, args.k, device)
    val_arrays = records_to_arrays(val_records, expr, meta, tissue_means)
    test_ds = RecordDataset(test_records, expr, meta, tissue_means)
    test_true, test_cml = predict_retrieval(model, train_z, train_expr, test_ds, args.batch, args.k, device)
    test_arrays = records_to_arrays(test_records, expr, meta, tissue_means)

    gate, val_gate_mae = train_gate(
        _expert_stack(train_arrays, train_cml),
        train_arrays["true"],
        train_arrays["target_tissue"],
        train_arrays["known_counts"],
        _expert_stack(val_arrays, val_cml),
        val_true,
        val_arrays["target_tissue"],
        val_arrays["known_counts"],
        int(tissue.max()) + 1,
        args.gate_epochs,
        args.gate_batch,
        args.lr,
        device,
    )
    test_gate, test_weights = predict_gate(
        gate,
        _expert_stack(test_arrays, test_cml),
        test_arrays["target_tissue"],
        test_arrays["known_counts"],
        args.gate_batch,
        device,
    )

    train_donor_count = int(pd.Index(_donors_for_indices(meta, subset_idx)).nunique())
    extra = {
        "fraction": float(fraction),
        "fraction_tag": _fraction_tag(fraction),
        "train_samples": int(len(subset_idx)),
        "train_donors": train_donor_count,
        "train_records": int(len(train_records)),
        "val_cml_mae": float(val_cml_mae),
        "val_gate_mae": float(val_gate_mae),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "evaluation_protocol": "hide_one",
        "test_records": int(len(test_records)),
        "test_record_hash": test_record_hash,
        "mixer_epochs": int(args.gate_epochs),
        "mixer_batch": int(args.gate_batch),
    }
    rows = [
        _row_with_residual("donor_level_average_residual", test_arrays["true"], test_arrays["donor_level_average"], test_arrays, tissue_means, extra),
        _row_with_residual("cml_retrieval", test_true, test_cml, test_arrays, tissue_means, extra),
        _row_with_residual("two_expert_cml_gate", test_arrays["true"], test_gate, test_arrays, tissue_means, extra),
    ]
    metrics = pd.DataFrame(rows).sort_values("mae")
    tag = _fraction_tag(fraction)
    metrics.to_csv(out_dir / f"{tag}_overall_metrics.csv", index=False)
    pd.DataFrame(
        {
            "fraction": float(fraction),
            "donor_level_average_weight": test_weights[:, 0],
            "retrieval_weight": test_weights[:, 1],
            "target_tissue": test_arrays["target_tissue"],
            "known_counts": test_arrays["known_counts"],
        }
    ).to_csv(out_dir / f"{tag}_gate_weights.csv", index=False)
    print(metrics[["fraction_tag", "method", "mae", "residual_sample_pearson_mean"]].to_string(index=False))
    return metrics


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_gene_set(args.genes)
    split = split_donors(data.meta, seed=args.seed)
    all_metrics = []
    for fraction in parse_fractions(args.fractions):
        all_metrics.append(run_fraction(data, split, fraction, args, out_dir))
    summary = pd.concat(all_metrics, ignore_index=True)
    summary.sort_values(["fraction", "mae"]).to_csv(
        out_dir / "hide_one_training_size_scaling_overall.csv", index=False
    )
    gate_summary = (
        summary[summary["method"] == "two_expert_cml_gate"]
        .sort_values("fraction")
        .reset_index(drop=True)
    )
    gate_summary.to_csv(out_dir / "hide_one_training_size_scaling_dorian_curve.csv", index=False)
    test_hashes = sorted(summary["test_record_hash"].dropna().astype(str).unique().tolist())
    test_counts = sorted(summary["test_records"].dropna().astype(int).unique().tolist())
    if int(args.max_records) == 0 and (len(test_hashes) != 1 or test_counts != [3922]):
        raise RuntimeError(
            f"Formal test identity changed across fractions: counts={test_counts}, hashes={test_hashes}"
        )
    manifest = {
        "experiment": "hide_one_training_size_scaling",
        "evaluation_protocol": "hide_one",
        "genes": int(args.genes),
        "seed": int(args.seed),
        "fractions": parse_fractions(args.fractions),
        "encoder_epochs": int(args.epochs),
        "encoder_batch": int(args.batch),
        "embedding_dim": int(args.emb_dim),
        "learning_rate": float(args.lr),
        "top_k": int(args.k),
        "mixer_epochs": int(args.gate_epochs),
        "mixer_batch": int(args.gate_batch),
        "test_record_count": test_counts[0] if len(test_counts) == 1 else test_counts,
        "ordered_test_record_sha256": test_hashes[0] if len(test_hashes) == 1 else test_hashes,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print("wrote", out_dir / "hide_one_training_size_scaling_overall.csv")
    return summary


if __name__ == "__main__":
    run(parse_args())
