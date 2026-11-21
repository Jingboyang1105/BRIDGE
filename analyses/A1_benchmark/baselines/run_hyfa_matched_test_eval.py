"""Evaluate a trained official HYFA checkpoint on CML matched test records."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from CML_Gene_Imputation.src.benchmark_adapter import save_prediction_bundle, write_evaluation_tables
from CML_Gene_Imputation.src.data import load_gene_set
from CML_Gene_Imputation.src.run_hyfa_record_baseline import (
    RecordHypergraphDataset,
    build_adata,
    collate_records,
    evaluate_loader,
    load_hyfa_official_modules,
    make_config,
)


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RECORD_COLUMNS = {
    "record_id",
    "donor",
    "target_index",
    "target_tissue",
    "known_indices",
    "known_tissues",
    "actual_known_count",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.00045568167228053424)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "results"
            / "hyfa_official_split_20k_seed42_full"
            / "hyfa_official_split_20000genes_seed42.pt"
        ),
    )
    p.add_argument(
        "--test_record_csv",
        default=str(
            ROOT
            / "results"
            / "rich_dictionary_20k_m5_hide_one_only_full"
            / "rich_dictionary_hide_one_only_m5_20000genes_seed42_test_records.csv"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=str(ROOT / "results" / "hyfa_official_split_20k_seed42_matched_cml_random_m5_test"),
    )
    p.add_argument("--method_name", default="hyfa_official_matched_cml_random_m5")
    p.add_argument("--task_name", default="matched_cml_random_m5_test")
    p.add_argument("--output_prefix", default="")
    return p.parse_args()


def load_matched_records(path, max_records=0, seed=42):
    records = pd.read_csv(path)
    missing = sorted(REQUIRED_RECORD_COLUMNS.difference(records.columns))
    if missing:
        raise ValueError(f"matched record CSV is missing required columns: {missing}")
    records = records.reset_index(drop=True)
    if max_records and len(records) > int(max_records):
        records = records.head(int(max_records)).reset_index(drop=True)
    return records


def _checkpoint_state(checkpoint):
    loaded = torch.load(checkpoint, map_location="cpu")
    if isinstance(loaded, dict) and "state" in loaded:
        return loaded["state"], loaded
    return loaded, {}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_gene_set(args.genes)
    adata = build_adata(data)
    records = load_matched_records(args.test_record_csv, args.max_records, args.seed)
    output_prefix = args.output_prefix or f"hyfa_matched_cml_random_m5_{args.genes}genes_seed{args.seed}"
    records.to_csv(out_dir / f"{output_prefix}_test_records.csv", index=False)

    config = make_config(args, adata)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _, _, model_cls, _, _ = load_hyfa_official_modules()
    model = model_cls(config).to(device)
    state, checkpoint_meta = _checkpoint_state(args.checkpoint)
    model.load_state_dict(state)

    loader = DataLoader(
        RecordHypergraphDataset(adata, records),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_records,
    )
    y_true, y_pred = evaluate_loader(model, loader, device)
    gene_ids = np.array([f"gene_{i}" for i in range(y_true.shape[1])], dtype=object)
    bundle = out_dir / f"{output_prefix}_predictions.npz"
    save_prediction_bundle(
        bundle,
        record_ids=records["record_id"].astype(str).to_numpy(),
        gene_ids=gene_ids,
        y_true=y_true,
        predictions={args.method_name: y_pred},
    )
    extra = {
        "gene_set": args.genes,
        "seed": args.seed,
        "task": args.task_name,
        "baseline": "HYFA_official_split_checkpoint",
        "checkpoint": str(args.checkpoint),
        "n_test_records": len(records),
        "best_val_mae": checkpoint_meta.get("best_val_mae", ""),
    }
    overall, _, _ = write_evaluation_tables(
        bundle,
        records,
        out_dir,
        output_prefix,
        extra=extra,
    )
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
