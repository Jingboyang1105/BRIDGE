"""Evaluate an already-trained HYFA checkpoint on an arbitrary record CSV."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from CML_Gene_Imputation.src.benchmark_adapter import save_prediction_bundle, write_evaluation_tables
from CML_Gene_Imputation.src.data import load_gene_set
from CML_Gene_Imputation.src.run_hyfa_record_baseline import (
    Config,
    RecordHypergraphDataset,
    build_adata,
    collate_records,
    evaluate_loader,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
HYFA_ROOT = REPO_ROOT / "external" / "HYFA"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no_demographics", dest="use_demographics", action="store_false")
    p.set_defaults(use_demographics=True)
    p.add_argument("--records", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pred_key", default="HYFA_accessible")
    p.add_argument("--prefix", default="hyfa_accessible_20000genes_seed42")
    p.add_argument("--out_dir", default=str(ROOT / "accessible" / "results" / "hyfa_accessible_20k_seed42"))
    return p.parse_args()


def load_model(checkpoint_path, device):
    if str(HYFA_ROOT) not in sys.path:
        sys.path.insert(0, str(HYFA_ROOT))
    from src.hnn import HypergraphNeuralNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("HYFA checkpoint must contain a 'config' entry.")
    if isinstance(config, dict):
        config = Config(config)
    model = HypergraphNeuralNet(config).to(device)
    state = checkpoint.get("state", checkpoint.get("model_state_dict", checkpoint.get("state_dict")))
    if state is None:
        raise ValueError("HYFA checkpoint must contain 'state', 'state_dict', or 'model_state_dict'.")
    model.load_state_dict(state)
    return model, checkpoint


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_gene_set(args.genes)
    adata = build_adata(data, use_demographics=args.use_demographics)
    records = pd.read_csv(args.records).reset_index(drop=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    loader = DataLoader(
        RecordHypergraphDataset(adata, records),
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_records,
    )
    y_true, y_pred = evaluate_loader(model, loader, device)
    records.to_csv(out_dir / f"{args.prefix}_test_records.csv", index=False)
    gene_ids = np.array([f"gene_{i}" for i in range(y_true.shape[1])], dtype=object)
    bundle = out_dir / f"{args.prefix}_predictions.npz"
    save_prediction_bundle(
        bundle,
        record_ids=records["record_id"].astype(str).to_numpy(),
        gene_ids=gene_ids,
        y_true=y_true,
        predictions={args.pred_key: y_pred},
    )
    extra = {
        "gene_set": args.genes,
        "seed": args.seed,
        "task": records["task"].iloc[0] if "task" in records and len(records) else "record_evaluation",
        "baseline": args.pred_key,
        "checkpoint": str(args.checkpoint),
        "checkpoint_best_val_mae": checkpoint.get("best_val_mae", np.nan),
    }
    overall, _, _ = write_evaluation_tables(bundle, records, out_dir, args.prefix, extra=extra)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
