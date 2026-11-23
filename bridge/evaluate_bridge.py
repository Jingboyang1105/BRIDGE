"""Evaluate a trained BRIDGE checkpoint on an arbitrary record CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bridge.benchmark_io import save_prediction_bundle, write_evaluation_tables
from bridge.data import load_gene_set
from bridge.sample_matching_expert import SampleMatchingEncoder, RecordDataset
from bridge.adaptive_mixer import AdaptivePredictionMixer, predict_with_adaptive_mixer
from bridge.records import donor_to_indices
from bridge.donor_split import split_donors
from bridge.training_utils import (
    build_task_records,
    encode_target_bank,
    predict_retrieval,
    records_to_arrays,
    tissue_mean_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gate_batch", type=int, default=4096)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--records", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prefix", default="dorian_records_20000genes_seed42")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def checkpoint_args(loaded, cli):
    stored = dict(loaded.get("args", {}))
    stored.setdefault("task", "hide_one")
    stored.setdefault("known_count", 0)
    stored.setdefault("known_counts", "2,4,6,8,12,16,20")
    stored.setdefault("emb_dim", 128)
    stored.setdefault("seed", cli.seed)
    stored.setdefault("genes", cli.genes)
    stored["batch"] = cli.batch
    stored["gate_batch"] = cli.gate_batch
    stored["k"] = cli.k
    return argparse.Namespace(**stored)


def load_records(path):
    frame = pd.read_csv(path).reset_index(drop=True)
    records = []
    for row in frame.to_dict(orient="records"):
        rec = dict(row)
        if "idx" not in rec and "target_index" in rec:
            rec["idx"] = int(rec["target_index"])
        if isinstance(rec.get("known_indices"), str):
            rec["known_indices"] = [int(x) for x in rec["known_indices"].split(";") if x != ""]
        if isinstance(rec.get("known_tissues"), str):
            rec["known_tissues"] = [int(x) for x in rec["known_tissues"].split(";") if x != ""]
        records.append(rec)
    return frame, records


def main():
    cli = parse_args()
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    args = checkpoint_args(checkpoint, cli)

    data = load_gene_set(cli.genes)
    expr = data.expr
    meta = data.meta
    tissue = meta["tissue_idx"].to_numpy()
    split = split_donors(meta, seed=cli.seed)
    train_map = donor_to_indices(meta, split.train_idx)
    tissue_means = tissue_mean_matrix(expr, tissue, split.train_idx)

    train_records = build_task_records(meta, split.train_idx, train_map, args, 0)
    eval_frame, eval_records = load_records(cli.records)
    eval_frame.to_csv(out_dir / f"{cli.prefix}_test_records.csv", index=False)

    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    model = SampleMatchingEncoder(cli.genes, int(tissue.max()) + 1, args.emb_dim).to(device)
    # Historical checkpoints store the memory encoder under the key ``cml_state``.
    model.load_state_dict(checkpoint["cml_state"])
    gate = AdaptivePredictionMixer(int(tissue.max()) + 1).to(device)
    gate.load_state_dict(checkpoint["gate_state"])

    train_ds = RecordDataset(train_records, expr, meta, tissue_means)
    train_z, train_expr = encode_target_bank(model, train_ds, cli.batch, device)
    eval_ds = RecordDataset(eval_records, expr, meta, tissue_means)
    eval_arrays = records_to_arrays(eval_records, expr, meta, tissue_means)
    _, eval_memory = predict_retrieval(model, train_z, train_expr, eval_ds, cli.batch, cli.k, device)
    eval_experts = np.stack([eval_arrays["donor_level_average"], eval_memory], axis=1)
    eval_bridge, eval_weights = predict_with_adaptive_mixer(
        gate,
        eval_experts,
        eval_arrays["target_tissue"],
        eval_arrays["known_counts"],
        cli.gate_batch,
        device,
    )

    bundle = out_dir / f"{cli.prefix}_predictions.npz"
    save_prediction_bundle(
        bundle,
        record_ids=eval_frame["record_id"].astype(str).to_numpy(),
        gene_ids=np.array([f"gene_{i}" for i in range(eval_arrays["true"].shape[1])], dtype=object),
        y_true=eval_arrays["true"],
        predictions={
            "bridge": eval_bridge,
            "donor_level_average_residual": eval_arrays["donor_level_average"],
            "sample_matching": eval_memory,
            "target_tissue_mean": eval_arrays["base"],
        },
    )
    pd.DataFrame(eval_weights, columns=["donor_level_average_weight", "sample_matching_weight"]).to_csv(
        out_dir / f"{cli.prefix}_mixer_weights.csv", index=False
    )
    extra = {
        "gene_set": cli.genes,
        "seed": cli.seed,
        "task": eval_frame["task"].iloc[0] if "task" in eval_frame and len(eval_frame) else "record_evaluation",
        "baseline": "BRIDGE_checkpoint",
        "checkpoint": str(cli.checkpoint),
        "n_train_records": len(train_records),
        "n_test_records": len(eval_records),
    }
    overall, _, _ = write_evaluation_tables(bundle, eval_frame, out_dir, cli.prefix, extra=extra)
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


