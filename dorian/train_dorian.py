"""Train a donor-conditioned DORIAN memory-retrieval model.

The manuscript-facing main line uses hide-one training/evaluation records.
This module also retains historical mixed-source training modes for provenance,
but output metadata now records the selected training protocol explicitly so
old random-known terminology does not leak into the main hide-one analysis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dorian.data import load_gene_set
from dorian.metrics import overall_metrics, per_tissue_metrics
from dorian.memory_retrieval_expert import MemoryRetrievalEncoder, RecordDataset
from dorian.adaptive_mixer import predict_with_adaptive_mixer, train_adaptive_mixer
from dorian.records import (
    build_hide_one_records,
    build_random_feasible_known_records,
    donor_to_indices,
)
from dorian.donor_split import split_donors
from dorian.training_utils import (
    encode_target_bank,
    limit_records,
    predict_retrieval,
    record_target_indices,
    records_to_arrays,
    tissue_mean_matrix,
    train_memory_encoder,
    write_metrics,
    write_record_qc,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "rich_dictionary"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mixer_epochs", dest="gate_epochs", metavar="MIXER_EPOCHS", type=int, default=8)
    p.add_argument("--gate_epochs", dest="gate_epochs", type=int, help=argparse.SUPPRESS)
    p.add_argument("--mixer_batch", dest="gate_batch", metavar="MIXER_BATCH", type=int, default=4096)
    p.add_argument("--gate_batch", dest="gate_batch", type=int, help=argparse.SUPPRESS)
    p.add_argument("--train_source", choices=["combined", "hide_one_only", "random_m_only"], default="combined")
    p.add_argument("--max_records", type=int, default=0, help="Smoke/debug limit per split; 0 means full.")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


def protocol_label(train_source):
    labels = {
        "combined": "mixed_hide_one_and_feasible_subset_training",
        "hide_one_only": "hide_one_training",
        "random_m_only": "feasible_subset_training",
    }
    return labels.get(str(train_source), str(train_source))


def _mark_source(records, source):
    out = []
    for draw_id, record in enumerate(records):
        row = dict(record)
        row.setdefault("source", source)
        row.setdefault("draw_id", draw_id)
        out.append(row)
    return out


def build_rich_protocol_records(
    meta, train_idx, val_idx, test_idx, train_map, val_map, test_map, m, seed, train_source="combined"
):
    """Return train/validation/test records for the rich dictionary protocol."""
    train_hide = _mark_source(build_hide_one_records(meta, train_idx, train_map), "hide_one")
    train_random = build_random_feasible_known_records(meta, train_idx, train_map, m=m, seed=seed, task_name="random_known_m")
    val_random = build_random_feasible_known_records(meta, val_idx, val_map, m=m, seed=seed + 1, task_name="random_known_m")
    test_random = build_random_feasible_known_records(meta, test_idx, test_map, m=m, seed=seed + 2, task_name="random_known_m")
    if train_source == "combined":
        train_records = train_hide + train_random
    elif train_source == "hide_one_only":
        train_records = train_hide
    elif train_source == "random_m_only":
        train_records = train_random
    else:
        raise ValueError(f"Unsupported train_source: {train_source}")
    return train_records, val_random, test_random


@torch.no_grad()
def predict_retrieval_with_neighbors(model, train_z, train_expr, ds, batch, k, device):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    train_z = train_z / (np.linalg.norm(train_z, axis=1, keepdims=True) + 1e-12)
    preds = []
    trues = []
    top_indices = []
    model.eval()
    for known_expr, known_t, known_mask, target_t, _, target_expr, _ in loader:
        zq = model.encode_query(
            known_expr.to(device),
            known_t.to(device),
            known_mask.to(device),
            target_t.to(device),
        ).cpu().numpy()
        zq = zq / (np.linalg.norm(zq, axis=1, keepdims=True) + 1e-12)
        sim = zq @ train_z.T
        kk = min(int(k), train_z.shape[0])
        top = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        top_sim = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_sim, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)
        w = np.exp(top_sim - top_sim.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        preds.append(np.einsum("bk,bkg->bg", w, train_expr[top]).astype(np.float32))
        trues.append(target_expr.numpy())
        top_indices.append(top.astype(np.int64))
    return (
        np.vstack(trues).astype(np.float32),
        np.vstack(preds).astype(np.float32),
        np.vstack(top_indices).astype(np.int64),
    )


def records_to_frame(records, split_name, gene_set, seed, m):
    rows = []
    for i, record in enumerate(records):
        known = [int(x) for x in record["known_indices"]]
        tissues = [int(x) for x in record["known_tissues"]]
        rows.append(
            {
                "record_id": f"{gene_set}:{split_name}:{record.get('source', record['task'])}:{i}",
                "gene_set": int(gene_set),
                "seed": int(seed),
                "m": int(m),
                "split": split_name,
                "source": record.get("source", record["task"]),
                "task": record["task"],
                "draw_id": int(record.get("draw_id", -1)),
                "donor": record["donor"],
                "target_index": int(record["idx"]),
                "target_tissue": int(record["target_tissue"]),
                "known_indices": ";".join(str(x) for x in known),
                "known_tissues": ";".join(str(x) for x in tissues),
                "actual_known_count": int(record["n_known_tissues"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_neighbor_source_composition(query_records, train_records, top_indices):
    def source_of(record):
        return str(record.get("source") or record.get("task") or "unknown")

    source_names = sorted({source_of(record) for record in train_records})
    record_rows = []
    source_rows = []
    for query_i, (query, neighbors) in enumerate(zip(query_records, top_indices)):
        neighbor_records = [train_records[int(i)] for i in neighbors]
        neighbor_sources = [source_of(record) for record in neighbor_records]
        neighbor_counts = np.array([int(record["n_known_tissues"]) for record in neighbor_records], dtype=np.float32)
        row = {
            "query_row": int(query_i),
            "target_tissue": int(query["target_tissue"]),
            "query_known_count": int(query["n_known_tissues"]),
            "mean_neighbor_known_count": float(neighbor_counts.mean()),
            "mean_abs_known_count_delta": float(np.abs(neighbor_counts - int(query["n_known_tissues"])).mean()),
        }
        for source in source_names:
            row[f"{source}_neighbor_fraction"] = float(np.mean([src == source for src in neighbor_sources]))
            source_rows.append(
                {
                    "query_row": int(query_i),
                    "neighbor_source": source,
                    "fraction": row[f"{source}_neighbor_fraction"],
                }
            )
        record_rows.append(row)
    record_df = pd.DataFrame(record_rows)
    aggregate_df = (
        pd.DataFrame(source_rows)
        .groupby("neighbor_source", as_index=False)["fraction"]
        .agg(["mean", "median"])
        .reset_index()
    )
    if "level_0" in aggregate_df.columns:
        aggregate_df = aggregate_df.drop(columns=["level_0"])
    return record_df, aggregate_df


def write_source_metrics(out_dir, tag, arrays, method_preds, records, tissue_names, extra):
    sources = np.array([str(record.get("source", record["task"])) for record in records])
    rows = []
    for source in sorted(np.unique(sources)):
        mask = sources == source
        source_extra = dict(extra)
        source_extra["query_source"] = source
        for method, pred in method_preds.items():
            rows.append(overall_metrics(method, arrays["true"][mask], pred[mask], extra=source_extra))
    pd.DataFrame(rows).sort_values(["query_source", "mae"]).to_csv(out_dir / f"{tag}_by_query_source_metrics.csv", index=False)


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_gene_set(args.genes)
    expr = data.expr
    meta = data.meta
    tissue = meta["tissue_idx"].to_numpy()
    split = split_donors(meta, seed=args.seed)
    maps = {
        "train": donor_to_indices(meta, split.train_idx),
        "val": donor_to_indices(meta, split.val_idx),
        "test": donor_to_indices(meta, split.test_idx),
    }
    tissue_means = tissue_mean_matrix(expr, tissue, split.train_idx)
    train_records, val_records, test_records = build_rich_protocol_records(
        meta,
        split.train_idx,
        split.val_idx,
        split.test_idx,
        maps["train"],
        maps["val"],
        maps["test"],
        args.m,
        args.seed,
        args.train_source,
    )
    train_records = limit_records(train_records, args.max_records, args.seed)
    val_records = limit_records(val_records, args.max_records, args.seed + 1)
    test_records = limit_records(test_records, args.max_records, args.seed + 2)
    print(f"records train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    print(
        "train sources "
        + ", ".join(
            f"{source}={count}" for source, count in pd.Series([r["source"] for r in train_records]).value_counts().items()
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryRetrievalEncoder(args.genes, int(tissue.max()) + 1, args.emb_dim).to(device)
    val_memory_mae = train_memory_encoder(model, train_records, val_records, expr, meta, tissue_means, args, device)
    train_ds = RecordDataset(train_records, expr, meta, tissue_means)
    train_z, train_expr = encode_target_bank(model, train_ds, args.batch, device)

    train_arrays = records_to_arrays(train_records, expr, meta, tissue_means)
    _, train_memory = predict_retrieval(
        model,
        train_z,
        train_expr,
        RecordDataset(train_records, expr, meta, tissue_means),
        args.batch,
        args.k,
        device,
        exclude_self=True,
        train_target_indices=record_target_indices(train_records),
    )
    val_arrays = records_to_arrays(val_records, expr, meta, tissue_means)
    _, val_memory, val_top = predict_retrieval_with_neighbors(
        model, train_z, train_expr, RecordDataset(val_records, expr, meta, tissue_means), args.batch, args.k, device
    )
    test_arrays = records_to_arrays(test_records, expr, meta, tissue_means)
    _, test_memory, test_top = predict_retrieval_with_neighbors(
        model, train_z, train_expr, RecordDataset(test_records, expr, meta, tissue_means), args.batch, args.k, device
    )

    train_experts = np.stack([train_arrays["same_donor"], train_memory], axis=1)
    val_experts = np.stack([val_arrays["same_donor"], val_memory], axis=1)
    test_experts = np.stack([test_arrays["same_donor"], test_memory], axis=1)
    gate, val_mixer_mae = train_adaptive_mixer(
        train_experts,
        train_arrays["true"],
        train_arrays["target_tissue"],
        train_arrays["known_counts"],
        val_experts,
        val_arrays["true"],
        val_arrays["target_tissue"],
        val_arrays["known_counts"],
        int(tissue.max()) + 1,
        args.gate_epochs,
        args.gate_batch,
        args.lr,
        device,
    )
    test_dorian, test_weights = predict_with_adaptive_mixer(
        gate, test_experts, test_arrays["target_tissue"], test_arrays["known_counts"], args.gate_batch, device
    )

    tissue_names = {int(v): k for k, v in data.tissue_encoder.items()} if isinstance(data.tissue_encoder, dict) else {}
    source_tag = "rich_dictionary" if args.train_source == "combined" else f"rich_dictionary_{args.train_source}"
    tag = f"{source_tag}_m{args.m}_{args.genes}genes_seed{args.seed}"
    extra = {
        "gene_set": args.genes,
        "seed": args.seed,
        "task": "dorian_memory_retrieval",
        "m": args.m,
        "train_source": args.train_source,
        "training_protocol": protocol_label(args.train_source),
        "evaluation_protocol": "feasible_known_subset_records",
        "known_count": "record_specific_available_source_tissues",
        "val_memory_mae": val_memory_mae,
        "val_mixer_mae": val_mixer_mae,
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "n_test_records": len(test_records),
        "n_train_hide_one_records": sum(r["source"] == "hide_one" for r in train_records),
        "n_train_random_m_records": sum(r["source"] == f"random_known_m{args.m}" for r in train_records),
    }
    method_preds = {
        "dorian": test_dorian,
        "same_donor_residual": test_arrays["same_donor"],
        "memory_retrieval": test_memory,
        "target_tissue_mean": test_arrays["base"],
    }
    write_metrics(out_dir, tag, test_arrays, method_preds, tissue_names, extra)
    write_source_metrics(out_dir, tag, test_arrays, method_preds, test_records, tissue_names, extra)
    write_record_qc(out_dir, tag, test_records, tissue_names)

    records_to_frame(train_records, "train", args.genes, args.seed, args.m).to_csv(out_dir / f"{tag}_train_records.csv", index=False)
    records_to_frame(val_records, "val", args.genes, args.seed, args.m).to_csv(out_dir / f"{tag}_val_records.csv", index=False)
    records_to_frame(test_records, "test", args.genes, args.seed, args.m).to_csv(out_dir / f"{tag}_test_records.csv", index=False)
    pd.DataFrame(test_weights, columns=["same_donor_weight", "memory_weight"]).to_csv(
        out_dir / f"{tag}_mixer_weights.csv", index=False
    )
    record_neighbor_df, aggregate_neighbor_df = summarize_neighbor_source_composition(test_records, train_records, test_top)
    record_neighbor_df.to_csv(out_dir / f"{tag}_neighbor_source_composition_by_record.csv", index=False)
    aggregate_neighbor_df.to_csv(out_dir / f"{tag}_neighbor_source_composition_summary.csv", index=False)
    np.savez_compressed(
        out_dir / f"{tag}_predictions.npz",
        true=test_arrays["true"],
        dorian=test_dorian,
        same_donor_residual=test_arrays["same_donor"],
        memory_retrieval=test_memory,
        target_tissue_mean=test_arrays["base"],
        target_tissue=test_arrays["target_tissue"],
        known_counts=test_arrays["known_counts"],
        top_train_indices=test_top,
        val_top_train_indices=val_top,
    )
    torch.save(
        # Historical key names are retained only for checkpoint compatibility.
        {"cml_state": model.state_dict(), "gate_state": gate.state_dict(), "args": vars(args), "tag": tag},
        out_dir / f"{tag}.pt",
    )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()


