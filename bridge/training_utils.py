"""Train strict-protocol hide-one-tissue BRIDGE two-expert model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bridge.data import load_gene_set
from bridge.metrics import overall_metrics, per_tissue_metrics
from bridge.sample_matching_expert import SampleMatchingEncoder, RecordDataset, contrastive_loss
from bridge.adaptive_mixer import predict_with_adaptive_mixer, train_adaptive_mixer
from bridge.records import (
    build_hide_one_records,
    build_known_count_records,
    build_random_known_count_records,
    donor_to_indices,
)
from bridge.donor_split import split_donors


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "hide_one"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--gate_epochs", type=int, default=6)
    p.add_argument("--gate_batch", type=int, default=4096)
    p.add_argument("--max_records", type=int, default=0, help="Smoke/debug limit per split; 0 means full.")
    p.add_argument("--task", choices=["hide_one", "known_count", "random_known_count"], default="hide_one")
    p.add_argument("--known_count", type=int, default=0)
    p.add_argument("--known_counts", default="2,4,6,8,12,16,20")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    return p.parse_args()


def build_task_records(meta, split_idx, donor_map, args, seed_offset):
    if args.task == "hide_one":
        return build_hide_one_records(meta, split_idx, donor_map)
    if args.task == "random_known_count":
        grid = [int(x.strip()) for x in str(args.known_counts).split(",") if x.strip()]
        return build_random_known_count_records(meta, split_idx, donor_map, grid, args.seed + seed_offset)
    if args.known_count <= 0:
        raise ValueError("--known_count is required when --task known_count")
    return build_known_count_records(meta, split_idx, donor_map, args.known_count, args.seed + seed_offset)


def tissue_mean_matrix(expr, tissue, train_idx):
    n_tissues = int(tissue.max()) + 1
    global_mean = expr[train_idx].mean(axis=0).astype(np.float32)
    means = np.repeat(global_mean[None, :], n_tissues, axis=0)
    for t in range(n_tissues):
        mask = tissue[train_idx] == t
        if mask.any():
            means[t] = expr[train_idx][mask].mean(axis=0)
    return means.astype(np.float32)


def limit_records(records, max_records, seed):
    if not max_records or len(records) <= max_records:
        return records
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(len(records)), size=max_records, replace=False))
    return [records[int(i)] for i in idx]


def records_to_arrays(records, expr, meta, tissue_means):
    tissue = meta["tissue_idx"].to_numpy()
    true = []
    base = []
    same = []
    target_tissue = []
    counts = []
    for r in records:
        idx = int(r["idx"])
        target = int(tissue[idx])
        source_resids = [expr[int(j)] - tissue_means[int(tissue[int(j)])] for j in r["known_indices"]]
        mean_resid = np.vstack(source_resids).mean(axis=0).astype(np.float32)
        target_base = tissue_means[target].astype(np.float32)
        true.append(expr[idx].astype(np.float32))
        base.append(target_base)
        same.append((target_base + mean_resid).astype(np.float32))
        target_tissue.append(target)
        counts.append(len(r["known_indices"]))
    return {
        "true": np.vstack(true).astype(np.float32),
        "base": np.vstack(base).astype(np.float32),
        "donor_level_average": np.vstack(same).astype(np.float32),
        "target_tissue": np.array(target_tissue, dtype=np.int64),
        "known_counts": np.array(counts, dtype=np.float32),
    }


def train_memory_encoder(model, train_records, val_records, expr, meta, tissue_means, args, device):
    train_ds = RecordDataset(train_records, expr, meta, tissue_means)
    loss_ds = RecordDataset(train_records, expr, meta, tissue_means, return_target_index=True)
    val_ds = RecordDataset(val_records, expr, meta, tissue_means)
    loader = DataLoader(loss_ds, batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None
    best_val = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for known_expr, known_t, known_mask, target_t, target_resid, _, _, target_idx in loader:
            zq, zt = model(
                known_expr.to(device),
                known_t.to(device),
                known_mask.to(device),
                target_t.to(device),
                target_resid.to(device),
            )
            loss = contrastive_loss(zq, zt, target_indices=target_idx.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(target_t)
            n += len(target_t)
        train_z, train_expr = encode_target_bank(model, train_ds, args.batch, device)
        val_true, val_pred = predict_retrieval(model, train_z, train_expr, val_ds, args.batch, args.k, device)
        val_mae = float(np.abs(val_pred - val_true).mean())
        print(f"epoch={epoch}/{args.epochs} memory_loss={total / max(n, 1):.4f} val_memory_mae={val_mae:.4f}")
        if best_val is None or val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def encode_target_bank(model, ds, batch, device):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    zs = []
    exprs = []
    model.eval()
    for _, _, _, _, target_resid, target_expr, _ in loader:
        z = model.encode_expr(target_resid.to(device))
        zs.append(z.cpu().numpy())
        exprs.append(target_expr.numpy())
    return np.vstack(zs).astype(np.float32), np.vstack(exprs).astype(np.float32)


def mask_same_target_similarities(similarities, query_target_indices, memory_target_indices):
    """Exclude every memory record backed by the same measured target row."""

    result = np.asarray(similarities).copy()
    query = np.asarray(query_target_indices, dtype=np.int64)
    memory = np.asarray(memory_target_indices, dtype=np.int64)
    if result.shape != (len(query), len(memory)):
        raise ValueError("similarity dimensions must match query and memory target indices")
    result[query[:, None] == memory[None, :]] = -np.inf
    return result


def record_target_indices(records):
    return np.asarray([int(record["idx"]) for record in records], dtype=np.int64)


@torch.no_grad()
def predict_retrieval(
    model,
    train_z,
    train_expr,
    ds,
    batch,
    k,
    device,
    exclude_self=False,
    train_target_indices=None,
):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    train_z = train_z / (np.linalg.norm(train_z, axis=1, keepdims=True) + 1e-12)
    preds = []
    trues = []
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
        if exclude_self:
            offset = len(preds) * batch
            if train_target_indices is None:
                raise ValueError("train_target_indices are required when excluding duplicate targets")
            query_target_indices = record_target_indices(ds.records[offset : offset + sim.shape[0]])
            sim = mask_same_target_similarities(sim, query_target_indices, train_target_indices)
        available = np.isfinite(sim).sum(axis=1)
        if np.any(available == 0):
            raise ValueError("no retrieval candidate remains after target-duplicate exclusion")
        kk = min(int(k), int(available.min()))
        top = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        top_sim = np.take_along_axis(sim, top, axis=1)
        w = np.exp(top_sim - top_sim.max(axis=1, keepdims=True))
        w = w / w.sum(axis=1, keepdims=True)
        preds.append(np.einsum("bk,bkg->bg", w, train_expr[top]).astype(np.float32))
        trues.append(target_expr.numpy())
    return np.vstack(trues).astype(np.float32), np.vstack(preds).astype(np.float32)


def write_metrics(out_dir, tag, arrays, method_preds, tissue_names, extra):
    overall = []
    per_tissue = []
    for method, pred in method_preds.items():
        overall.append(overall_metrics(method, arrays["true"], pred, extra=extra))
        per = per_tissue_metrics(method, arrays["true"], pred, arrays["target_tissue"], tissue_names, extra=extra)
        per_tissue.append(per)
    overall_df = pd.DataFrame(overall).sort_values("mae")
    per_df = pd.concat(per_tissue, ignore_index=True).sort_values(["method", "target_tissue_name"])
    overall_df.to_csv(out_dir / f"{tag}_overall_metrics.csv", index=False)
    per_df.to_csv(out_dir / f"{tag}_per_tissue_metrics.csv", index=False)
    known_rows = []
    for count in np.sort(np.unique(arrays["known_counts"])):
        mask = arrays["known_counts"] == count
        count_extra = dict(extra)
        count_extra["actual_known_count"] = int(count)
        for method, pred in method_preds.items():
            known_rows.append(overall_metrics(method, arrays["true"][mask], pred[mask], extra=count_extra))
    pd.DataFrame(known_rows).sort_values(["actual_known_count", "mae"]).to_csv(
        out_dir / f"{tag}_by_known_count_metrics.csv", index=False
    )
    print(overall_df.to_string(index=False))


def write_record_qc(out_dir, tag, records, tissue_names):
    target_rows = []
    known_rows = []
    for r in records:
        target_rows.append(
            {
                "target_tissue": int(r["target_tissue"]),
                "target_tissue_name": tissue_names.get(int(r["target_tissue"]), str(int(r["target_tissue"]))),
                "actual_known_count": int(r["n_known_tissues"]),
            }
        )
        for tissue in r["known_tissues"]:
            known_rows.append(
                {
                    "known_tissue": int(tissue),
                    "known_tissue_name": tissue_names.get(int(tissue), str(int(tissue))),
                    "actual_known_count": int(r["n_known_tissues"]),
                }
            )
    target = pd.DataFrame(target_rows)
    known = pd.DataFrame(known_rows)
    if not target.empty:
        target.groupby(["target_tissue", "target_tissue_name", "actual_known_count"]).size().reset_index(
            name="target_records"
        ).to_csv(out_dir / f"{tag}_target_tissue_record_counts.csv", index=False)
    if not known.empty:
        known.groupby(["known_tissue", "known_tissue_name"]).size().reset_index(name="times_used_as_known").sort_values(
            "times_used_as_known", ascending=False
        ).to_csv(out_dir / f"{tag}_known_tissue_frequency.csv", index=False)
        known.groupby(["known_tissue", "known_tissue_name", "actual_known_count"]).size().reset_index(
            name="times_used_as_known"
        ).sort_values(["actual_known_count", "times_used_as_known"], ascending=[True, False]).to_csv(
            out_dir / f"{tag}_known_tissue_frequency_by_count.csv", index=False
        )


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
    train_records = limit_records(build_task_records(meta, split.train_idx, maps["train"], args, 0), args.max_records, args.seed)
    val_records = limit_records(build_task_records(meta, split.val_idx, maps["val"], args, 1), args.max_records, args.seed + 1)
    test_records = limit_records(build_task_records(meta, split.test_idx, maps["test"], args, 2), args.max_records, args.seed + 2)
    print(f"records train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SampleMatchingEncoder(args.genes, int(tissue.max()) + 1, args.emb_dim).to(device)
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
    _, val_memory = predict_retrieval(model, train_z, train_expr, RecordDataset(val_records, expr, meta, tissue_means), args.batch, args.k, device)
    test_arrays = records_to_arrays(test_records, expr, meta, tissue_means)
    _, test_memory = predict_retrieval(model, train_z, train_expr, RecordDataset(test_records, expr, meta, tissue_means), args.batch, args.k, device)

    train_experts = np.stack([train_arrays["donor_level_average"], train_memory], axis=1)
    val_experts = np.stack([val_arrays["donor_level_average"], val_memory], axis=1)
    test_experts = np.stack([test_arrays["donor_level_average"], test_memory], axis=1)
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
    test_bridge, test_weights = predict_with_adaptive_mixer(
        gate, test_experts, test_arrays["target_tissue"], test_arrays["known_counts"], args.gate_batch, device
    )
    tissue_names = {int(v): k for k, v in data.tissue_encoder.items()} if isinstance(data.tissue_encoder, dict) else {}
    if args.task == "hide_one":
        task_label = "hide_one_tissue"
        tag = f"hide_one_{args.genes}genes_seed{args.seed}"
        known_count_label = "all_other"
    elif args.task == "random_known_count":
        task_label = "random_known_count"
        tag = f"random_known_count_{args.genes}genes_seed{args.seed}"
        known_count_label = args.known_counts
    else:
        task_label = f"known_count_{args.known_count}"
        tag = f"{args.task}_{args.known_count}known_{args.genes}genes_seed{args.seed}"
        known_count_label = args.known_count
    extra = {
        "gene_set": args.genes,
        "seed": args.seed,
        "task": task_label,
        "known_count": known_count_label,
        "val_memory_mae": val_memory_mae,
        "val_mixer_mae": val_mixer_mae,
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "n_test_records": len(test_records),
    }
    write_metrics(
        out_dir,
        tag,
        test_arrays,
        {
            "bridge": test_bridge,
            "donor_level_average_residual": test_arrays["donor_level_average"],
            "sample_matching": test_memory,
            "target_tissue_mean": test_arrays["base"],
        },
        tissue_names,
        extra,
    )
    write_record_qc(out_dir, tag, test_records, tissue_names)
    pd.DataFrame(test_weights, columns=["donor_level_average_weight", "memory_weight"]).to_csv(
        out_dir / f"{tag}_mixer_weights.csv", index=False
    )
    # Keep the historical checkpoint keys so previously trained models remain loadable.
    torch.save({"cml_state": model.state_dict(), "gate_state": gate.state_dict(), "args": vars(args)}, out_dir / f"{tag}.pt")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()


