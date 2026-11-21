"""Train HYFA official split baseline on a 20k AnnData table plus donor splits.

This is the closest method baseline to HYFA's own training logic: the model,
forward pass, loss, and donor-level source/target sampling come from
`external/HYFA/src`. CML record dictionaries are not used for training.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from CML_Gene_Imputation.src.benchmark_adapter import save_prediction_bundle, write_evaluation_tables
from CML_Gene_Imputation.src.data import load_gene_set
from CML_Gene_Imputation.src.run_hyfa_record_baseline import build_adata, make_config
from CML_Gene_Imputation.src.splits import donor_id, split_donors


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
HYFA_ROOT = REPO_ROOT / "external" / "HYFA"

HYFA_DATA = None
HYFA_DATASET = None
HYFA_DEFAULT_SAMPLE_FN = None
HYFA_MODEL = None
HYFA_FORWARD = None
HYFA_COMPUTE_LOSS = None


def load_hyfa_official_modules():
    """Load official HYFA classes/functions lazily."""
    global HYFA_DATA, HYFA_DATASET, HYFA_DEFAULT_SAMPLE_FN, HYFA_MODEL, HYFA_FORWARD, HYFA_COMPUTE_LOSS
    if HYFA_DATA is None:
        if str(HYFA_ROOT) not in sys.path:
            sys.path.insert(0, str(HYFA_ROOT))
        from src.data import Data as HyfaData
        from src.dataset import HypergraphDataset, default_sample_fn
        from src.hnn import HypergraphNeuralNet
        from src.train_utils import compute_loss, forward

        HYFA_DATA = HyfaData
        HYFA_DATASET = HypergraphDataset
        HYFA_DEFAULT_SAMPLE_FN = default_sample_fn
        HYFA_MODEL = HypergraphNeuralNet
        HYFA_FORWARD = forward
        HYFA_COMPUTE_LOSS = compute_loss
    return HYFA_DATA, HYFA_DATASET, HYFA_DEFAULT_SAMPLE_FN, HYFA_MODEL, HYFA_FORWARD, HYFA_COMPUTE_LOSS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--genes", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.00045568167228053424)
    p.add_argument("--max_train_donors", type=int, default=0)
    p.add_argument("--max_val_donors", type=int, default=0)
    p.add_argument("--max_test_donors", type=int, default=0)
    p.add_argument("--out_dir", default=str(ROOT / "results" / "hyfa_official_split_20k_seed42"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no_demographics", dest="use_demographics", action="store_false")
    p.set_defaults(use_demographics=True)
    return p.parse_args()


def donor_ids_from_indices(meta, indices):
    donors = sorted({donor_id(meta.index[int(idx)]) for idx in indices})
    return donors


def _limit_donors(donors, n, seed):
    donors = list(donors)
    if not n or len(donors) <= int(n):
        return donors
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(np.array(donors, dtype=object), size=int(n), replace=False).tolist())
    return keep


def collate_official(items):
    data_cls, _, _, _, _, _ = load_hyfa_official_modules()
    return data_cls.from_datalist(items)


def make_official_dataset(adata, donors, static, seed):
    _, dataset_cls, default_sample_fn, _, _, _ = load_hyfa_official_modules()
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        return dataset_cls(
            adata,
            obs_source={"Participant ID": list(donors)},
            obs_target={"Participant ID": list(donors)},
            donor_key="Participant ID",
            sample_fn=default_sample_fn,
            static=bool(static),
            disjoint=True,
            verbose=False,
            dtype=torch.float32,
        )
    finally:
        np.random.set_state(state)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def batch_records_from_hyfa_data(batch, split_name, prefix, start):
    source_donors = np.asarray(batch.source_dynamic["Participant ID"], dtype=object)
    target_donors = np.asarray(batch.target_dynamic["Participant ID"], dtype=object)
    target_tissue = _to_numpy(batch.target["Tissue"]).astype(int)
    rows = []
    for offset, donor in enumerate(target_donors):
        known_count = int(np.sum(source_donors == donor))
        rows.append(
            {
                "record_id": f"{prefix}:{split_name}:{start + offset}",
                "split": split_name,
                "source": "hyfa_official_sampled",
                "task": "hyfa_official_sampled",
                "donor": str(donor),
                "target_index": -1,
                "target_tissue": int(target_tissue[offset]),
                "known_indices": "",
                "known_tissues": "",
                "actual_known_count": known_count,
            }
        )
    return pd.DataFrame(rows)


def evaluate_official_loader(model, loader, device, split_name, prefix):
    _, _, _, _, forward, _ = load_hyfa_official_modules()
    model.eval()
    trues = []
    preds = []
    records = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            out, _ = forward(batch, model, device=device, preprocess_fn=None)
            pred = out["px_rate"].detach().cpu().numpy().astype(np.float32)
            true = batch.x_target.detach().cpu().numpy().astype(np.float32)
            trues.append(true)
            preds.append(pred)
            frame = batch_records_from_hyfa_data(batch, split_name, prefix, offset)
            records.append(frame)
            offset += len(frame)
    return np.vstack(trues), np.vstack(preds), pd.concat(records, ignore_index=True)


def write_split_files(out_dir, split_name, donors):
    frame = pd.DataFrame({"donor": list(donors), "split": split_name})
    frame.to_csv(out_dir / f"hyfa_official_split_{split_name}_donors.csv", index=False)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_gene_set(args.genes)
    adata = build_adata(data, use_demographics=args.use_demographics)
    split = split_donors(data.meta, seed=args.seed)
    train_donors = _limit_donors(donor_ids_from_indices(data.meta, split.train_idx), args.max_train_donors, args.seed)
    val_donors = _limit_donors(donor_ids_from_indices(data.meta, split.val_idx), args.max_val_donors, args.seed + 1)
    test_donors = _limit_donors(donor_ids_from_indices(data.meta, split.test_idx), args.max_test_donors, args.seed + 2)
    for split_name, donors in [("train", train_donors), ("validation", val_donors), ("test", test_donors)]:
        write_split_files(out_dir, split_name, donors)

    train_ds = make_official_dataset(adata, train_donors, static=False, seed=args.seed)
    val_ds = make_official_dataset(adata, val_donors, static=True, seed=args.seed + 1)
    test_ds = make_official_dataset(adata, test_donors, static=True, seed=args.seed + 2)
    print(
        f"donors train={len(train_donors)} val={len(val_donors)} test={len(test_donors)} "
        f"samples train={train_ds.adata_source.shape[0]} val={val_ds.adata_source.shape[0]} test={test_ds.adata_source.shape[0]}",
        flush=True,
    )

    config = make_config(args, adata)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _, _, _, model_cls, forward, compute_loss = load_hyfa_official_modules()
    model = model_cls(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_official)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_official)
    best_state = None
    best_val = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad()
            out, node_features = forward(batch, model, device=device, preprocess_fn=None)
            loss = compute_loss(batch, out, node_features, beta=config.beta)["loss"]
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_true, val_pred, _ = evaluate_official_loader(model, val_loader, device, "validation", "hyfa_official")
        val_mae = float(np.abs(val_pred - val_true).mean())
        print(f"epoch={epoch}/{args.epochs} train_loss={np.mean(losses):.6f} val_mae={val_mae:.6f}", flush=True)
        if best_val is None or val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_official)
    y_true, y_pred, test_records = evaluate_official_loader(model, test_loader, device, "test", "hyfa_official")
    test_records.to_csv(out_dir / f"hyfa_official_split_{args.genes}genes_seed{args.seed}_test_records.csv", index=False)
    gene_ids = np.array([f"gene_{i}" for i in range(y_true.shape[1])], dtype=object)
    bundle = out_dir / f"hyfa_official_split_{args.genes}genes_seed{args.seed}_predictions.npz"
    save_prediction_bundle(
        bundle,
        record_ids=test_records["record_id"].astype(str).to_numpy(),
        gene_ids=gene_ids,
        y_true=y_true,
        predictions={"hyfa_official_split": y_pred},
    )
    extra = {
        "gene_set": args.genes,
        "seed": args.seed,
        "task": "hyfa_official_split_sampled",
        "baseline": "HYFA_official_split",
        "train_loss": "official",
        "best_val_mae": best_val,
        "n_train_donors": len(train_donors),
        "n_val_donors": len(val_donors),
        "n_test_donors": len(test_donors),
    }
    overall, _, _ = write_evaluation_tables(
        bundle,
        test_records,
        out_dir,
        f"hyfa_official_split_{args.genes}genes_seed{args.seed}",
        extra=extra,
    )
    torch.save(
        {"state": model.state_dict(), "config": dict(config), "args": vars(args), "best_val_mae": best_val},
        out_dir / f"hyfa_official_split_{args.genes}genes_seed{args.seed}.pt",
    )
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
