"""Train the fixed-duration full-data BRIDGE encoder for resource production.

This is deliberately separate from benchmark training: every multi-tissue donor
contributes measured hide-one records, no validation subset is used for model
selection, and the epoch count is fixed before training.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--compat-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    from production_resource import build_full_hide_one_records, donor_vector
    from CML_Gene_Imputation.src.models.cml import CrossTissueCML, RecordDataset, contrastive_loss
    from CML_Gene_Imputation.src.train_hide_one import tissue_mean_matrix

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    compat = Path(args.compat_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = np.load(str(compat / "expr.npy"), mmap_mode="r")
    meta = pd.read_csv(str(compat / "meta.csv"), index_col=0)
    with (compat / "tissue_encoder.json").open("r") as handle:
        tissue_encoder = {key: int(value) for key, value in json.load(handle).items()}
    if expr.shape != (len(meta), 20000):
        raise ValueError("Expected metadata-aligned N x 20,000 expression matrix")
    if not np.isfinite(expr).all():
        raise ValueError("Expression contains non-finite values")

    records = build_full_hide_one_records(meta)
    tissue = meta["tissue_idx"].astype(int).to_numpy()
    all_indices = np.arange(len(meta), dtype=np.int64)
    tissue_means = tissue_mean_matrix(expr, tissue, all_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossTissueCML(20000, len(tissue_encoder), args.emb_dim).to(device)
    dataset = RecordDataset(records, expr, meta, tissue_means)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_path = out_dir / "dorian_full_data_seed42.pt"
    start_epoch = 1
    history = []
    if checkpoint_path.exists():
        saved = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(saved["cml_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history = list(saved.get("history", []))
        start_epoch = int(saved["epoch"]) + 1
        print("Resuming at epoch {}".format(start_epoch), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for known_expr, known_tissue, known_mask, target_tissue, target_resid, _, _ in loader:
            query, target = model(
                known_expr.to(device),
                known_tissue.to(device),
                known_mask.to(device),
                target_tissue.to(device),
                target_resid.to(device),
            )
            loss = contrastive_loss(query, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target_tissue)
            count += len(target_tissue)
        epoch_loss = total / max(count, 1)
        history.append({"epoch": epoch, "contrastive_loss": epoch_loss})
        print("epoch={}/{} contrastive_loss={:.8f}".format(epoch, args.epochs, epoch_loss), flush=True)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            payload = {
                "cml_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "production_config": vars(args),
                "n_records": len(records),
                "n_samples": len(meta),
                "n_donors": int(pd.Series(donor_vector(meta)).nunique()),
                "n_tissues": len(tissue_encoder),
                "training_protocol": "all_measured_multitissue_hide_one_fixed_duration",
                "model_selection": "none_fixed_200_epochs",
            }
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(payload, str(temporary))
            os.replace(str(temporary), str(checkpoint_path))
            pd.DataFrame(history).to_csv(str(out_dir / "encoder_training_history.csv"), index=False)

    np.save(str(out_dir / "tissue_means_all_measured.npy"), tissue_means.astype(np.float32))
    manifest = {
        "status": "encoder_complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "n_records": len(records),
        "n_samples": len(meta),
        "n_donors": int(pd.Series(donor_vector(meta)).nunique()),
        "n_tissues": len(tissue_encoder),
        "n_genes": int(expr.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch": args.batch,
        "emb_dim": args.emb_dim,
        "lr": args.lr,
        "memory_source": "originally_measured_profiles_only",
    }
    with (out_dir / "encoder_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    (out_dir / "ENCODER_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
