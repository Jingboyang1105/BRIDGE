"""Build measured-only memory, train the fixed mixer, and export the resource."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--compat-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--tissue-support", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--mixer-epochs", type=int, default=50)
    parser.add_argument("--mixer-batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))


def query_tensors(records, expr, tissue, tissue_means, start, end, max_known=64):
    batch = records[start:end]
    genes = expr.shape[1]
    known_expr = np.zeros((len(batch), max_known, genes), dtype=np.float32)
    known_tissue = np.zeros((len(batch), max_known), dtype=np.int64)
    known_mask = np.zeros((len(batch), max_known), dtype=np.float32)
    target_tissue = np.zeros(len(batch), dtype=np.int64)
    donors = []
    for row, record in enumerate(batch):
        known = list(record["known_indices"])[:max_known]
        for column, index in enumerate(known):
            source_tissue = int(tissue[int(index)])
            known_expr[row, column] = expr[int(index)] - tissue_means[source_tissue]
            known_tissue[row, column] = source_tissue
            known_mask[row, column] = 1.0
        target_tissue[row] = int(record["target_tissue"])
        donors.append(str(record["donor"]))
    return known_expr, known_tissue, known_mask, target_tissue, np.asarray(donors, dtype=object)


def same_donor_batch(records, expr, tissue, tissue_means, indices):
    result = np.empty((len(indices), expr.shape[1]), dtype=np.float32)
    for row, record_index in enumerate(indices):
        record = records[int(record_index)]
        residuals = []
        for known_index in record["known_indices"]:
            source_tissue = int(tissue[int(known_index)])
            residuals.append(expr[int(known_index)] - tissue_means[source_tissue])
        result[row] = tissue_means[int(record["target_tissue"])] + np.mean(residuals, axis=0)
    return result


@torch.no_grad()
def retrieve_batches(model, records, expr, tissue, tissue_means, memory_z, memory_donors, k, batch_size, device, destination, diagnostics):
    from production_resource import select_topk_excluding_donor

    memory_z = memory_z / (np.linalg.norm(memory_z, axis=1, keepdims=True) + 1e-12)
    for start in range(0, len(records), batch_size):
        end = min(start + batch_size, len(records))
        known_expr, known_tissue, known_mask, target_tissue, query_donors = query_tensors(
            records, expr, tissue, tissue_means, start, end
        )
        query = model.encode_query(
            torch.from_numpy(known_expr).to(device),
            torch.from_numpy(known_tissue).to(device),
            torch.from_numpy(known_mask).to(device),
            torch.from_numpy(target_tissue).to(device),
        ).cpu().numpy()
        similarity = query @ memory_z.T
        top, scores = select_topk_excluding_donor(similarity, query_donors, memory_donors, k)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        destination[start:end] = np.einsum("bk,bkg->bg", weights, expr[top]).astype(np.float32)
        destination.flush()
        diagnostics["top_indices"][start:end] = top
        diagnostics["top_scores"][start:end] = scores.astype(np.float32)
        diagnostics["query_donors"][start:end] = query_donors
        print("retrieval {}/{}".format(end, len(records)), flush=True)


def train_mixer(records, expr, tissue, tissue_means, retrieval, n_tissues, args, device, output_path):
    from CML_Gene_Imputation.src.models.two_expert_gate import TwoExpertGate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    gate = TwoExpertGate(n_tissues).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    order = np.arange(len(records), dtype=np.int64)
    history = []
    for epoch in range(1, args.mixer_epochs + 1):
        np.random.shuffle(order)
        total = 0.0
        count = 0
        gate.train()
        for start in range(0, len(order), args.mixer_batch):
            indices = order[start : start + args.mixer_batch]
            target_indices = np.asarray([int(records[int(i)]["idx"]) for i in indices], dtype=np.int64)
            target_tissues = np.asarray([int(records[int(i)]["target_tissue"]) for i in indices], dtype=np.int64)
            known_counts = np.asarray([int(records[int(i)]["n_known_tissues"]) for i in indices], dtype=np.float32)
            same = same_donor_batch(records, expr, tissue, tissue_means, indices)
            memory = np.asarray(retrieval[indices], dtype=np.float32)
            experts = np.stack([same, memory], axis=1)
            true = np.asarray(expr[target_indices], dtype=np.float32)
            prediction, _ = gate(
                torch.from_numpy(experts).to(device),
                torch.from_numpy(target_tissues).to(device),
                torch.from_numpy(known_counts).to(device),
            )
            loss = torch.nn.functional.smooth_l1_loss(prediction, torch.from_numpy(true).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(indices)
            count += len(indices)
        epoch_loss = total / max(count, 1)
        history.append({"epoch": epoch, "smooth_l1_loss": epoch_loss})
        print("mixer_epoch={}/{} loss={:.8f}".format(epoch, args.mixer_epochs, epoch_loss), flush=True)
        atomic_torch_save(
            {
                "gate_state": gate.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "training_protocol": "all_measured_hide_one_strict_query_donor_exclusion_fixed_duration",
            },
            output_path,
        )
    return gate, history


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from production_resource import build_full_hide_one_records, build_missing_queries, combine_experts, donor_vector
    from CML_Gene_Imputation.src.models.cml import CrossTissueCML
    from CML_Gene_Imputation.src.models.two_expert_gate import TwoExpertGate
    from CML_Gene_Imputation.src.train_hide_one import tissue_mean_matrix

    compat = Path(args.compat_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    memory_dir = out_dir / "memory_bank"
    completed_dir = out_dir / "completed_expression"
    metadata_dir = out_dir / "metadata"
    qc_dir = out_dir / "quality_control"
    for directory in [memory_dir, completed_dir, metadata_dir, qc_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    expr = np.load(str(compat / "expr.npy"), mmap_mode="r")
    meta = pd.read_csv(str(compat / "meta.csv"), index_col=0)
    with (compat / "tissue_encoder.json").open("r") as handle:
        encoder = {key: int(value) for key, value in json.load(handle).items()}
    tissue_names = {value: key for key, value in encoder.items()}
    tissue = meta["tissue_idx"].astype(int).to_numpy()
    donors = donor_vector(meta)
    tissue_means = np.load(str(model_dir / "tissue_means_all_measured.npy"), mmap_mode="r")
    checkpoint = torch.load(str(model_dir / "dorian_full_data_seed42.pt"), map_location="cpu")
    expected_checkpoint = {
        "epoch": 200,
        "n_records": 19787,
        "n_samples": 19788,
        "n_donors": 946,
        "n_tissues": 68,
    }
    for key, expected in expected_checkpoint.items():
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError("Checkpoint {} expected {}, found {}".format(key, expected, checkpoint.get(key)))
    for name, value in checkpoint["cml_state"].items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Checkpoint contains non-finite parameter: {}".format(name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossTissueCML(20000, len(encoder), 128).to(device)
    model.load_state_dict(checkpoint["cml_state"])
    model.eval()

    memory_z_path = memory_dir / "measured_memory_embeddings.npy"
    if memory_z_path.exists():
        memory_z = np.load(str(memory_z_path), mmap_mode="r")
    else:
        memory_z_out = np.lib.format.open_memmap(str(memory_z_path), mode="w+", dtype=np.float32, shape=(len(meta), 128))
        with torch.no_grad():
            for start in range(0, len(meta), args.encode_batch):
                end = min(start + args.encode_batch, len(meta))
                residual = np.asarray(expr[start:end], dtype=np.float32) - tissue_means[tissue[start:end]]
                memory_z_out[start:end] = model.encode_expr(torch.from_numpy(residual).to(device)).cpu().numpy()
                memory_z_out.flush()
                print("memory {}/{}".format(end, len(meta)), flush=True)
        memory_z = np.load(str(memory_z_path), mmap_mode="r")
    pd.DataFrame(
        {"memory_index": np.arange(len(meta)), "sample_id": meta.index.astype(str), "donor": donors, "tissue": tissue}
    ).to_csv(str(memory_dir / "measured_memory_manifest.csv"), index=False)

    train_records = build_full_hide_one_records(meta)
    train_retrieval_path = memory_dir / "mixer_training_memory_predictions.npy"
    train_diag_path = memory_dir / "mixer_training_retrieval_diagnostics.npz"
    train_retrieval_complete = memory_dir / "MIXER_RETRIEVAL_COMPLETE"
    if not train_retrieval_complete.exists():
        train_retrieval = np.lib.format.open_memmap(
            str(train_retrieval_path), mode="w+", dtype=np.float32, shape=(len(train_records), expr.shape[1])
        )
        train_diag = {
            "top_indices": np.empty((len(train_records), args.k), dtype=np.int64),
            "top_scores": np.empty((len(train_records), args.k), dtype=np.float32),
            "query_donors": np.empty(len(train_records), dtype=object),
        }
        retrieve_batches(
            model, train_records, expr, tissue, tissue_means, memory_z, donors,
            args.k, args.query_batch, device, train_retrieval, train_diag,
        )
        np.savez_compressed(str(train_diag_path), **train_diag)
        train_retrieval_complete.write_text("complete\n")
    train_retrieval = np.load(str(train_retrieval_path), mmap_mode="r")
    mixer_path = model_dir / "adaptive_mixer_full_data_seed42.pt"
    if mixer_path.exists() and int(torch.load(str(mixer_path), map_location="cpu").get("epoch", 0)) >= args.mixer_epochs:
        mixer_saved = torch.load(str(mixer_path), map_location="cpu")
        gate = TwoExpertGate(len(encoder)).to(device)
        gate.load_state_dict(mixer_saved["gate_state"])
        mixer_history = mixer_saved.get("history", [])
    else:
        gate, mixer_history = train_mixer(
            train_records, expr, tissue, tissue_means, train_retrieval,
            len(encoder), args, device, mixer_path,
        )
    gate.eval()
    pd.DataFrame(mixer_history).to_csv(str(model_dir / "mixer_training_history.csv"), index=False)

    support = pd.read_csv(args.tissue_support)
    eligible = set(support.loc[support["production_status"] == "eligible", "target_tissue"].astype(int))
    queries = build_missing_queries(meta, eligible, tissue_names)
    if len(queries) != 37030:
        raise ValueError("Expected 37,030 eligible missing cells, found {}".format(len(queries)))
    same_path = completed_dir / "imputed_same_donor_expert.npy"
    memory_path = completed_dir / "imputed_memory_expert.npy"
    final_path = completed_dir / "imputed_dorian.npy"
    memory_out = np.lib.format.open_memmap(str(memory_path), mode="w+", dtype=np.float32, shape=(len(queries), expr.shape[1]))
    diagnostics = {
        "top_indices": np.empty((len(queries), args.k), dtype=np.int64),
        "top_scores": np.empty((len(queries), args.k), dtype=np.float32),
        "query_donors": np.empty(len(queries), dtype=object),
    }
    retrieve_batches(
        model, queries, expr, tissue, tissue_means, memory_z, donors,
        args.k, args.query_batch, device, memory_out, diagnostics,
    )
    same_out = np.lib.format.open_memmap(str(same_path), mode="w+", dtype=np.float32, shape=(len(queries), expr.shape[1]))
    final_out = np.lib.format.open_memmap(str(final_path), mode="w+", dtype=np.float32, shape=(len(queries), expr.shape[1]))
    weights_out = np.empty((len(queries), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(queries), args.mixer_batch):
            end = min(start + args.mixer_batch, len(queries))
            indices = np.arange(start, end)
            same = same_donor_batch(queries, expr, tissue, tissue_means, indices)
            memory = np.asarray(memory_out[start:end], dtype=np.float32)
            experts = np.stack([same, memory], axis=1)
            target_tissues = np.asarray([queries[i]["target_tissue"] for i in indices], dtype=np.int64)
            counts = np.asarray([queries[i]["n_known_tissues"] for i in indices], dtype=np.float32)
            _, weights = gate(
                torch.from_numpy(experts).to(device),
                torch.from_numpy(target_tissues).to(device),
                torch.from_numpy(counts).to(device),
            )
            weights = weights.cpu().numpy().astype(np.float32)
            same_out[start:end] = same
            final_out[start:end] = combine_experts(same, memory, weights)
            weights_out[start:end] = weights
            same_out.flush(); final_out.flush()
            print("final {}/{}".format(end, len(queries)), flush=True)

    query_frame = pd.DataFrame(queries).drop(columns=["known_indices", "known_tissues"])
    query_frame.insert(0, "imputed_row", np.arange(len(query_frame)))
    query_frame["same_donor_weight"] = weights_out[:, 0]
    query_frame["memory_weight"] = weights_out[:, 1]
    query_frame.to_csv(str(metadata_dir / "imputed_profile_manifest.csv"), index=False)
    np.savez_compressed(
        str(metadata_dir / "imputed_retrieval_diagnostics.npz"),
        top_indices=diagnostics["top_indices"], top_scores=diagnostics["top_scores"],
    )

    observed_pairs = set(zip(donors.astype(str), tissue.astype(int)))
    predicted_pairs = set(zip(query_frame["donor"].astype(str), query_frame["target_tissue"].astype(int)))
    leakage = 0
    for row, query_donor in enumerate(diagnostics["query_donors"]):
        leakage += int(np.any(donors[diagnostics["top_indices"][row]].astype(str) == str(query_donor)))
    finite = bool(np.isfinite(final_out).all() and np.isfinite(same_out).all() and np.isfinite(memory_out).all())
    qc = {
        "n_donors": int(pd.Series(donors).nunique()),
        "n_tissues": len(encoder),
        "n_genes": int(expr.shape[1]),
        "n_measured_profiles": int(len(meta)),
        "n_imputed_profiles": int(len(queries)),
        "n_eligible_tissues": int(len(eligible)),
        "n_low_support_tissues": int((support["production_status"] != "eligible").sum()),
        "measured_prediction_overlap": int(len(observed_pairs & predicted_pairs)),
        "duplicate_predictions": int(query_frame.duplicated(["donor", "target_tissue"]).sum()),
        "query_donor_retrieval_leakage_records": int(leakage),
        "all_predictions_finite": finite,
        "no_chained_imputation": True,
        "memory_source": "originally_measured_profiles_only",
        "measured_expr_sha256": sha256(compat / "expr.npy"),
        "encoder_checkpoint_sha256": sha256(model_dir / "dorian_full_data_seed42.pt"),
        "mixer_checkpoint_sha256": sha256(mixer_path),
    }
    if qc["measured_prediction_overlap"] or qc["duplicate_predictions"] or leakage or not finite:
        raise RuntimeError("Production QC failed: {}".format(qc))
    with (qc_dir / "production_qc.json").open("w") as handle:
        json.dump(qc, handle, indent=2, sort_keys=True)
    release = {
        "status": "complete",
        "measured_expression": str(compat / "expr.npy"),
        "imputed_expression": str(final_path),
        "profile_manifest": str(metadata_dir / "imputed_profile_manifest.csv"),
        "tissue_support": str(args.tissue_support),
        "qc": str(qc_dir / "production_qc.json"),
        "value_source_rule": "measured profiles are immutable; only listed missing cells use imputed_dorian.npy",
    }
    with (out_dir / "resource_manifest.json").open("w") as handle:
        json.dump(release, handle, indent=2, sort_keys=True)
    (out_dir / "RESOURCE_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
