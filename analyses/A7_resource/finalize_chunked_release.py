"""Assemble a standalone chunked HDF5 release and add file hashes."""

from __future__ import print_function

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compat-dir", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--gene-order", required=True)
    parser.add_argument("--script-dir", required=True)
    parser.add_argument("--chunk-rows", type=int, default=16)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_matrix(source, destination, rows_per_chunk):
    for start in range(0, source.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, source.shape[0])
        destination[start:end] = source[start:end]


def main():
    args = parse_args()
    sys.path.insert(0, args.script_dir)
    from production_resource import build_status_matrix

    compat = Path(args.compat_dir)
    resource = Path(args.resource_dir)
    completed = resource / "completed_expression"
    metadata = resource / "metadata"
    qc_dir = resource / "quality_control"
    release_dir = resource / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    measured = np.load(str(compat / "expr.npy"), mmap_mode="r")
    same = np.load(str(completed / "imputed_same_donor_expert.npy"), mmap_mode="r")
    memory = np.load(str(completed / "imputed_memory_expert.npy"), mmap_mode="r")
    dorian = np.load(str(completed / "imputed_dorian.npy"), mmap_mode="r")
    meta = pd.read_csv(str(compat / "meta.csv"), index_col=0)
    profiles = pd.read_csv(str(metadata / "imputed_profile_manifest.csv"))
    queries = profiles[["donor", "target_tissue"]].to_dict("records")
    donors, status = build_status_matrix(meta, queries, 68)
    gene_frame = pd.read_csv(args.gene_order)
    gene_ids = gene_frame.iloc[:, 0].astype(str).to_numpy()
    if len(gene_ids) != 20000:
        raise ValueError("Gene order must contain 20,000 rows")

    output = release_dir / "dorian_gtex_v11_completed_resource.h5"
    temporary = Path(str(output) + ".tmp")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(temporary), "w") as handle:
        handle.attrs["dataset"] = "GTEx v11"
        handle.attrs["method"] = "DORIAN"
        handle.attrs["seed"] = 42
        handle.attrs["value_source_codes"] = "0=not-imputed,1=measured,2=imputed"
        measured_ds = handle.create_dataset(
            "measured/expression", shape=measured.shape, dtype="f4",
            chunks=(args.chunk_rows, measured.shape[1]), compression="lzf",
        )
        dorian_ds = handle.create_dataset(
            "imputed/dorian", shape=dorian.shape, dtype="f4",
            chunks=(args.chunk_rows, dorian.shape[1]), compression="lzf",
        )
        same_ds = handle.create_dataset(
            "imputed/same_donor_expert", shape=same.shape, dtype="f4",
            chunks=(args.chunk_rows, same.shape[1]), compression="lzf",
        )
        memory_ds = handle.create_dataset(
            "imputed/memory_expert", shape=memory.shape, dtype="f4",
            chunks=(args.chunk_rows, memory.shape[1]), compression="lzf",
        )
        copy_matrix(measured, measured_ds, args.chunk_rows)
        copy_matrix(dorian, dorian_ds, args.chunk_rows)
        copy_matrix(same, same_ds, args.chunk_rows)
        copy_matrix(memory, memory_ds, args.chunk_rows)
        handle.create_dataset("measured/sample_id", data=meta.index.astype(str).to_numpy(), dtype=string_dtype)
        handle.create_dataset("measured/tissue", data=meta["tissue_idx"].astype(np.int16).to_numpy())
        handle.create_dataset("imputed/donor", data=profiles["donor"].astype(str).to_numpy(), dtype=string_dtype)
        handle.create_dataset("imputed/target_tissue", data=profiles["target_tissue"].astype(np.int16).to_numpy())
        handle.create_dataset("imputed/same_donor_weight", data=profiles["same_donor_weight"].astype(np.float32).to_numpy())
        handle.create_dataset("imputed/memory_weight", data=profiles["memory_weight"].astype(np.float32).to_numpy())
        handle.create_dataset("manifest/donor", data=np.asarray(donors, dtype=object), dtype=string_dtype)
        handle.create_dataset("manifest/gene_id", data=gene_ids, dtype=string_dtype)
        handle.create_dataset("manifest/value_source", data=status, chunks=(min(128, len(donors)), 68), compression="lzf")
    temporary.replace(output)

    with h5py.File(str(output), "r") as handle:
        if handle["measured/expression"].shape != (19788, 20000):
            raise ValueError("Measured dataset shape mismatch")
        if handle["imputed/dorian"].shape != (37030, 20000):
            raise ValueError("Imputed dataset shape mismatch")
        if int((handle["manifest/value_source"][:] == 1).sum()) != 19788:
            raise ValueError("Measured status count mismatch")
        if int((handle["manifest/value_source"][:] == 2).sum()) != 37030:
            raise ValueError("Imputed status count mismatch")
        for name in ["measured/expression", "imputed/dorian", "imputed/same_donor_expert", "imputed/memory_expert"]:
            dataset = handle[name]
            for start in range(0, dataset.shape[0], 128):
                if not np.isfinite(dataset[start : start + 128]).all():
                    raise ValueError("Non-finite values in {}".format(name))

    qc_path = qc_dir / "production_qc.json"
    qc = json.loads(qc_path.read_text())
    qc["chunked_release_path"] = str(output)
    qc["chunked_release_bytes"] = output.stat().st_size
    qc["chunked_release_sha256"] = sha256(output)
    qc["chunked_release_measured_count"] = int((status == 1).sum())
    qc["chunked_release_imputed_count"] = int((status == 2).sum())
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True))
    manifest_path = resource / "resource_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunked_release"] = str(output)
    manifest["chunked_release_sha256"] = qc["chunked_release_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (release_dir / "CHUNKED_RELEASE_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
