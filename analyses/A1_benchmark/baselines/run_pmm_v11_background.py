"""Run resumable GTEx v11 single-imputation PMM gene jobs in parallel."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

from CML_Gene_Imputation.src.accessible_protocol import build_filtered_accessible_frames
from CML_Gene_Imputation.src.data import GeneSetData
from CML_Gene_Imputation.src.pmm_v11_mainline import schedule_record_batches
from CML_Gene_Imputation.src.prepare_pmm_v11_run import prepare_metadata


ROOT = Path(__file__).resolve().parents[1]
R_RUNNER = ROOT / "src" / "run_pmm_v11_shard.R"


def load_data(cache_path: Path) -> GeneSetData:
    if cache_path.is_dir():
        expression = np.load(cache_path / "expr.npy", mmap_mode="r")
        metadata = pd.read_csv(cache_path / "meta.csv", index_col=0)
        tissue_encoder = json.loads(
            (cache_path / "tissue_encoder.json").read_text(encoding="utf-8")
        )
        tissue_encoder = {str(key): int(value) for key, value in tissue_encoder.items()}
        return GeneSetData(
            expr=expression,
            meta=metadata,
            tissue_encoder=tissue_encoder,
            gene_set=20_000,
            cache_file=cache_path,
        )
    with cache_path.open("rb") as handle:
        loaded = pickle.load(handle)
    return GeneSetData(
        expr=np.asarray(loaded[0], dtype=np.float32),
        meta=loaded[1],
        tissue_encoder=loaded[2],
        gene_set=20_000,
        cache_file=cache_path,
    )


def protocol_records(data: GeneSetData, protocol: str, hide_one_records: Path) -> pd.DataFrame:
    if protocol == "hide_one":
        records = pd.read_csv(hide_one_records)
    else:
        records = build_filtered_accessible_frames(data, seed=42, min_test_records=25)["test"]
    records = records[
        ["record_id", "donor", "target_index", "target_tissue", "known_indices"]
    ].copy()
    records["known_indices"] = records["known_indices"].map(
        lambda value: value if isinstance(value, str) else ";".join(map(str, value))
    )
    return schedule_record_batches(records)


def run_gene(
    gene_index: int,
    gene_id: str,
    expression: np.ndarray,
    shared_dir: Path,
    rscript: Path,
) -> tuple[int, np.ndarray, float]:
    job_dir = shared_dir / "jobs" / f"{gene_index:05d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    expression_csv = job_dir / "expression.csv"
    output_csv = job_dir / "predictions.csv"
    if output_csv.exists():
        prediction = pd.read_csv(output_csv)
        return gene_index, prediction["prediction"].to_numpy(dtype=np.float32), 0.0
    pd.DataFrame(
        {
            "sample_index": np.arange(expression.shape[0], dtype=int),
            gene_id: expression[:, gene_index],
        }
    ).to_csv(expression_csv, index=False)
    started = time.monotonic()
    result = subprocess.run(
        [
            str(rscript),
            "--vanilla",
            str(R_RUNNER),
            str(expression_csv),
            str(shared_dir / "metadata.csv"),
            str(shared_dir / "records.csv"),
            str(output_csv),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        (job_dir / "error.log").write_text(result.stderr, encoding="utf-8")
        raise RuntimeError(f"gene {gene_index} PMM failed: {result.stderr[-1000:]}")
    prediction = pd.read_csv(output_csv)
    if expression_csv.exists():
        expression_csv.unlink()
    return gene_index, prediction["prediction"].to_numpy(dtype=np.float32), elapsed


def run_protocol(
    data: GeneSetData,
    gene_ids: list[str],
    protocol: str,
    hide_one_records: Path,
    output_root: Path,
    rscript: Path,
    workers: int,
    gene_stop: int,
) -> None:
    output = output_root / protocol
    output.mkdir(parents=True, exist_ok=True)
    metadata, _ = prepare_metadata(data)
    records = protocol_records(data, protocol, hide_one_records)
    metadata.to_csv(output / "metadata.csv", index=False)
    records.to_csv(output / "records.csv", index=False)
    truth = data.expr[records["target_index"].to_numpy(dtype=int), :gene_stop]
    np.save(output / "truth.npy", truth.astype(np.float32))

    prediction_path = output / "predictions.npy"
    complete_path = output / "completed.npy"
    if prediction_path.exists():
        predictions = np.lib.format.open_memmap(prediction_path, mode="r+")
        completed = np.load(complete_path)
    else:
        predictions = np.lib.format.open_memmap(
            prediction_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), gene_stop),
        )
        predictions[:] = np.nan
        predictions.flush()
        completed = np.zeros(gene_stop, dtype=bool)
        np.save(complete_path, completed)

    pending = [index for index in range(gene_stop) if not completed[index]]
    durations: list[float] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        iterator = iter(pending)
        for _ in range(workers):
            try:
                index = next(iterator)
            except StopIteration:
                break
            future = pool.submit(
                run_gene, index, gene_ids[index], data.expr, output, rscript
            )
            active[future] = index
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                index, values, elapsed = future.result()
                if len(values) != len(records) or not np.isfinite(values).all():
                    raise RuntimeError(f"invalid prediction coverage for gene {index}")
                predictions[:, index] = values
                predictions.flush()
                completed[index] = True
                np.save(complete_path, completed)
                durations.append(elapsed)
                status = {
                    "protocol": protocol,
                    "completed_genes": int(completed.sum()),
                    "total_genes": int(gene_stop),
                    "records": int(len(records)),
                    "workers": int(workers),
                    "recent_seconds_per_gene": float(np.mean(durations[-20:])),
                    "failed": False,
                }
                (output / "status.json").write_text(
                    json.dumps(status, indent=2), encoding="utf-8"
                )
                try:
                    next_index = next(iterator)
                except StopIteration:
                    continue
                next_future = pool.submit(
                    run_gene,
                    next_index,
                    gene_ids[next_index],
                    data.expr,
                    output,
                    rscript,
                )
                active[next_future] = next_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--gene-order", required=True, type=Path)
    parser.add_argument("--hide-one-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rscript", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gene-stop", type=int, default=20_000)
    parser.add_argument(
        "--protocols", nargs="+", choices=["hide_one", "accessible"], default=["hide_one", "accessible"]
    )
    args = parser.parse_args()
    data = load_data(args.cache)
    gene_ids = (
        pd.read_csv(args.gene_order)["ensembl_id_version"].astype(str).tolist()
    )
    if len(gene_ids) != data.expr.shape[1]:
        raise ValueError("gene order does not match the v11 cache")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for protocol in args.protocols:
        run_protocol(
            data,
            gene_ids,
            protocol,
            args.hide_one_records,
            args.output_root,
            args.rscript,
            args.workers,
            min(args.gene_stop, len(gene_ids)),
        )
    (args.output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
