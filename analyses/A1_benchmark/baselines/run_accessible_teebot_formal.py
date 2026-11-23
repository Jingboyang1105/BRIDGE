"""Run the locked GTEx v11 four-accessible-tissue TEEBoT benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _add_repo_parent():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CML_Gene_Imputation").is_dir():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("could not locate CML_Gene_Imputation parent")


REPO_PARENT = _add_repo_parent()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import metric_row, residual_pearson_by_record
from protocol import (
    ACCESSIBLE_TISSUES,
    assert_disjoint_donors,
    assert_shared_records,
    canonicalize_accessible_frame,
    parse_known_indices,
    record_hash,
    target_indices,
)
from teebot import build_accessible_design, fit_predict_teebot_style

from CML_Gene_Imputation.src.accessible_protocol import build_filtered_accessible_frames
from CML_Gene_Imputation.src.benchmark_adapter import load_prediction_bundle, save_prediction_bundle
from CML_Gene_Imputation.src.data import load_gene_set
from CML_Gene_Imputation.src.splits import split_donors
from CML_Gene_Imputation.src.train_hide_one import tissue_mean_matrix


ROOT = REPO_PARENT / "CML_Gene_Imputation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--genes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--min_test_records", type=int, default=25)
    parser.add_argument(
        "--bridge_bundle",
        default=str(ROOT / "accessible/results/cml_accessible_20k_seed42/cml_accessible_20000genes_seed42_predictions.npz"),
    )
    parser.add_argument(
        "--bridge_records",
        default=str(ROOT / "accessible/results/cml_accessible_20k_seed42/cml_accessible_20000genes_seed42_test_records.csv"),
    )
    parser.add_argument(
        "--hyfa_bundle",
        default=str(ROOT / "accessible/results/hyfa_accessible_20k_seed42/hyfa_accessible_20000genes_seed42_predictions.npz"),
    )
    parser.add_argument(
        "--hyfa_records",
        default=str(ROOT / "accessible/results/hyfa_accessible_20k_seed42/hyfa_accessible_20000genes_seed42_test_records.csv"),
    )
    return parser.parse_args()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values) -> str:
    array = np.asarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _source_expression(expr, frame: pd.DataFrame) -> np.ndarray:
    indices = np.asarray([parse_known_indices(value) for value in frame["known_indices"]], dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("accessible source index matrix must contain four columns")
    return np.asarray(expr[indices], dtype=np.float32)


def _age_sex(meta: pd.DataFrame, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first_source = np.asarray([parse_known_indices(value)[0] for value in frame["known_indices"]], dtype=np.int64)
    age_year_scale = meta.iloc[first_source]["age_idx"].to_numpy(dtype=np.float32) * 10.0
    sex = meta.iloc[first_source]["sex_idx"].to_numpy(dtype=np.float32)
    return age_year_scale, sex


def fit_teebot_variants(data, raw_frames, components: int, seed: int):
    train = raw_frames["train"].reset_index(drop=True)
    test = raw_frames["test"].reset_index(drop=True)
    expression = data.expr
    truth = np.asarray(expression[target_indices(test)], dtype=np.float32)
    output = {
        "TEEBoT_accessible_age_sex": np.empty_like(truth),
        "TEEBoT_accessible_expression_only": np.empty_like(truth),
    }
    support = []
    for target_id in sorted(test["target_tissue"].astype(int).unique()):
        train_target = train[train["target_tissue"].astype(int) == target_id].reset_index(drop=True)
        test_mask = test["target_tissue"].astype(int).to_numpy() == target_id
        test_target = test.loc[test_mask].reset_index(drop=True)
        if len(train_target) < 2:
            raise ValueError(f"target tissue {target_id} has fewer than two training donors")
        source_train = _source_expression(expression, train_target)
        source_test = _source_expression(expression, test_target)
        age_train, sex_train = _age_sex(data.meta, train_target)
        age_test, sex_test = _age_sex(data.meta, test_target)
        y_train = np.asarray(expression[target_indices(train_target)], dtype=np.float32)
        for label, demographics in (
            ("TEEBoT_accessible_age_sex", True),
            ("TEEBoT_accessible_expression_only", False),
        ):
            x_train = build_accessible_design(
                source_train, age_train, sex_train, include_demographics=demographics
            )
            x_test = build_accessible_design(
                source_test, age_test, sex_test, include_demographics=demographics
            )
            output[label][test_mask] = fit_predict_teebot_style(
                x_train, y_train, x_test, n_components=components, seed=seed
            )
        support.append({
            "target_tissue_id": int(target_id),
            "train_records": int(len(train_target)),
            "test_records": int(len(test_target)),
        })
    return truth, output, pd.DataFrame(support)


def summarize(truth, predictions, target_ids, tissue_means, tissue_names):
    mean_rows = tissue_means[np.asarray(target_ids, dtype=np.int64)]
    overall = pd.DataFrame([
        metric_row(method, truth, prediction, mean_rows)
        for method, prediction in predictions.items()
    ])
    per_tissue_rows = []
    record_rows = []
    target_ids = np.asarray(target_ids, dtype=np.int64)
    for method, prediction in predictions.items():
        residual = residual_pearson_by_record(truth, prediction, mean_rows)
        mae = np.mean(np.abs(np.asarray(prediction) - np.asarray(truth)), axis=1)
        for order, (target, rec_mae, rec_residual) in enumerate(zip(target_ids, mae, residual)):
            record_rows.append({
                "method": method,
                "record_order": int(order),
                "target_tissue_id": int(target),
                "target_tissue_name": tissue_names[int(target)],
                "mae": float(rec_mae),
                "residual_pearson": float(rec_residual) if np.isfinite(rec_residual) else np.nan,
            })
        for target in sorted(np.unique(target_ids)):
            mask = target_ids == target
            row = metric_row(method, truth[mask], prediction[mask], mean_rows[mask])
            row.update({
                "target_tissue_id": int(target),
                "target_tissue_name": tissue_names[int(target)],
            })
            per_tissue_rows.append(row)
    return overall, pd.DataFrame(per_tissue_rows), pd.DataFrame(record_rows)


def run(args):
    np.random.seed(args.seed)
    out = Path(args.out_dir)
    manifests = out / "manifests"
    raw = out / "raw"
    tables = out / "tables"
    for directory in (out, manifests, raw, tables):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_gene_set(args.genes)
    if data.expr.shape[1] != args.genes:
        raise ValueError("loaded gene count differs from locked gene count")
    raw_frames = build_filtered_accessible_frames(
        data, seed=args.seed, min_test_records=args.min_test_records
    )
    canonical = {name: canonicalize_accessible_frame(frame) for name, frame in raw_frames.items()}
    assert_disjoint_donors(canonical)
    locked_test = canonical["test"].reset_index(drop=True)
    locked_hash = record_hash(locked_test)
    for name, frame in raw_frames.items():
        frame.to_csv(manifests / f"accessible_locked_{name}_records.csv", index=False)

    bridge_records = canonicalize_accessible_frame(pd.read_csv(args.bridge_records))
    hyfa_records = canonicalize_accessible_frame(pd.read_csv(args.hyfa_records))
    assert_shared_records(locked_test, bridge_records)
    assert_shared_records(locked_test, hyfa_records)

    bridge = load_prediction_bundle(args.bridge_bundle)
    hyfa = load_prediction_bundle(args.hyfa_bundle)
    expected_ids = locked_test["record_id"].astype(str).tolist()
    if bridge["record_ids"].astype(str).tolist() != expected_ids:
        raise ValueError("BRIDGE record IDs differ from locked manifest")
    if hyfa["record_ids"].astype(str).tolist() != expected_ids:
        raise ValueError("HYFA record IDs differ from locked manifest")
    if bridge["gene_ids"].astype(str).tolist() != hyfa["gene_ids"].astype(str).tolist():
        raise ValueError("BRIDGE and HYFA gene order differs")
    if not np.array_equal(bridge["y_true"], hyfa["y_true"]):
        raise ValueError("BRIDGE and HYFA truth matrices differ")

    truth, teebot, support = fit_teebot_variants(data, raw_frames, args.components, args.seed)
    if not np.array_equal(truth, bridge["y_true"]):
        raise ValueError("TEEBoT truth differs from existing formal truth")

    split = split_donors(data.meta, seed=args.seed)
    tissue_ids = data.meta["tissue_idx"].to_numpy(dtype=np.int64)
    tissue_means = tissue_mean_matrix(data.expr, tissue_ids, split.train_idx)
    target_ids = raw_frames["test"]["target_tissue"].to_numpy(dtype=np.int64)
    predictions = {
        "BRIDGE_accessible": np.asarray(bridge["predictions"]["CML_accessible_hide_one_gate"], dtype=np.float32),
        **teebot,
        "HYFA_accessible": np.asarray(hyfa["predictions"]["HYFA_accessible"], dtype=np.float32),
        "target_tissue_mean": np.asarray(tissue_means[target_ids], dtype=np.float32),
    }
    for method, values in predictions.items():
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction coverage for {method}")

    gene_ids = bridge["gene_ids"].astype(str)
    save_prediction_bundle(
        raw / "accessible_teebot_benchmark_predictions.npz",
        np.asarray(expected_ids, dtype=object),
        gene_ids,
        truth,
        predictions,
    )
    tissue_names = {int(value): str(key) for key, value in data.tissue_encoder.items()}
    overall, per_tissue, by_record = summarize(
        truth, predictions, target_ids, tissue_means, tissue_names
    )
    by_record.insert(1, "record_id", [expected_ids[i] for i in by_record["record_order"]])
    overall.to_csv(tables / "Table_accessible_overall_metrics.csv", index=False)
    per_tissue.to_csv(tables / "Table_accessible_per_tissue_metrics.csv", index=False)
    by_record.to_csv(tables / "Table_accessible_record_metrics.csv", index=False)
    support.to_csv(tables / "Table_accessible_target_support.csv", index=False)

    provenance = {
        "status": "complete",
        "dataset": "GTEx v11",
        "genes": int(args.genes),
        "seed": int(args.seed),
        "source_tissues": list(ACCESSIBLE_TISSUES),
        "train_records": int(len(raw_frames["train"])),
        "validation_records": int(len(raw_frames["validation"])),
        "test_records": int(len(raw_frames["test"])),
        "test_donors": int(raw_frames["test"]["donor"].nunique()),
        "target_tissues": int(raw_frames["test"]["target_tissue"].nunique()),
        "test_record_hash": locked_hash,
        "gene_order_hash": array_hash(gene_ids.astype("U")),
        "bridge_bundle_sha256": sha256(args.bridge_bundle),
        "hyfa_bundle_sha256": sha256(args.hyfa_bundle),
        "teebot_components": int(args.components),
        "teebot_primary_demographics": ["age_divided_by_100", "encoded_sex"],
        "bridge_retrained": False,
        "donor_disjoint_split": True,
        "shared_record_order": True,
        "shared_truth": True,
        "finite_predictions": True,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
