"""GTEx v11 seed-42 single-imputation PMM baseline utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PMMMainlineConfig:
    genes: int = 20_000
    seed: int = 42
    imputations: int = 1
    scale: str = "log1p_tpm"


MAINLINE_CONFIG = PMMMainlineConfig()
KEY_COLUMNS = ["record_id", "gene_id", "donor", "target_tissue"]


def parse_indices(value: object) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return [int(item) for item in json.loads(text)]
        return [int(item) for item in text.split(";") if item != ""]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(item) for item in value]
    raise ValueError(f"unsupported index list: {value!r}")


def schedule_record_batches(records: pd.DataFrame) -> pd.DataFrame:
    """Assign records to the fewest deterministic donor-unique batches."""
    required = {"record_id", "donor", "target_index", "target_tissue", "known_indices"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    if records["record_id"].duplicated().any():
        raise ValueError("record_id must be unique")
    scheduled = records.copy()
    scheduled["batch_id"] = (
        scheduled.sort_values(["donor", "target_tissue", "record_id"], kind="stable")
        .groupby("donor")
        .cumcount()
        .reindex(scheduled.index)
        .astype(int)
    )
    if scheduled.duplicated(["batch_id", "donor"]).any():
        raise AssertionError("a query donor appears twice in one PMM batch")
    return scheduled


def _metadata_donors(metadata: pd.DataFrame) -> pd.Series:
    if "donor" in metadata:
        return metadata["donor"].astype(str)
    return pd.Series(
        ["-".join(str(sample).split("-")[:2]) for sample in metadata.index],
        index=metadata.index,
    )


def build_batch_cells(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    records: pd.DataFrame,
    *,
    gene_ids: list[str],
    train_indices: np.ndarray,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one long-form PMM batch from training references and test queries."""
    if protocol not in {"hide_one", "accessible"}:
        raise ValueError("protocol must be hide_one or accessible")
    values = np.asarray(expression)
    if values.shape != (len(metadata), len(gene_ids)):
        raise ValueError("expression shape does not match metadata and gene IDs")
    donors = _metadata_donors(metadata)
    tissues = metadata["tissue_idx"].astype(int)
    train_indices = np.asarray(train_indices, dtype=int)
    query_donors = set(records["donor"].astype(str))
    reference_donors = set(donors.iloc[train_indices])
    if query_donors.intersection(reference_donors):
        raise ValueError("training reference donors overlap query donors")

    rows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for sample_index in train_indices:
        donor = str(donors.iloc[sample_index])
        tissue = int(tissues.iloc[sample_index])
        for gene_index, gene_id in enumerate(gene_ids):
            rows.append(
                {
                    "row_id": f"reference:{donor}",
                    "record_id": "",
                    "role": "reference",
                    "gene_id": str(gene_id),
                    "donor": donor,
                    "tissue": tissue,
                    "value": float(values[sample_index, gene_index]),
                    "is_target": False,
                }
            )

    for record in records.itertuples(index=False):
        target_index = int(record.target_index)
        known_indices = parse_indices(record.known_indices)
        donor = str(record.donor)
        if str(donors.iloc[target_index]) != donor:
            raise ValueError(f"target donor mismatch for {record.record_id}")
        if target_index in known_indices:
            raise ValueError(f"target appears among known inputs for {record.record_id}")
        if any(str(donors.iloc[index]) != donor for index in known_indices):
            raise ValueError(f"known inputs cross donor identity for {record.record_id}")
        target_tissue = int(record.target_tissue)
        if int(tissues.iloc[target_index]) != target_tissue:
            raise ValueError(f"target tissue mismatch for {record.record_id}")

        for gene_index, gene_id in enumerate(gene_ids):
            for sample_index in known_indices:
                rows.append(
                    {
                        "row_id": f"query:{record.record_id}",
                        "record_id": str(record.record_id),
                        "role": "query",
                        "gene_id": str(gene_id),
                        "donor": donor,
                        "tissue": int(tissues.iloc[sample_index]),
                        "value": float(values[sample_index, gene_index]),
                        "is_target": False,
                    }
                )
            rows.append(
                {
                    "row_id": f"query:{record.record_id}",
                    "record_id": str(record.record_id),
                    "role": "query",
                    "gene_id": str(gene_id),
                    "donor": donor,
                    "tissue": target_tissue,
                    "value": np.nan,
                    "is_target": True,
                }
            )
            targets.append(
                {
                    "record_id": str(record.record_id),
                    "gene_id": str(gene_id),
                    "donor": donor,
                    "target_tissue": target_tissue,
                    "target_value": float(values[target_index, gene_index]),
                }
            )
    cells = pd.DataFrame(rows)
    requested = pd.DataFrame(targets)
    if cells.duplicated(["row_id", "gene_id", "tissue"]).any():
        raise ValueError("duplicate donor-query gene-tissue cell")
    return cells, requested


def validate_prediction_frame(
    predictions: pd.DataFrame, requested: pd.DataFrame
) -> pd.DataFrame:
    required_prediction = set(KEY_COLUMNS + ["prediction"])
    missing = required_prediction.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    missing = set(KEY_COLUMNS).difference(requested.columns)
    if missing:
        raise ValueError(f"request columns missing: {sorted(missing)}")
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate prediction keys")
    left = requested[KEY_COLUMNS].astype(
        {"record_id": str, "gene_id": str, "donor": str, "target_tissue": int}
    )
    right = predictions[KEY_COLUMNS + ["prediction"]].copy()
    right = right.astype(
        {"record_id": str, "gene_id": str, "donor": str, "target_tissue": int}
    )
    if set(map(tuple, left.to_numpy())) != set(
        map(tuple, right[KEY_COLUMNS].to_numpy())
    ):
        raise ValueError("predictions do not exactly cover requested targets")
    aligned = left.merge(right, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if not np.isfinite(aligned["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("predictions must be finite")
    return aligned


def r_runner_path() -> Path:
    return Path(__file__).with_name("run_pmm_v11_mainline.R")
