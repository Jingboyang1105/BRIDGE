from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


REQUIRED_METRIC_COLUMNS = {
    "target_tissue", "target_tissue_name", "n_samples", "dorian_residual",
    "hyfa_residual", "dorian_mae", "hyfa_mae",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_tissues(metrics: pd.DataFrame, per_stratum: int = 2) -> pd.DataFrame:
    missing = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
    if missing:
        raise ValueError(f"missing metric columns: {sorted(missing)}")
    frame = metrics.copy()
    frame = frame.loc[frame["n_samples"].ge(25)].copy()
    numeric = ["dorian_residual", "hyfa_residual", "dorian_mae", "hyfa_mae"]
    frame = frame.dropna(subset=numeric)
    frame["residual_gap"] = frame["dorian_residual"] - frame["hyfa_residual"]
    frame["mae_gap"] = frame["hyfa_mae"] - frame["dorian_mae"]
    selected: list[pd.DataFrame] = []
    used: set[int] = set()

    advantage = frame.loc[frame["mae_gap"].ge(0)].sort_values(
        ["residual_gap", "target_tissue"], ascending=[False, True]
    )
    chosen = advantage.head(per_stratum).copy()
    chosen["selection_stratum"] = "clear_advantage"
    selected.append(chosen)
    used.update(chosen["target_tissue"].astype(int))

    remaining = frame.loc[~frame["target_tissue"].isin(used)].copy()
    remaining["abs_residual_gap"] = remaining["residual_gap"].abs()
    chosen = remaining.sort_values(
        ["abs_residual_gap", "target_tissue"], ascending=[True, True]
    ).head(per_stratum).copy()
    chosen["selection_stratum"] = "matched"
    selected.append(chosen)
    used.update(chosen["target_tissue"].astype(int))

    remaining = frame.loc[~frame["target_tissue"].isin(used)]
    chosen = remaining.sort_values(
        ["n_samples", "target_tissue"], ascending=[True, True]
    ).head(per_stratum).copy()
    chosen["selection_stratum"] = "low_sample"
    selected.append(chosen)
    result = pd.concat(selected, ignore_index=True)
    if len(result) != 3 * per_stratum or not result["target_tissue"].is_unique:
        raise ValueError("could not select six unique eligible tissues")
    return result


def write_locked_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"locked artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
