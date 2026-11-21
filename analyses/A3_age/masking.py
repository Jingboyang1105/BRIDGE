from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _seed(base_seed: int, tissue: int, rate: float, repeat: int) -> int:
    key = f"{base_seed}|{tissue}|{rate:.6f}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _stratified_indices(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> set[int]:
    groups = list(frame.groupby(["age_idx", "sex_idx"], sort=True).groups.values())
    quotas = np.array([len(g) * count / len(frame) for g in groups])
    take = np.floor(quotas).astype(int)
    remaining = count - int(take.sum())
    order = np.argsort(-(quotas - take), kind="stable")
    for idx in order[:remaining]:
        take[idx] += 1
    chosen: set[int] = set()
    for indices, n_take in zip(groups, take, strict=True):
        values = np.asarray(list(indices), dtype=int)
        if n_take:
            chosen.update(rng.choice(values, size=n_take, replace=False).tolist())
    return chosen


def make_masks(
    donors: pd.DataFrame,
    tissue: int,
    rates: list[float],
    repeats: int,
    base_seed: int = 42,
) -> pd.DataFrame:
    required = {"donor", "age_idx", "sex_idx"}
    if required.difference(donors.columns):
        raise ValueError("donors must contain donor, age_idx, and sex_idx")
    if donors["donor"].duplicated().any():
        raise ValueError("one row per donor is required")
    rows = []
    base = donors.reset_index(drop=True)
    for rate in rates:
        count = int(round(len(base) * rate))
        for repeat in range(repeats):
            seed = _seed(base_seed, tissue, rate, repeat)
            chosen = _stratified_indices(base, count, np.random.default_rng(seed))
            part = base.copy()
            part["tissue"] = int(tissue)
            part["missing_rate"] = float(rate)
            part["repeat"] = int(repeat)
            part["mask_seed"] = int(seed)
            part["masked"] = part.index.isin(chosen)
            rows.append(part)
    return pd.concat(rows, ignore_index=True)
