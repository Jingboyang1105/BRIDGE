"""Donor-level split utilities for the controlled DORIAN experiments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_TISSUE_COUNT_BINS = (2, 4, 8, 12, 20, np.inf)


@dataclass(frozen=True)
class DonorSplit:
    train_donors: np.ndarray
    val_donors: np.ndarray
    test_donors: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    donor_tissue_counts: pd.Series
    bin_summary: pd.DataFrame


def donor_id(sample_id):
    """Return the GTEx donor id from a GTEx sample id."""
    return "-".join(str(sample_id).split("-")[:2])


def sample_donors(meta):
    return np.array([donor_id(idx) for idx in meta.index])


def donor_tissue_counts(meta):
    """Count unique measured tissues per donor."""
    donors = sample_donors(meta)
    frame = pd.DataFrame({"donor": donors, "tissue": meta["tissue_idx"].to_numpy()})
    return frame.groupby("donor")["tissue"].nunique().sort_index()


def eligible_donors(meta, min_tissues=2):
    counts = donor_tissue_counts(meta)
    return counts[counts >= int(min_tissues)].index.to_numpy()


def _count_bin(count, bins):
    for left, right in zip(bins[:-1], bins[1:]):
        if count >= left and count < right:
            return f"[{int(left)},{'inf' if np.isinf(right) else int(right)})"
    raise ValueError(f"Tissue count {count} did not fit bins {bins}")


def _allocate_bin(donors, rng, train_frac, val_frac):
    donors = np.array(donors, dtype=object)
    rng.shuffle(donors)
    n = len(donors)
    if n < 3:
        return donors, np.array([], dtype=object), np.array([], dtype=object)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_train -= 1
        n_test = 1

    train = donors[:n_train]
    val = donors[n_train : n_train + n_val]
    test = donors[n_train + n_val :]
    return train, val, test


def _indices_for_donors(all_donors, selected):
    selected = set(selected)
    return np.where(np.array([donor in selected for donor in all_donors]))[0].astype(int)


def split_donors(
    meta,
    seed=42,
    train_frac=0.70,
    val_frac=0.10,
    test_frac=0.20,
    bins=DEFAULT_TISSUE_COUNT_BINS,
    min_tissues=2,
):
    """Split eligible donors 70/10/20, stratified by donor tissue count bins.

    Very small bins cannot be divided three ways; those donors are kept in train
    so they do not leak into validation/test with unstable strata.
    """
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    counts = donor_tissue_counts(meta)
    eligible = counts[counts >= int(min_tissues)]
    by_bin = {}
    for donor, count in eligible.items():
        by_bin.setdefault(_count_bin(int(count), bins), []).append(donor)

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    summary_rows = []
    for label in sorted(by_bin):
        donors = np.array(sorted(by_bin[label]), dtype=object)
        train, val, test = _allocate_bin(donors, rng, train_frac, val_frac)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
        summary_rows.append(
            {
                "bin": label,
                "n_donors": int(len(donors)),
                "train": int(len(train)),
                "val": int(len(val)),
                "test": int(len(test)),
            }
        )

    train_donors = np.sort(np.concatenate(train_parts) if train_parts else np.array([], dtype=object))
    val_donors = np.sort(np.concatenate(val_parts) if val_parts else np.array([], dtype=object))
    test_donors = np.sort(np.concatenate(test_parts) if test_parts else np.array([], dtype=object))
    donors = sample_donors(meta)
    return DonorSplit(
        train_donors=train_donors,
        val_donors=val_donors,
        test_donors=test_donors,
        train_idx=_indices_for_donors(donors, train_donors),
        val_idx=_indices_for_donors(donors, val_donors),
        test_idx=_indices_for_donors(donors, test_donors),
        donor_tissue_counts=counts,
        bin_summary=pd.DataFrame(summary_rows).sort_values("bin").reset_index(drop=True),
    )


