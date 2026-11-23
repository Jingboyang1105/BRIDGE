"""Donor-target record construction under strict split boundaries."""

import numpy as np
import pandas as pd

from bridge.donor_split import donor_id


def donor_to_indices(meta, indices):
    out = {}
    for idx in indices:
        idx = int(idx)
        out.setdefault(donor_id(meta.index[idx]), []).append(idx)
    for donor in out:
        out[donor] = sorted(out[donor])
    return out


def _idx_to_tissue(meta):
    tissue = meta["tissue_idx"].to_numpy()
    return {idx: int(tissue[idx]) for idx in range(len(meta))}


def build_hide_one_records(meta, indices, donor_map):
    """Build records where one target tissue is hidden and all others are known."""
    tissue_lookup = _idx_to_tissue(meta)
    records = []
    for idx in indices:
        idx = int(idx)
        donor = donor_id(meta.index[idx])
        known = [j for j in donor_map.get(donor, []) if int(j) != idx]
        if not known:
            continue
        records.append(
            {
                "idx": idx,
                "donor": donor,
                "target_tissue": tissue_lookup[idx],
                "known_indices": known,
                "known_tissues": [tissue_lookup[j] for j in known],
                "n_known_tissues": len(known),
                "task": "hide_one_tissue",
            }
        )
    return records


def build_known_count_records(meta, indices, donor_map, known_count, seed):
    """Build one target record per measured tissue with per-record random known tissues.

    For a donor with 16 measured tissues and known_count=2, this creates 16
    records. Each record hides a different target tissue and samples two known
    tissues from the other 15 tissues.
    """
    tissue_lookup = _idx_to_tissue(meta)
    records = []
    index_set = set(map(int, indices))
    for donor, donor_indices in donor_map.items():
        donor_indices = [int(i) for i in donor_indices if int(i) in index_set]
        for idx in donor_indices:
            candidates = [j for j in donor_indices if int(j) != int(idx)]
            if len(candidates) < int(known_count):
                continue
            rng = np.random.default_rng(seed + int(idx) * 1009 + int(known_count) * 997)
            known = sorted(rng.choice(candidates, size=int(known_count), replace=False).astype(int).tolist())
            records.append(
                {
                    "idx": idx,
                    "donor": donor,
                    "target_tissue": tissue_lookup[idx],
                    "known_indices": known,
                    "known_tissues": [tissue_lookup[j] for j in known],
                    "n_known_tissues": int(known_count),
                    "task": f"known_count_{known_count}",
                }
            )
    return records


def build_random_known_count_records(meta, indices, donor_map, known_count_grid, seed):
    """Build one target record per tissue and randomly choose a feasible known count.

    If the sampled count is not feasible for a query, this uses the equivalent
    rejection-free implementation: sample from counts that are <= the number of
    non-target candidate tissues.
    """
    tissue_lookup = _idx_to_tissue(meta)
    grid = sorted(int(x) for x in known_count_grid)
    records = []
    index_set = set(map(int, indices))
    for donor, donor_indices in donor_map.items():
        donor_indices = [int(i) for i in donor_indices if int(i) in index_set]
        for idx in donor_indices:
            candidates = [j for j in donor_indices if int(j) != int(idx)]
            feasible = [count for count in grid if count <= len(candidates)]
            if not feasible:
                continue
            rng = np.random.default_rng(seed + int(idx) * 1009 + len(candidates) * 101)
            known_count = int(rng.choice(feasible))
            known = sorted(rng.choice(candidates, size=known_count, replace=False).astype(int).tolist())
            records.append(
                {
                    "idx": idx,
                    "donor": donor,
                    "target_tissue": tissue_lookup[idx],
                    "known_indices": known,
                    "known_tissues": [tissue_lookup[j] for j in known],
                    "n_known_tissues": known_count,
                    "known_count_grid": ",".join(str(x) for x in grid),
                    "task": "random_known_count",
                }
            )
    return records


def build_random_feasible_known_records(meta, indices, donor_map, m, seed, task_name="random_known_m"):
    """Build m random feasible known-subset records for every donor-target query.

    For each target tissue, the known count is sampled uniformly from all
    feasible counts, 1..number of non-target tissues. This is the main realistic
    protocol: a donor with 16 non-target tissues can never draw 20 known tissues.
    """
    tissue_lookup = _idx_to_tissue(meta)
    records = []
    index_set = set(map(int, indices))
    m = int(m)
    if m <= 0:
        raise ValueError("m must be positive")
    for donor, donor_indices in donor_map.items():
        donor_indices = [int(i) for i in donor_indices if int(i) in index_set]
        for idx in donor_indices:
            candidates = [j for j in donor_indices if int(j) != int(idx)]
            if not candidates:
                continue
            for draw_id in range(m):
                rng = np.random.default_rng(seed + int(idx) * 1009 + int(draw_id) * 9176 + len(candidates) * 101)
                known_count = int(rng.integers(1, len(candidates) + 1))
                known = sorted(rng.choice(candidates, size=known_count, replace=False).astype(int).tolist())
                records.append(
                    {
                        "idx": idx,
                        "donor": donor,
                        "target_tissue": tissue_lookup[idx],
                        "known_indices": known,
                        "known_tissues": [tissue_lookup[j] for j in known],
                        "n_known_tissues": known_count,
                        "task": task_name,
                        "source": f"random_known_m{m}",
                        "draw_id": int(draw_id),
                    }
                )
    return records


def build_whole_blood_records(meta, indices, donor_map, blood_tissue):
    """Build TEEBoT/HYFA aligned records using only Whole Blood as source."""
    tissue_lookup = _idx_to_tissue(meta)
    records = []
    index_set = set(map(int, indices))
    for donor, donor_indices in donor_map.items():
        donor_indices = [int(i) for i in donor_indices if int(i) in index_set]
        blood = [idx for idx in donor_indices if tissue_lookup[idx] == int(blood_tissue)]
        if not blood:
            continue
        blood_idx = int(blood[0])
        for idx in donor_indices:
            idx = int(idx)
            if idx == blood_idx or tissue_lookup[idx] == int(blood_tissue):
                continue
            records.append(
                {
                    "idx": idx,
                    "donor": donor,
                    "target_tissue": tissue_lookup[idx],
                    "known_indices": [blood_idx],
                    "known_tissues": [int(blood_tissue)],
                    "n_known_tissues": 1,
                    "task": "whole_blood_to_tissue",
                }
            )
    return records


def _encode_list(values):
    return ";".join(str(int(v)) for v in values)


def build_unified_record_repository(meta, indices, donor_map, gene_set, split_name, known_count_grid, seed):
    """Return one table containing both hide-one and random-known-count queries."""
    records = []
    for source, built in [
        ("hide_one", build_hide_one_records(meta, indices, donor_map)),
        ("random_known_count", build_random_known_count_records(meta, indices, donor_map, known_count_grid, seed)),
    ]:
        for local_id, record in enumerate(built):
            row = {
                "record_id": f"{gene_set}:{split_name}:{source}:{local_id}",
                "gene_set": int(gene_set),
                "split": split_name,
                "source": source,
                "task": record["task"],
                "donor": record["donor"],
                "target_index": int(record["idx"]),
                "target_tissue": int(record["target_tissue"]),
                "known_indices": _encode_list(record["known_indices"]),
                "known_tissues": _encode_list(record["known_tissues"]),
                "actual_known_count": int(record["n_known_tissues"]),
            }
            if "known_count_grid" in record:
                row["known_count_grid"] = record["known_count_grid"]
            else:
                row["known_count_grid"] = "all_other"
            records.append(row)
    columns = [
        "record_id",
        "gene_set",
        "split",
        "source",
        "task",
        "donor",
        "target_index",
        "target_tissue",
        "known_indices",
        "known_tissues",
        "actual_known_count",
        "known_count_grid",
    ]
    return pd.DataFrame(records, columns=columns)


