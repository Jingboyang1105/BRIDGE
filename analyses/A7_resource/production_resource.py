"""Full-data BRIDGE production-resource helpers and command-line workflow."""

from collections import defaultdict

import numpy as np


def _donor_from_sample(sample_id):
    parts = str(sample_id).split("-")
    if len(parts) < 2:
        raise ValueError("Cannot derive GTEx donor from sample ID: {}".format(sample_id))
    return "-".join(parts[:2])


def donor_vector(meta):
    if "donor" in meta.columns:
        return meta["donor"].astype(str).to_numpy()
    return np.asarray([_donor_from_sample(value) for value in meta.index], dtype=object)


def build_full_hide_one_records(meta):
    donors = donor_vector(meta)
    tissue = meta["tissue_idx"].astype(int).to_numpy()
    by_donor = defaultdict(list)
    for index, donor in enumerate(donors):
        by_donor[str(donor)].append(int(index))
    records = []
    for index, donor in enumerate(donors):
        known = [value for value in by_donor[str(donor)] if value != index]
        if not known:
            continue
        records.append(
            {
                "idx": int(index),
                "donor": str(donor),
                "target_tissue": int(tissue[index]),
                "known_indices": known,
                "known_tissues": [int(tissue[value]) for value in known],
                "n_known_tissues": len(known),
                "task": "full_data_hide_one",
                "source": "measured_hide_one",
            }
        )
    return records


def build_missing_queries(meta, eligible_tissues, tissue_names):
    donors = donor_vector(meta)
    tissue = meta["tissue_idx"].astype(int).to_numpy()
    by_donor = defaultdict(list)
    for index, donor in enumerate(donors):
        by_donor[str(donor)].append(int(index))
    queries = []
    for donor in sorted(by_donor):
        known = by_donor[donor]
        observed = {int(tissue[index]) for index in known}
        for target in sorted(int(value) for value in eligible_tissues):
            if target in observed:
                continue
            queries.append(
                {
                    "donor": donor,
                    "target_tissue": target,
                    "target_tissue_name": str(tissue_names[target]),
                    "known_indices": list(known),
                    "known_tissues": [int(tissue[index]) for index in known],
                    "n_known_tissues": len(known),
                }
            )
    return queries


def select_topk_excluding_donor(similarity, query_donors, memory_donors, k):
    similarity = np.asarray(similarity)
    query_donors = np.asarray(query_donors).astype(str)
    memory_donors = np.asarray(memory_donors).astype(str)
    if similarity.ndim != 2 or similarity.shape[0] != len(query_donors) or similarity.shape[1] != len(memory_donors):
        raise ValueError("Similarity and donor dimensions do not agree")
    top_rows = []
    score_rows = []
    for row, donor in zip(similarity, query_donors):
        permitted = np.flatnonzero(memory_donors != donor)
        if len(permitted) < int(k):
            raise ValueError("Query donor has fewer than k eligible other-donor memory rows")
        permitted_scores = row[permitted]
        local = np.argpartition(-permitted_scores, kth=int(k) - 1)[: int(k)]
        order = np.argsort(-permitted_scores[local])
        selected = permitted[local[order]]
        top_rows.append(selected.astype(np.int64))
        score_rows.append(row[selected])
    return np.vstack(top_rows), np.vstack(score_rows)


def combine_experts(donor_level_average, memory, weights):
    donor_level_average = np.asarray(donor_level_average, dtype=np.float32)
    memory = np.asarray(memory, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if donor_level_average.shape != memory.shape or weights.shape != (donor_level_average.shape[0], 2):
        raise ValueError("Expert and weight dimensions do not agree")
    return (donor_level_average * weights[:, :1] + memory * weights[:, 1:2]).astype(np.float32)


def build_status_matrix(meta, queries, n_tissues):
    """Return donor order and status matrix: 0 unavailable, 1 measured, 2 imputed."""
    sample_donors = donor_vector(meta).astype(str)
    sample_tissues = meta["tissue_idx"].astype(int).to_numpy()
    donors = sorted(set(sample_donors))
    donor_row = {donor: row for row, donor in enumerate(donors)}
    status = np.zeros((len(donors), int(n_tissues)), dtype=np.int8)
    for donor, tissue in zip(sample_donors, sample_tissues):
        status[donor_row[str(donor)], int(tissue)] = 1
    for query in queries:
        row = donor_row[str(query["donor"])]
        column = int(query["target_tissue"])
        if status[row, column] != 0:
            raise ValueError("Imputed query overlaps a measured donor-tissue cell")
        status[row, column] = 2
    return donors, status
