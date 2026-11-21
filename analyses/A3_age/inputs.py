from __future__ import annotations

import numpy as np


def align_matrix(values, source_record_ids, target_record_ids) -> np.ndarray:
    values = np.asarray(values)
    source = [str(x) for x in source_record_ids]
    target = [str(x) for x in target_record_ids]
    if values.shape[0] != len(source) or len(source) != len(set(source)):
        raise ValueError("source record IDs must be unique and aligned with matrix rows")
    lookup = {record_id: index for index, record_id in enumerate(source)}
    if len(target) != len(set(target)) or not set(target).issubset(lookup):
        raise ValueError("target record IDs must be unique and fully covered by source")
    return values[[lookup[record_id] for record_id in target]]
