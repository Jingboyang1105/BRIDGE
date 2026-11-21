from __future__ import annotations

import numpy as np

from .association import bh_fdr


def fast_age_beta(expression, age, sex) -> np.ndarray:
    """Age coefficient after residualizing standardized age on intercept and sex."""
    y = np.asarray(expression, dtype=float)
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    z = (age - age.mean()) / age.std(ddof=0)
    nuisance = np.column_stack([np.ones(len(age)), sex])
    residual_age = z - nuisance @ (np.linalg.pinv(nuisance) @ z)
    return (residual_age @ y) / (residual_age @ z)


def parse_memberships(memberships) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {}
    for index, value in enumerate(memberships):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        for pathway in str(value).split(";"):
            pathway = pathway.strip()
            if pathway:
                result.setdefault(pathway, []).append(index)
    return {key: np.asarray(value, dtype=int) for key, value in result.items()}


def calibration_metrics(p_value, t_value) -> dict[str, float]:
    p = np.asarray(p_value, dtype=float)
    t = np.asarray(t_value, dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    p, t = p[valid], t[valid]
    fdr = bh_fdr(p)
    return {
        "n_genes": int(len(p)),
        "raw_p05_count": int(np.sum(p < 0.05)),
        "raw_p05_fraction": float(np.mean(p < 0.05)),
        "fdr05_count": int(np.sum(fdr < 0.05)),
        "lambda_median_t2": float(np.median(t * t) / 0.4549364231195727),
    }
