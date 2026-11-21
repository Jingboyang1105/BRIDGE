from __future__ import annotations

import numpy as np
from scipy import stats


def mean_fill_arm(truth: np.ndarray, masked: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=float)
    masked = np.asarray(masked, dtype=bool)
    if truth.shape[0] != masked.size or masked.all():
        raise ValueError("mask must leave at least one observed row")
    result = truth.copy()
    result[masked] = np.nanmean(truth[~masked], axis=0)
    return result


def imputed_arm(truth: np.ndarray, predictions: np.ndarray, masked: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    masked = np.asarray(masked, dtype=bool)
    if truth.shape != predictions.shape or truth.shape[0] != masked.size:
        raise ValueError("truth, predictions, and mask are not aligned")
    result = truth.copy()
    result[masked] = predictions[masked]
    return result


def fit_age_ols(expression: np.ndarray, age: np.ndarray, sex: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(expression, dtype=float)
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    if y.ndim != 2 or y.shape[0] != age.size or age.size != sex.size:
        raise ValueError("expression rows must align with age and sex")
    age_sd = age.std(ddof=0)
    if age_sd == 0:
        raise ValueError("age has zero variance")
    z_age = (age - age.mean()) / age_sd
    x = np.column_stack([np.ones(age.size), z_age, sex])
    xtx_inv = np.linalg.pinv(x.T @ x)
    coefficients = xtx_inv @ x.T @ y
    residual = y - x @ coefficients
    dof = age.size - np.linalg.matrix_rank(x)
    rss = np.sum(residual * residual, axis=0)
    sigma2 = rss / dof
    se = np.sqrt(np.maximum(0, xtx_inv[1, 1] * sigma2))
    beta = coefficients[1]
    t_value = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_value = 2 * stats.t.sf(np.abs(t_value), dof)
    return {"age_beta": beta, "age_se": se, "t_value": t_value, "p_value": p_value}


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(values)
    restored[order] = np.minimum(ranked, 1.0)
    result[valid] = restored
    return result
