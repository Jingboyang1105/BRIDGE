from __future__ import annotations

import numpy as np
import pandas as pd

from .association import bh_fdr, fit_age_ols, imputed_arm, mean_fill_arm
from .endpoints import recovery_metrics
from .masking import make_masks


def _fit(y: np.ndarray, age: np.ndarray, sex: np.ndarray) -> dict[str, np.ndarray]:
    result = fit_age_ols(y, age, sex)
    result["fdr"] = bh_fdr(result["p_value"])
    return result


def run_tissue(
    tissue: int,
    tissue_name: str,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    demographics: pd.DataFrame,
    rates: list[float],
    repeats: int,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    truth = np.asarray(truth, dtype=float)
    if len(demographics) != truth.shape[0]:
        raise ValueError("demographics and expression rows are not aligned")
    if not np.isfinite(truth).all():
        raise ValueError("truth contains non-finite values")
    for method, values in predictions.items():
        if np.asarray(values).shape != truth.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction matrix for {method}")
    age = demographics["age_year"].to_numpy(dtype=float)
    sex = demographics["sex_idx"].to_numpy(dtype=float)
    oracle = _fit(truth, age, sex)
    arrays: dict[str, np.ndarray] = {
        "oracle_beta": oracle["age_beta"],
        "oracle_se": oracle["age_se"],
        "oracle_p": oracle["p_value"],
        "oracle_fdr": oracle["fdr"],
    }
    masks = make_masks(
        demographics[["donor", "age_idx", "sex_idx"]], tissue=tissue,
        rates=rates, repeats=repeats, base_seed=base_seed,
    )
    summary_rows = []
    for (rate, repeat), mask_frame in masks.groupby(["missing_rate", "repeat"], sort=True):
        masked = mask_frame["masked"].to_numpy(dtype=bool)
        arms = {
            "Observed-only": truth[~masked],
            "Mean-fill": mean_fill_arm(truth, masked),
        }
        for method, values in predictions.items():
            arms[method] = imputed_arm(truth, values, masked)
        for arm, expression in arms.items():
            if arm == "Observed-only":
                fitted = _fit(expression, age[~masked], sex[~masked])
                n = int((~masked).sum())
            else:
                fitted = _fit(expression, age, sex)
                n = len(age)
            key = f"rate{rate:.1f}_repeat{int(repeat)}_{arm.lower().replace('-', '_').replace(' ', '_')}"
            arrays[f"{key}_beta"] = fitted["age_beta"]
            arrays[f"{key}_p"] = fitted["p_value"]
            arrays[f"{key}_fdr"] = fitted["fdr"]
            metrics = recovery_metrics(fitted["age_beta"], oracle["age_beta"])
            metrics.update(
                tissue=int(tissue), tissue_name=tissue_name,
                missing_rate=float(rate), repeat=int(repeat), arm=arm,
                n_analysis=n, n_masked=int(masked.sum()),
                oracle_fdr05=int(np.sum(oracle["fdr"] < 0.05)),
                arm_fdr05=int(np.sum(fitted["fdr"] < 0.05)),
            )
            summary_rows.append(metrics)
    return pd.DataFrame(summary_rows), arrays, masks
