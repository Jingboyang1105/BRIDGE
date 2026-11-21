from __future__ import annotations

import numpy as np


def recovery_metrics(method_beta, oracle_beta, top_ks=(100, 500, 1000)) -> dict[str, float]:
    method = np.asarray(method_beta, dtype=float)
    oracle = np.asarray(oracle_beta, dtype=float)
    valid = np.isfinite(method) & np.isfinite(oracle)
    method = method[valid]
    oracle = oracle[valid]
    if method.size < 3:
        raise ValueError("at least three finite genes are required")
    slope = np.polyfit(oracle, method, 1)[0]
    result = {
        "n_genes": int(method.size),
        "beta_pearson": float(np.corrcoef(method, oracle)[0, 1]),
        "beta_slope": float(slope),
        "beta_mae": float(np.mean(np.abs(method - oracle))),
        "sign_agreement": float(np.mean(np.sign(method) == np.sign(oracle))),
    }
    for k in top_ks:
        effective = min(int(k), method.size)
        oracle_top = set(np.argpartition(np.abs(oracle), -effective)[-effective:])
        method_top = set(np.argpartition(np.abs(method), -effective)[-effective:])
        result[f"top_{k}_recall"] = len(oracle_top & method_top) / effective
    return result
