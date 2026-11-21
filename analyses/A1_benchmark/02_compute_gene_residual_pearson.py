"""Build tissue-stratified gene-level residual Pearson result tables.

Within one target tissue, subtracting the training tissue mean is a
gene-specific constant translation. Pearson correlation across donors is
therefore unchanged. The formal per-tissue gene correlations can be relabelled
as residual-gene correlations and macro-averaged across target tissues without
mixing tissue-level expression differences into the statistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "02_Benchmark_vs_HYFA_PMM"
ACCESSIBLE_TEEBOT = ROOT / "09_Accessible_TEEBoT_Benchmark"
OUTPUT = BENCHMARK / "results" / "final_tables"


HIDE_ONE_METHODS = {
    "two_expert_cml_gate": "DORIAN",
    "same_donor_residual": "Same-donor residual expert",
    "cml_retrieval": "Memory retrieval expert",
    "hyfa_hide_one_test": "HYFA",
    "PMM_mice_m1_seed42": "PMM",
    "target_tissue_mean": "Target-tissue mean",
}

CONTROL_METHODS = {
    "no_target_identity": "DORIAN without target-tissue identity",
    "no_any_identity": "DORIAN without any tissue identity",
    "demographic_two_expert_gate": "Age/sex control mixer",
    "v2_age_sex_top10": "Age/sex top-10 only",
}

ACCESSIBLE_METHODS = {
    "DORIAN_accessible": "DORIAN",
    "HYFA_accessible": "HYFA",
    "TEEBoT_accessible_age_sex": "TEEBoT (age/sex)",
    "TEEBoT_accessible_expression_only": "TEEBoT (expression-only)",
    "PMM_mice_m1_seed42": "PMM",
    "target_tissue_mean": "Target-tissue mean",
}

METHOD_ORDER = [
    "DORIAN",
    "DORIAN without target-tissue identity",
    "DORIAN without any tissue identity",
    "Age/sex control mixer",
    "Age/sex top-10 only",
    "HYFA",
    "PMM",
    "TEEBoT (age/sex)",
    "TEEBoT (expression-only)",
    "Same-donor residual expert",
    "Memory retrieval expert",
    "Target-tissue mean",
]


def _paired_pearson_by_gene(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("truth and prediction must be two-dimensional arrays with equal shape")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.sum(x * y, axis=0)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return np.where(denominator > 0, values, np.nan)


def gene_residual_pearson(
    truth: np.ndarray, prediction: np.ndarray, tissue_mean: np.ndarray
) -> np.ndarray:
    true = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    mean = np.asarray(tissue_mean, dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("truth and prediction must share a two-dimensional shape")
    if mean.shape not in {(true.shape[1],), true.shape}:
        raise ValueError("tissue mean must contain one value per gene or match the data shape")
    return _paired_pearson_by_gene(true - mean, pred - mean)


def macro_summary(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, method), group in source.groupby(["task", "method"], sort=False):
        values = pd.to_numeric(group["gene_residual_pearson_mean"], errors="coerce")
        valid = values[np.isfinite(values)]
        rows.append(
            {
                "task": task,
                "method": method,
                "gene_residual_pearson": float(valid.mean()) if len(valid) else np.nan,
                "median_tissue_gene_residual_pearson": (
                    float(valid.median()) if len(valid) else np.nan
                ),
                "n_evaluable_tissues": int(len(valid)),
                "n_total_tissues": int(group["target_tissue"].nunique()),
                "n_test_records": int(group["n_samples"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["_method_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    out["_task_order"] = out["task"].map({"hide_one": 0, "accessible": 1})
    return (
        out.sort_values(["_task_order", "_method_order"])
        .drop(columns=["_task_order", "_method_order"])
        .reset_index(drop=True)
    )


def _standardize(
    frame: pd.DataFrame,
    task: str,
    method_map: dict[str, str],
    value_column: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "task": task,
            "method": frame["method"].map(method_map),
            "target_tissue": pd.to_numeric(
                frame.get("target_tissue", frame.get("target_tissue_id")), errors="raise"
            ).astype(int),
            "target_tissue_name": frame["target_tissue_name"].astype(str),
            "n_samples": pd.to_numeric(frame["n_samples"], errors="raise").astype(int),
            "gene_residual_pearson_mean": pd.to_numeric(
                frame[value_column], errors="coerce"
            ),
        }
    )
    if out["method"].isna().any():
        unknown = sorted(frame.loc[out["method"].isna(), "method"].astype(str).unique())
        raise ValueError(f"unmapped methods: {unknown}")
    return out


def build_hide_one() -> pd.DataFrame:
    final = BENCHMARK / "results" / "final_tables"
    controls = BENCHMARK / "results" / "formal_control_per_tissue"
    pmm = BENCHMARK / "results" / "pmm_v11_seed42" / "formal_olga" / "hide_one"
    dorian = pd.read_csv(final / "dorian_hide_one_per_tissue_metrics.csv")
    hyfa = pd.read_csv(final / "hyfa_hide_one_per_tissue_metrics.csv")
    pmm_frame = pd.read_csv(pmm / "pmm_per_tissue_metrics.csv")

    tissue_names = pmm_frame.set_index("target_tissue")["target_tissue_name"].to_dict()
    dorian["target_tissue_name"] = dorian["target_tissue"].map(tissue_names)
    hyfa["target_tissue_name"] = hyfa["target_tissue"].map(tissue_names)

    no_target = pd.read_csv(controls / "no_target_identity_per_tissue_metrics.csv")
    no_target = no_target.loc[no_target["method"] == "two_expert_cml_gate"].copy()
    no_target["method"] = "no_target_identity"
    no_target["target_tissue_name"] = no_target["target_tissue"].map(tissue_names)

    no_any = pd.read_csv(controls / "no_any_identity_per_tissue_metrics.csv")
    no_any = no_any.loc[no_any["method"] == "two_expert_cml_gate"].copy()
    no_any["method"] = "no_any_identity"
    no_any["target_tissue_name"] = no_any["target_tissue"].map(tissue_names)

    demographic = pd.read_csv(controls / "demographic_controls_per_tissue_metrics.csv")
    demographic = demographic.loc[
        demographic["method"].isin(["demographic_two_expert_gate", "v2_age_sex_top10"])
    ].copy()

    frames = [
        _standardize(dorian, "hide_one", HIDE_ONE_METHODS, "hyfa_gene_pearson_mean"),
        _standardize(hyfa, "hide_one", HIDE_ONE_METHODS, "hyfa_gene_pearson_mean"),
        _standardize(pmm_frame, "hide_one", HIDE_ONE_METHODS, "residual_gene_pearson_mean"),
        _standardize(no_target, "hide_one", CONTROL_METHODS, "hyfa_gene_pearson_mean"),
        _standardize(no_any, "hide_one", CONTROL_METHODS, "hyfa_gene_pearson_mean"),
        _standardize(demographic, "hide_one", CONTROL_METHODS, "hyfa_gene_pearson_mean"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out.loc[out["method"] == "Target-tissue mean", "gene_residual_pearson_mean"] = np.nan
    return out.sort_values(["target_tissue_name", "method"]).reset_index(drop=True)


def build_accessible() -> pd.DataFrame:
    base = pd.read_csv(
        ACCESSIBLE_TEEBOT / "results" / "tables" / "Table_accessible_per_tissue_metrics.csv"
    )
    pmm = pd.read_csv(
        BENCHMARK
        / "results"
        / "pmm_v11_seed42"
        / "formal_olga"
        / "accessible"
        / "pmm_per_tissue_metrics.csv"
    )
    frames = [
        _standardize(base, "accessible", ACCESSIBLE_METHODS, "gene_pearson_mean"),
        _standardize(pmm, "accessible", ACCESSIBLE_METHODS, "residual_gene_pearson_mean"),
    ]
    out = pd.concat(frames, ignore_index=True)
    out.loc[out["method"] == "Target-tissue mean", "gene_residual_pearson_mean"] = np.nan
    return out.sort_values(["target_tissue_name", "method"]).reset_index(drop=True)


def main() -> None:
    hide_one = build_hide_one()
    accessible = build_accessible()
    overall = macro_summary(pd.concat([hide_one, accessible], ignore_index=True))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hide_one.to_csv(OUTPUT / "gene_residual_pearson_hide_one_per_tissue.csv", index=False)
    accessible.to_csv(OUTPUT / "gene_residual_pearson_accessible_per_tissue.csv", index=False)
    overall.to_csv(OUTPUT / "gene_residual_pearson_overall.csv", index=False)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
