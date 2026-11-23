from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "all_eligible_formal_seed42"
INPUT = RESULTS / "age_recovery_endpoints.csv"
OUT = RESULTS / "figures"

METHODS = ["BRIDGE", "HYFA", "PMM", "Observed-only", "Mean-fill"]
COLORS = {
    "BRIDGE": "#6C5CE7",
    "HYFA": "#F2B84B",
    "PMM": "#E66B9A",
    "Observed-only": "#777777",
    "Mean-fill": "#35A7A0",
}
LINESTYLES = {
    "BRIDGE": "-",
    "HYFA": "-",
    "PMM": "-",
    "Observed-only": "--",
    "Mean-fill": "--",
}


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame = frame.loc[frame["arm"].isin(METHODS)].copy()

    # Five masks are repeated measurements within a tissue. Average them first,
    # then quantify uncertainty across the 49 tissues.
    tissue_means = (
        frame.groupby(["arm", "missing_rate", "tissue", "tissue_name"], as_index=False)
        ["beta_pearson"]
        .mean()
    )

    rows: list[dict[str, float | str | int]] = []
    for method_index, method in enumerate(METHODS):
        for rate_index, rate in enumerate(sorted(tissue_means["missing_rate"].unique())):
            values = tissue_means.loc[
                (tissue_means["arm"] == method)
                & (tissue_means["missing_rate"] == rate),
                "beta_pearson",
            ].to_numpy(dtype=float)
            if values.size != 49 or not np.isfinite(values).all():
                raise ValueError(f"Unexpected tissue coverage for {method}, {rate}: {values.size}")
            low, high = bootstrap_ci(values, seed=42 + method_index * 10 + rate_index)
            rows.append(
                {
                    "method": method,
                    "missing_rate": float(rate),
                    "mean_beta_pearson": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_tissues": int(values.size),
                    "mask_repeats_per_tissue": 5,
                }
            )

    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "age_beta_pearson_missing_rate_curve_source_data.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(4.65, 3.35))
    x = np.array([20, 40, 60, 80], dtype=float)
    for method in METHODS:
        part = summary.loc[summary["method"] == method].sort_values("missing_rate")
        y = part["mean_beta_pearson"].to_numpy()
        lo = part["ci95_low"].to_numpy()
        hi = part["ci95_high"].to_numpy()
        primary = method in {"BRIDGE", "HYFA", "PMM"}
        ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.13 if primary else 0.08, linewidth=0)
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.25 if primary else 1.55,
            marker="o" if method in {"BRIDGE", "HYFA"} else ("D" if method == "PMM" else "s"),
            markersize=5.2 if primary else 4.1,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=method,
            zorder=3 if primary else 2,
        )

    ax.set_xlabel("Masked target-tissue profiles")
    ax.set_ylabel("Age-beta Pearson correlation")
    ax.set_xticks(x, [f"{int(v)}%" for v in x])
    ax.set_ylim(0.00, 1.00)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.2, width=0.75)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=0.95,
    )

    fig.tight_layout(pad=0.7)
    stem = OUT / "age_beta_pearson_missing_rate_curve"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
