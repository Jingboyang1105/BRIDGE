"""Plot age and correctly decoded sex composition for the seed-42 donor split."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "seed42_donor_demographics_source.csv"
SPLITS = ["Train", "Validation", "Test"]
AGE_LABELS = ["20–29", "30–39", "40–49", "50–59", "60–69", "70–79"]
SPLIT_COLORS = ["#5F84AD", "#D9A441", "#62A88E"]
SEX_LABELS = {0: "Male", 1: "Female"}
FEMALE = "#C77A86"
MALE = "#5F84AD"


def main() -> None:
    frame = pd.read_csv(INPUT).drop_duplicates("donor")
    if set(frame["split"]) != set(SPLITS):
        raise ValueError("Expected Train, Validation, and Test donor splits")
    if not frame["age_idx"].isin(range(6)).all():
        raise ValueError("Expected six encoded GTEx age groups")
    if not frame["sex_idx"].isin(SEX_LABELS).all():
        raise ValueError("Expected GTEx sex_idx values 0 (Male) and 1 (Female)")

    expected_labels = frame["sex_idx"].map(SEX_LABELS)
    if not frame["sex_label"].equals(expected_labels):
        raise ValueError("sex_label disagrees with GTEx-derived sex_idx (0 = Male, 1 = Female)")
    sex_source = (
        frame.groupby(["split", "sex_label"]).size().rename("n").reset_index()
    )
    totals = frame.groupby("split")["donor"].nunique()
    sex_source["split_total"] = sex_source["split"].map(totals)
    sex_source["fraction"] = sex_source["n"] / sex_source["split_total"]
    sex_source.to_csv(HERE / "seed42_sex_composition_source.csv", index=False)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })

    fig, (ax_age, ax_sex) = plt.subplots(
        1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )

    x_age = np.arange(6)
    width = 0.24
    for offset, split, color in zip([-width, 0, width], SPLITS, SPLIT_COLORS):
        subset = frame.loc[frame["split"] == split]
        counts = subset["age_idx"].value_counts().reindex(range(6), fill_value=0)
        ax_age.bar(
            x_age + offset,
            counts.to_numpy() / len(subset),
            width=width,
            color=color,
            label=f"{split} (n={len(subset)})",
        )
    ax_age.set_xticks(x_age, AGE_LABELS)
    ax_age.set_xlabel("Age group (years)")
    ax_age.set_ylabel("Donor fraction")
    ax_age.set_ylim(0, 0.40)
    ax_age.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_age.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    pivot_n = (
        sex_source.pivot(index="split", columns="sex_label", values="n")
        .reindex(index=SPLITS, columns=["Female", "Male"], fill_value=0)
    )
    pivot_f = (
        sex_source.pivot(index="split", columns="sex_label", values="fraction")
        .reindex(index=SPLITS, columns=["Female", "Male"], fill_value=0)
    )
    x_sex = np.arange(len(SPLITS))
    female = pivot_f["Female"].to_numpy()
    male = pivot_f["Male"].to_numpy()
    ax_sex.bar(x_sex, female, width=0.58, color=FEMALE, label="Female")
    ax_sex.bar(x_sex, male, bottom=female, width=0.58, color=MALE, label="Male")
    for i, split in enumerate(SPLITS):
        ax_sex.text(i, female[i] / 2, str(int(pivot_n.loc[split, "Female"])),
                    ha="center", va="center", color="white")
        ax_sex.text(i, female[i] + male[i] / 2, str(int(pivot_n.loc[split, "Male"])),
                    ha="center", va="center", color="white")
    split_totals = pivot_n.sum(axis=1).astype(int)
    ax_sex.set_xticks(x_sex, [f"{split}\nn={split_totals.loc[split]}" for split in SPLITS])
    ax_sex.set_ylim(0, 1)
    ax_sex.set_ylabel("Donor fraction")
    ax_sex.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_sex.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    for label, ax in zip("ab", [ax_age, ax_sex]):
        ax.grid(axis="y", color="#D9DDE3", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="top")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.92, wspace=0.22)
    base = HERE / "seed42_train_val_test_age_sex_distribution_no_title"
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
