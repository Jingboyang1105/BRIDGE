from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MISS = HERE / "missingness"
DEMO = HERE / "demographics"
OUT = HERE / "combined_final"
OUT.mkdir(exist_ok=True)

OBSERVED = "#6757D9"
MISSING = "#E9E6F7"
TRAIN = "#5F84AD"
VALIDATION = "#D9A441"
TEST = "#62A88E"
FEMALE = "#C77A86"
MALE = "#5F84AD"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def clean_axis(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


donor = pd.read_csv(MISS / "donor_missingness_descending_source.csv")
tissue = pd.read_csv(MISS / "tissue_missingness_ascending_source.csv")
demo = pd.read_csv(DEMO / "seed42_donor_demographics_source.csv").drop_duplicates("donor")
sex = pd.read_csv(DEMO / "seed42_sex_composition_source.csv")

fig = plt.figure(figsize=(12.0, 14.0), constrained_layout=False)
gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], hspace=0.25, wspace=0.28,
                      left=0.075, right=0.98, top=0.975, bottom=0.065)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# a, donor-level tissue availability.
y = np.arange(len(donor))
observed = donor["observed_count"].to_numpy()
total = int(donor["total_tissues"].iloc[0])
ax_a.barh(y, observed, color=OBSERVED, height=1.0, linewidth=0)
ax_a.barh(y, total - observed, left=observed, color=MISSING, height=1.0, linewidth=0)
ax_a.invert_yaxis()
ax_a.set_yticks([])
ax_a.set_xlim(0, total)
ax_a.set_xticks([0, 10, 20, 30, 40, 50, 60, 65])
ax_a.set_xlabel("Number of tissues per donor")
ax_a.set_ylabel(f"Training donors (n={len(donor)})")
median = float(np.median(observed))
ax_a.axvline(median, color="#555555", linestyle="--", linewidth=1.1)
ax_a.text(median + 1.0, 0.985, f"Median = {median:.0f}",
          transform=ax_a.get_xaxis_transform(), ha="left", va="top", color="#555555")
ax_a.legend(handles=[Patch(color=OBSERVED, label="Observed tissues"),
                     Patch(color=MISSING, label="Missing tissues")],
            frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.015))
ax_a.spines[["top", "right", "left"]].set_visible(False)
ax_a.tick_params(axis="y", length=0)

# b, tissue-level availability. Extra vertical space preserves all 65 labels.
y = np.arange(len(tissue))
observed_fraction = 1.0 - tissue["missing_rate"].to_numpy()
ax_b.barh(y, observed_fraction, color=OBSERVED, height=0.82, linewidth=0)
ax_b.barh(y, tissue["missing_rate"].to_numpy(), left=observed_fraction,
          color=MISSING, height=0.82, linewidth=0)
ax_b.set_yticks(y, tissue["tissue"].tolist(), fontsize=6.3)
ax_b.invert_yaxis()
ax_b.set_xlim(0, 1)
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax_b.set_xlabel("Proportion of donors")
ax_b.set_ylabel(f"Tissues (n={len(tissue)})", labelpad=8)
ax_b.legend(handles=[Patch(color=OBSERVED, label="Observed"),
                     Patch(color=MISSING, label="Missing")],
            frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.015))
ax_b.spines[["top", "right", "left"]].set_visible(False)
ax_b.tick_params(axis="y", length=0, pad=2)

# c, age-bin composition by donor split.
split_order = ["Train", "Validation", "Test"]
age_groups = list(range(6))
age_counts = (demo.groupby(["split", "age_idx"]).size()
              .unstack(fill_value=0).reindex(index=split_order, columns=age_groups, fill_value=0))
age_fraction = age_counts.div(age_counts.sum(axis=1), axis=0)
x = np.arange(len(age_groups))
width = 0.24
split_colors = [TRAIN, VALIDATION, TEST]
for offset, split_name, color in zip([-width, 0, width], split_order, split_colors):
    n = int(age_counts.loc[split_name].sum())
    ax_c.bar(x + offset, age_fraction.loc[split_name], width=width,
             color=color, label=f"{split_name} (n={n})")
ax_c.set_xticks(x, age_groups)
ax_c.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax_c.set_ylim(0, 0.38)
ax_c.set_xlabel("Encoded age group")
ax_c.set_ylabel("Donor fraction")
ax_c.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
clean_axis(ax_c)

# d, sex composition by donor split.
pivot_n = (sex.pivot(index="split", columns="sex_label", values="n")
           .reindex(split_order))
pivot_f = (sex.pivot(index="split", columns="sex_label", values="fraction")
           .reindex(split_order))
x = np.arange(len(split_order))
female = pivot_f["Female"].to_numpy()
male = pivot_f["Male"].to_numpy()
ax_d.bar(x, female, width=0.58, color=FEMALE, label="Female")
ax_d.bar(x, male, bottom=female, width=0.58, color=MALE, label="Male")
for i, split_name in enumerate(split_order):
    ax_d.text(i, female[i] / 2, str(int(pivot_n.loc[split_name, "Female"])),
              ha="center", va="center", color="white", fontsize=10)
    ax_d.text(i, female[i] + male[i] / 2, str(int(pivot_n.loc[split_name, "Male"])),
              ha="center", va="center", color="white", fontsize=10)
totals = pivot_n.sum(axis=1).astype(int)
ax_d.set_xticks(x, [f"{s}\nn={totals.loc[s]}" for s in split_order])
ax_d.set_ylim(0, 1)
ax_d.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax_d.set_ylabel("Donor fraction")
ax_d.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
clean_axis(ax_d)

for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
    ax.text(-0.13, 1.045, label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="left", va="top")

stem = OUT / "Supplementary_Figure_S1_GTEx_Tissue_Availability_and_Demographic_Composition"
for ext in ["svg", "pdf", "png", "tiff"]:
    kwargs = {"dpi": 600} if ext in {"png", "tiff"} else {}
    fig.savefig(stem.with_suffix("." + ext), bbox_inches="tight", facecolor="white", **kwargs)
plt.close(fig)
print(stem)
