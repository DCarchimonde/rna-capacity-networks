from pathlib import Path
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
F = P / "figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
TEAL = "#166978"
ORANGE = "#bb632f"


def save(fig, name):
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(F / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


fig, axs = plt.subplots(
    1, 2, figsize=(7.4, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1.2]}
)
a = axs[0]
df = pd.read_csv(P / "review/native_CSCORE_common_converged.csv")
for method, label, col, off in [
    ("equal_cell_moment", "Equal-cell moment", TEAL, -0.13),
    ("CSCORE_graph_raw", "Native CS-CORE", ORANGE, 0.13),
]:
    for i, co in enumerate(["D", "I"]):
        r = df[(df.cohort == co) & (df.method == method)].iloc[0]
        v = 100 * r.retention
        a.errorbar(
            i + off,
            v,
            yerr=[[v - 100 * r.patient_95_low], [100 * r.patient_95_high - v]],
            fmt="o",
            color=col,
            capsize=3,
            label=label if i == 0 else None,
        )
        a.annotate(
            f"{v:.1f}",
            (i + off, v),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=col,
        )
a.set(
    xticks=[0, 1],
    xticklabels=["D: 21–23 patients", "I: 9–12 patients"],
    xlim=(-0.5, 1.5),
    ylim=(35, 103),
    ylabel="Split direction retention (%)",
    title="A  Common converged patients",
)
a.tick_params(axis="x", labelsize=8)
a.legend(frameon=False, loc="lower left", fontsize=8)
a.grid(axis="y", alpha=0.15)
a = axs[1]
df = pd.read_csv(P / "data/R8_simulation_summary.csv")
cc = [
    ("mean_matched_joint", 64),
    ("mean_matched_joint", 256),
    ("second_moment_matched", 64),
    ("second_moment_matched", 256),
]
for method, label, col, off in [
    ("exact_conditional", "Conditional", TEAL, -0.11),
    ("exact_multinomial_plugin", "Plug-in", ORANGE, 0.11),
]:
    vals = [
        100
        * df[(df.family == f) & (df.base_depth == d) & (df.method == method)]
        .iloc[0]
        .R8_pair_accuracy
        for f, d in cc
    ]
    a.plot(vals, np.arange(4) + off, "o", ms=5, c=col, label=label)
a.set(
    yticks=range(4),
    yticklabels=[
        "Matched means\nL = 64",
        "Matched means\nL = 256",
        "Matched second moments\nL = 64",
        "Matched second moments\nL = 256",
    ],
    ylim=(3.6, -0.6),
    xlim=(48, 100),
    xlabel="True R8 direction accuracy (%)",
    title="B  Same fixed-resolution target",
)
a.tick_params(axis="y", labelsize=8)
a.grid(axis="x", alpha=0.15)
a.legend(frameon=False, loc="lower right", fontsize=8)
save(fig, "Figure_4")
fig, a = plt.subplots(figsize=(7.3, 3.7), layout="constrained")
df = pd.read_csv(P / "data/H_simulation_paired_comparison.csv", keep_default_na=False)
cases = [
    (
        "poisson",
        "independent_variable",
        "coupling_only",
        "Poisson; independent depth\nCoupling only",
    ),
    ("poisson", "independent_variable", "opposing", "Poisson; independent depth\nOpposing changes"),
    (
        "multinomial",
        "state_dependent",
        "coupling_only",
        "Multinomial; state-linked depth\nCoupling only",
    ),
    (
        "multinomial",
        "state_dependent",
        "opposing",
        "Multinomial; state-linked depth\nOpposing changes",
    ),
    ("multinomial", "state_dependent", "mean_only", "Multinomial; state-linked depth\nMean only"),
]
for i, (law, depth, fam, label) in enumerate(cases):
    r = df[
        (df.source == law)
        & (df.depth_design == depth)
        & (df.family == fam)
        & (df.comparator == "CSCORE_graph_fallback")
        & (df.endpoint == "strict_accuracy")
    ].iloc[0]
    v = 100 * r.equal_cell_minus_comparator
    a.errorbar(
        v,
        i,
        xerr=[[v - 100 * r.MC_95_low], [100 * r.MC_95_high - v]],
        fmt="o",
        c=TEAL if v >= 0 else ORANGE,
        capsize=3,
    )
a.axvline(0, c=".5", lw=0.8)
a.set(
    yticks=range(5),
    yticklabels=[c[-1] for c in cases],
    xlim=(-11, 27),
    xlabel="Equal-cell minus hybrid CS-CORE accuracy (percentage points)",
    title="Hybrid sensitivity analysis with failed-fit replacement",
)
a.invert_yaxis()
a.grid(axis="x", alpha=0.15)
save(fig, "Figure_S4")
print("Figure 4 and Supplementary Figure S4 generated")
