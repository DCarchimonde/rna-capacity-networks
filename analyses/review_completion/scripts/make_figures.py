from pathlib import Path
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT.parents[1]
O = ROOT / "results"
F = P / "figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.titlesize": 10,
    }
)
colors = ["#27647B", "#C2663B", "#648D58", "#745391", "#555B65"]


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(F / (name + "." + ext), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label(ax, s):
    ax.text(-0.15, 1.08, s, transform=ax.transAxes, fontweight="bold", fontsize=12)


s = pd.read_csv(O / "sparse_truth.csv")
power = pd.read_csv(O / "high_order_detection_power.csv")
bg = pd.read_csv(O / "background_only_truth.csv")
tie = pd.read_csv(O / "near_tie_bias.csv")
fig, axs = plt.subplots(2, 2, figsize=(8, 6.5), layout="constrained")
ax = axs[0, 0]
for k, c in zip([8, 128, 500], colors):
    d = s[s.k == k]
    ax.loglog(d.scale, d.difference, "o-", c=c, label=f"k = {k}")
ax.set(
    xlabel="Nonzero baseline gene probability s",
    ylabel=r"True $R_{k,A}-R_{k,B}$",
    title="Matched means and second moments",
)
ax.legend(frameon=False)
label(ax, "A")
ax = axs[0, 1]
for scale, c in zip([1e-4, 3e-4, 1e-3], colors):
    d = power[power.scale == scale]
    ax.errorbar(
        d.cells,
        d.empirical_power,
        yerr=[d.empirical_power - d.power_ci_low, d.power_ci_high - d.empirical_power],
        fmt="o-",
        capsize=2,
        c=c,
        label=f"s = {scale:g}",
    )
ax.axhline(0.05, color=".5", ls="--", lw=0.8)
ax.set(
    xscale="log",
    xlabel="Cells per population",
    ylabel="Two-sided detection power",
    ylim=(0, 0.25),
    title="Independent null calibration, k = 500",
)
ax.legend(frameon=False)
label(ax, "B")
ax = axs[1, 0]
for k, c in zip([2, 8, 32, 128, 500], colors):
    d = bg[bg.k == k]
    ax.semilogx(d.rho, d.R, c=c, label=f"k = {k}")
ax.axhline(
    float(
        pd.read_csv(O / "panel_background_invariant_truth.csv")
        .query("panel_resolution==8")
        .R.iloc[0]
    ),
    color="black",
    lw=1,
    ls=":",
    label="Panel-only n = 8",
)
ax.set(
    xlabel=r"Graph RNA fraction $\rho$",
    ylabel="Population score",
    title="Change background, keep panel composition",
)
ax.legend(frameon=False, ncol=2)
label(ax, "C")
ax = axs[1, 1]
for delta, c in zip([0, 0.02, 0.1], colors):
    d = tie[(tie.depth == 10000) & (tie.delta == delta)]
    ax.semilogx(d.cells, d.ratio_bias, "o-", c=c, label=f"Edge gap parameter = {delta:g}")
ax.axhline(0, color=".5", lw=0.8)
ax.set(
    xlabel="Cells",
    ylabel="Mean estimated score minus truth",
    title="Near-tied deletion effects, L = 10,000",
)
ax.legend(frameon=False)
label(ax, "D")
save(fig, "Figure_5")
# Expanded cohort: actual score differences and held-out biological comparison.
scores = pd.read_csv(O / "expanded_graph_scores.csv")
w = scores[scores.graph == "integrin26"].pivot(index="patient", columns="k", values="R")
meta = pd.read_csv(O / "expanded_patient_features.csv").set_index("patient").loc[w.index]
cv = (
    pd.read_csv(O / "MMR_fixed_model_cv.csv")
    .query('cohort=="expanded62" and graph=="integrin26"')
    .set_index("model")
)
fig = plt.figure(figsize=(8, 6.8), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
for status, c in zip(["MMRp", "MMRd"], colors):
    m = meta.MMR == status
    ax.scatter(w.loc[m, 2], w.loc[m, 500], label=f"{status} (n = {m.sum()})", c=c, s=25, alpha=0.85)
ax.plot([0.4, 0.8], [0.4, 0.8], color=".5", ls="--", lw=0.8)
ax.set(
    xlabel=r"$\hat R_2$",
    ylabel=r"$\hat R_{500}$",
    title="Full integrin family: 62 patients",
    xlim=(0.4, 0.8),
    ylim=(0.4, 0.8),
)
ax.legend(frameon=False)
label(ax, "A")
ax = fig.add_subplot(gs[0, 1])
data = [w.loc[meta.MMR == s, k] for k in [2, 500] for s in ["MMRp", "MMRd"]]
bp = ax.boxplot(
    data,
    positions=[0, 1, 3, 4],
    widths=0.6,
    showfliers=False,
    patch_artist=True,
    medianprops={"color": "black"},
)
for box, c in zip(bp["boxes"], [colors[0], colors[1]] * 2):
    box.set_facecolor(c)
    box.set_alpha(0.35)
rng = np.random.default_rng(26101883)
for i, v in zip([0, 1, 3, 4], data):
    ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=8, c=".35", alpha=0.55)
ax.set(
    xticks=[0.5, 3.5],
    xticklabels=["k = 2", "k = 500"],
    ylabel="Patient score",
    title="MMR groups overlap",
)
label(ax, "B")
ax = fig.add_subplot(gs[1, :])
names = ["mean_profile", "R2", "R500", "clinical_mean_R2", "clinical_mean_R2_plus_delta"]
labels = [
    "Mean gene\nprofile",
    "R2",
    "R500",
    "Clinical + mean\nprofile + R2",
    "Clinical + mean\nprofile + R2 + delta",
]
ax.bar(
    np.arange(5),
    cv.loc[names, "loo_log_loss"],
    color=[colors[4], colors[0], colors[1], colors[2], colors[3]],
    width=0.65,
)
ax.set(
    xticks=np.arange(5),
    xticklabels=labels,
    ylabel="Leave-one-patient-out log loss",
    ylim=(0, 0.85),
    title="Lower prediction loss is better",
)
for i, v in enumerate(cv.loc[names, "loo_log_loss"]):
    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
label(ax, "C")
save(fig, "Figure_6")
# Separate uncertainty display avoids stacking 23 overlapping ribbons.
d = pd.read_csv(O / "patient_score_intervals.csv")
fig, axs = plt.subplots(1, 2, figsize=(8, 8.3), layout="constrained", sharex=True)
for ax, co in zip(axs, ["D", "I"]):
    dd = d[d.cohort == co]
    order = dd[dd.k == 2].sort_values("R").patient.tolist()
    pos = {p: i for i, p in enumerate(order)}
    for k, shift, c in [(2, -0.13, colors[0]), (500, 0.13, colors[1])]:
        q = dd[dd.k == k].set_index("patient").loc[order]
        y = np.arange(len(order)) + shift
        ax.errorbar(
            q.R,
            y,
            xerr=[q.R - q.joint_component_95_low, q.joint_component_95_high - q.R],
            fmt="o",
            ms=3,
            elinewidth=0.8,
            c=c,
            label=f"k = {k}",
        )
    ax.set(
        yticks=np.arange(len(order)),
        yticklabels=order,
        xlabel="Patient score and projected 95% interval",
        xlim=(0, 1),
        title=f"{co}: within-patient cell bootstrap",
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False)
save(fig, "Figure_S3")
# All 26 vertices and all 24 curated edges, drawn explicitly.
fig, ax = plt.subplots(figsize=(8, 7))
pos = {
    "ITGB1": (0, 0),
    "ITGAV": (1.6, 1.9),
    "ITGA4": (1.6, -0.6),
    "ITGA6": (1.6, -1.7),
    "ITGB3": (3.6, 2.8),
    "ITGA2B": (5.3, 2.8),
    "ITGB5": (3.6, 1.9),
    "ITGB6": (3.6, 1.0),
    "ITGB8": (3.6, 0.1),
    "ITGB7": (3.6, -0.9),
    "ITGAE": (5.3, -0.9),
    "ITGB4": (3.6, -1.9),
    "ITGB2": (-0.1, -4.3),
    "ITGAL": (-2.2, -3.2),
    "ITGAM": (-2.2, -4.3),
    "ITGAX": (-2.2, -5.4),
    "ITGAD": (1.8, -4.3),
}
for gene, y in zip(
    ["ITGA1", "ITGA2", "ITGA3", "ITGA5", "ITGA7", "ITGA8", "ITGA9", "ITGA10", "ITGA11"],
    np.linspace(2.4, -2.4, 9),
):
    pos[gene] = (-2.2, y)
ed = pd.read_csv(ROOT / "inputs/integrin_edges_with_provenance.csv")
for a, b in ed[["alpha", "beta"]].itertuples(index=False, name=None):
    ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], c=".65", lw=1.2, zorder=1)
for g, (x, y) in pos.items():
    ax.text(
        x,
        y,
        g,
        ha="center",
        va="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=.3",
            "fc": "#E5EFF3" if g.startswith("ITGA") else "#F4E4DB",
            "ec": colors[0] if g.startswith("ITGA") else colors[1],
        },
        zorder=3,
    )
ax.set(xlim=(-3.1, 6), ylim=(-6.1, 3.6))
ax.axis("off")
ax.set_title("Complete integrin family: 26 genes, 24 heterodimers, two tree components", pad=15)
save(fig, "Figure_S2")
# Near-tie uncertainty coverage: compare the estimator, not only the narrative.
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
for ax, L in zip(axs, [1000, 10000]):
    q = tie[(tie.depth == L) & (tie.cells == 3000)]
    ax.plot(
        q.delta, q.percentile_interval_coverage, "o-", c=colors[1], label="Direct percentile ratio"
    )
    ax.plot(
        q.delta,
        q.joint_component_projection_coverage,
        "s-",
        c=colors[0],
        label="Joint component projection",
    )
    ax.axhline(0.95, ls="--", c=".5", lw=0.8)
    ax.set(
        xlabel="True edge imbalance parameter",
        ylabel="Empirical 95% interval coverage",
        ylim=(-0.03, 1.05),
        title=f"L = {L:,}; 3,000 cells",
    )
    ax.legend(frameon=False)
save(fig, "Figure_S1")
print("figures written", flush=True)
