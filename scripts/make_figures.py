from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "analyses/molecular_resolution"
C = ROOT / "analyses/strong_comparators"
OUT = ROOT
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "legend.fontsize": 7.5,
    }
)
TEAL = "#166978"
ORANGE = "#bb632f"
PURPLE = "#7866a6"
GRAY = "#aab6c2"
DATA = ROOT / "data"
(OUT / "figures").mkdir(exist_ok=True)


def save(fig, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / "figures" / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(
        1, 2, figsize=(6.8, 3.15), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]}
    )
    pos = {0: (0, 1.45), 1: (0, 0.05), 2: (1.65, 1.55), 3: (1.65, 0.77), 4: (1.65, -0.02)}
    for a, b in [(0, 2), (1, 2), (1, 3), (1, 4)]:
        ax[0].plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#8da3ae", lw=2, zorder=1)
    for i, name in enumerate(["ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGB6"]):
        ax[0].scatter(*pos[i], s=1000, color=TEAL if i < 2 else ORANGE, zorder=2)
        ax[0].text(*pos[i], name, color="white", ha="center", va="center", fontsize=8, zorder=3)
    ax[0].text(0.82, -0.60, r"$R_2=1-\max_e H_e\,/\,\sum_e H_e$", ha="center", fontsize=10.5)
    ax[0].text(
        0.82, -1.10, "RNA capacities are shared\nacross alternative pairs", ha="center", fontsize=8
    )
    ax[0].axis("off")
    ax[0].set_xlim(-0.5, 2.15)
    ax[0].set_ylim(-1.45, 2.0)
    ax[0].set_title("A   Selected integrin graph", loc="left")
    w = json.loads((M / "results/theory_checks.json").read_text())["second_moment_witness"]
    ks = [x["k"] for x in w if x["k"] > 1]
    for key, label, col in [("R_A", "Population A", TEAL), ("R_B", "Population B", ORANGE)]:
        ax[1].plot(ks, [x[key] for x in w if x["k"] > 1], "o-", c=col, lw=1.6, ms=4, label=label)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([2, 3, 4, 8, 16], [2, 3, 4, 8, 16])
    ax[1].set_ylim(0.585, 0.71)
    ax[1].set_xlabel("Molecules per cell k")
    ax[1].set_ylabel("Population resilience")
    ax[1].set_title("B   Matched first and second moments", loc="left", fontsize=8.2)
    ax[1].annotate(
        "Separation at k = 4",
        xy=(4, 0.6764),
        xytext=(4.4, 0.700),
        fontsize=7.5,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.01, 0.09))
    ax[1].text(0.02, 0.02, r"$R_\infty$: A = 4/7, B = 1/2", transform=ax[1].transAxes, fontsize=8)
    save(fig, "Figure_1")


def fig2():
    df = pd.read_csv(M / "results/simulation_patient_scores.csv")
    s = pd.read_csv(M / "results/simulation_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.25), layout="constrained")
    sub = df[(df.family == "identical_biology_depth_gradient") & (df.base_depth == 64)]
    a = axes[0, 0]
    for method, label, color in [
        ("exact_conditional", "Fixed k = 8", TEAL),
        ("depth_plugin", "Observed depth", ORANGE),
    ]:
        x = sub[sub.method == method].groupby("depth").R.agg(["mean", "std", "count"])
        a.errorbar(
            x.index,
            x["mean"],
            1.96 * x["std"] / np.sqrt(x["count"]),
            fmt="o-",
            ms=3,
            lw=1.3,
            color=color,
            label=label,
        )
    a.axhline(sub.true_R8.iloc[0], ls="--", c="#666", lw=0.8, label="Common true R8")
    a.set_xscale("log", base=2)
    a.set_xticks([64, 128, 256], [64, 128, 256])
    a.set_xlabel("Observed molecules per cell")
    a.set_ylabel("Mean patient score")
    a.set_title("A   Identical latent biology", loc="left")
    a.legend(frameon=False, loc="center left", fontsize=7)
    a = axes[0, 1]
    xx = np.arange(2)
    for off, method, label, col in [
        (-0.18, "exact_conditional", "Fixed k = 8", TEAL),
        (0.18, "depth_plugin", "Observed depth", ORANGE),
    ]:
        x = s[(s.family == "identical_biology_depth_gradient") & (s.method == method)].sort_values(
            "base_depth"
        )
        a.bar(xx + off, 100 * x.split_all_pair_retention, 0.33, color=col, label=label)
        for i, y in zip(xx + off, 100 * x.split_all_pair_retention):
            a.text(i, y + 2, f"{y:.1f}", ha="center", fontsize=7.7)
    a.axhline(50, ls="--", c="#666", lw=0.8)
    a.set_xticks(xx, ["64–256", "256–1024"])
    a.set_ylim(0, 90)
    a.set_xlabel("Depth range across patients")
    a.set_ylabel("Split direction retention (%)")
    a.set_title("B   Repeatable technical order", loc="left", fontsize=8.6)
    a.legend(
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=0.96,
        loc="lower right",
        fontsize=7,
    )
    for a, fam, title in [
        (axes[1, 0], "mean_matched_joint", "C   Matched means"),
        (axes[1, 1], "second_moment_matched", "D   Matched first and second moments"),
    ]:
        for off, method, label, col in [
            (-0.25, "exact_conditional", "Exact", TEAL),
            (0, "random_8", "8 subsets", "#7fb2b9"),
            (0.25, "random_1", "1 subset", GRAY),
        ]:
            x = s[(s.family == fam) & (s.method == method)].sort_values("base_depth")
            a.bar(xx + off, 100 * x.R8_pair_accuracy, 0.23, color=col, label=label)
            for i, y in zip(xx + off, 100 * x.R8_pair_accuracy):
                a.text(i, y + 2, f"{y:.1f}", ha="center", fontsize=6.8, rotation=90)
        a.axhline(50, ls="--", c="#666", lw=0.8)
        a.set_xticks(xx, ["64", "256"])
        a.set_ylim(0, 112)
        a.set_xlabel("Observed molecules per cell")
        a.set_ylabel("True R8 direction accuracy (%)")
        a.set_title(title, loc="left", fontsize=8.2)
        a.legend(
            frameon=True,
            facecolor="white",
            edgecolor="white",
            framealpha=0.96,
            loc="lower right",
            fontsize=7,
        )
    save(fig, "Figure_2")


def fig3():
    df = pd.read_csv(DATA / "patient_resolution_spectra.csv")
    ss = pd.read_csv(DATA / "resolution_cohort_summary.csv")
    fig, ax = plt.subplots(
        2, 2, figsize=(6.8, 5.8), layout="constrained", gridspec_kw={"height_ratios": [1.25, 1]}
    )
    for j, cohort in enumerate(["D", "I"]):
        sub = df[(df.cohort == cohort) & (df.graph == "common3") & (df.scope == "Epi")]
        p = sub.pivot(index="patient", columns="k", values="R_k")
        order = p[2].sort_values().index
        p = p.loc[order]
        extra = sub[sub.k == 500].set_index("patient").loc[order, "depth_varying_plugin_R"]
        values = np.column_stack([p[k] for k in [2, 8, 32, 128, 500]] + [extra])
        for r, y in enumerate(values):
            ax[0, j].plot(
                np.arange(5),
                y[:5],
                "-o",
                ms=2,
                lw=0.85,
                color=plt.cm.viridis(r / (len(values) - 1)),
                alpha=0.9,
            )
            ax[0, j].plot(
                [4, 5],
                y[4:],
                "--o",
                ms=2,
                lw=0.85,
                color=plt.cm.viridis(r / (len(values) - 1)),
                alpha=0.9,
            )
        ax[0, j].set_xticks(range(6), ["2", "8", "32", "128", "500", "Obs."])
        ax[0, j].set_xlabel("Molecular resolution")
        ax[0, j].set_ylabel("Patient graph score")
        ax[0, j].grid(axis="y", alpha=0.15)
        ax[0, j].set_title(
            ("A   GSE178341   n = 23" if j == 0 else "B   GSE166555   n = 12"),
            loc="left",
            fontsize=8.6,
        )
        z = ss[(ss.cohort == cohort) & (ss.graph == "common3")]
        for scope, label, col in [
            ("Epi", "Epithelial", TEAL),
            ("Fibro", "Fibroblast", ORANGE),
            ("balanced", "Balanced", PURPLE),
        ]:
            rr = z[z.scope == scope].sort_values("k")
            ax[1, j].plot(
                rr.k, 100 * rr.split_all_pair_retention, "o-", c=col, ms=3, lw=1.2, label=label
            )
        ax[1, j].set_xscale("log")
        ax[1, j].set_xticks([2, 8, 32, 128, 500], [2, 8, 32, 128, 500])
        ax[1, j].set_ylim(48, 73)
        ax[1, j].set_xlabel("Molecules per cell k")
        ax[1, j].set_ylabel("Split retention (%)")
        ax[1, j].set_title(
            ("C" if j == 0 else "D") + "   Complementary cell halves", loc="left", fontsize=8.2
        )
        ax[1, j].legend(frameon=False, loc="lower left", fontsize=6.8, ncol=2)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)),
        ax=ax[0, :],
        fraction=0.025,
        pad=0.02,
    )
    cb.set_label("Within-cohort R2 rank percentile", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    save(fig, "Figure_3")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
