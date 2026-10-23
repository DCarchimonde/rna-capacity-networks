from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT.parents[1] / "figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
COLORS = {"D_full26": "#176b87", "I_common4": "#bc5b32"}
LABELS = {"D_full26": "CRC D: full 26-gene graph", "I_common4": "CRC I: common 4-gene graph"}
summary = pd.read_csv(OUT / "competition_summary.csv")
detail = pd.read_csv(OUT / "competition_decomposition.csv")
power = pd.read_csv(OUT / "parity_permutation_power.csv")


def build_figures():
    d = detail[detail.k == 500]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    for key, g in d.groupby("cohort"):
        ax[0].scatter(
            100 * g.bivariate_resolution_change,
            100 * g.shared_capacity_change,
            s=24,
            alpha=0.8,
            label=LABELS[key],
            color=COLORS[key],
        )
    ax[0].axhline(0, color=".6", lw=0.8)
    ax[0].axvline(0, color=".6", lw=0.8)
    ax[0].set(
        xlabel="Bivariate change from R2 (pp)",
        ylabel="Shared-capacity change (pp)",
        title="A  Two sources of score change",
    )
    ax[0].legend(fontsize=7, loc="best", frameon=False)
    rng = np.random.default_rng(26101905)
    for pos, key in enumerate(["D_full26", "I_common4"]):
        v = 100 * d[d.cohort == key].competition_fraction.to_numpy()
        ax[1].scatter(
            pos + rng.uniform(-0.12, 0.12, len(v)), v, color=COLORS[key], s=20, alpha=0.75
        )
        ax[1].hlines(np.median(v), pos - 0.25, pos + 0.25, color="black", lw=2)
    ax[1].set(
        xticks=[0, 1],
        xticklabels=["D, n=62", "I, n=12"],
        ylabel="Independent-edge flow not jointly\nallocatable (%)",
        title="B  Competition fraction",
    )
    fig.savefig(FIG / "competition_components.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.8), layout="constrained", sharex=True, sharey=True
    )
    for ax, s in zip(axes.ravel(), sorted(power.s.unique())):
        d = power[(power.s == s) & (power.setting == "alternative")]
        for L, color in [(1000, "#bc5b32"), (10000, "#176b87")]:
            for method, ls, marker in [
                ("conditional", "-", "o"),
                ("multinomial_plugin", "--", "x"),
            ]:
                v = d[(d.L == L) & (d.method == method)].sort_values("n")
                label = f"L={L:,}, " + ("conditional" if method == "conditional" else "plug-in")
                ax.plot(
                    v.n,
                    100 * v.rejection_rate,
                    color=color,
                    ls=ls,
                    marker=marker,
                    ms=5,
                    label=label,
                    lw=1.5,
                )
            v = d[(d.L == L) & (d.method == "conditional")].sort_values("n")
            ax.fill_between(
                v.n, 100 * v.rejection_ci_low, 100 * v.rejection_ci_high, color=color, alpha=0.10
            )
        ax.set_xscale("log")
        ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1,000", "3,000"])
        ax.set_ylim(-3, 103)
        ax.set_title(f"s = {s:g}; fixed k = 500")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.supxlabel("Cells per group")
    fig.supylabel("Permutation-test rejection rate (%)")
    fig.savefig(FIG / "parity_detection.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    build_figures()
