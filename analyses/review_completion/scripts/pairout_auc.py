"""AUC from a common model for each held-out positive/negative patient pair.

Pooled LOO probabilities compare models with different training class counts;
we retain LOO proper scoring rules but replace pooled LOO AUC for interpretation.
"""

from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
rng = np.random.default_rng(26101882)
d = pd.read_csv(OUT / "expanded_patient_features.csv").set_index("patient")
s = pd.read_csv(OUT / "expanded_graph_scores.csv")
w = s[s.graph == "integrin26"].pivot(index="patient", columns="k", values="R")
d = d.join(w)
d["delta"] = d[500] - d[2]
y = (d.MMR == "MMRd").to_numpy(int)
n = len(y)
genes = [c for c in d.columns if str(c).startswith("mean_")]
mean = np.log10(d[genes].to_numpy() + 1e-8)
clinical = np.column_stack(
    [d.age.astype(float), (d.sex == "M").astype(float), (d.site == "right").astype(float)]
)
base = np.column_stack([clinical, mean, d[2]])
models = {
    "mean_abundance": np.log10(d[genes].sum(1).to_numpy()[:, None] + 1e-8),
    "mean_profile": mean,
    "R2": d[[2]].to_numpy(),
    "R500": d[[500]].to_numpy(),
    "R2_plus_delta": d[[2, "delta"]].to_numpy(),
    "clinical_mean_R2": base,
    "clinical_mean_R2_plus_delta": np.column_stack([base, d["delta"]]),
}
pairs = np.array([(i, j) for i in np.flatnonzero(y == 1) for j in np.flatnonzero(y == 0)])
weights = rng.multinomial(n, np.full(n, 1 / n), size=4000)
pairweights = weights[:, pairs[:, 0]] * weights[:, pairs[:, 1]]
den = pairweights.sum(1)
rows = []
wins = {}
details = []
for name, X in models.items():
    v = []
    for i, j in pairs:
        train = np.ones(n, bool)
        train[[i, j]] = False
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
        )
        model.fit(X[train], y[train])
        pr = model.predict_proba(X[[i, j]])[:, 1]
        win = float(pr[0] > pr[1]) + 0.5 * float(pr[0] == pr[1])
        v.append(win)
        details.append(
            dict(
                model=name,
                positive_patient=d.index[i],
                negative_patient=d.index[j],
                positive_probability=pr[0],
                negative_probability=pr[1],
                concordance=win,
            )
        )
    v = np.array(v)
    wins[name] = v
    bs = (pairweights @ v) / den
    rows.append(
        dict(
            model=name,
            patients=n,
            opposite_class_pairs=len(pairs),
            leave_pair_out_AUC=float(v.mean()),
            conditional_patient_bootstrap_low=float(np.quantile(bs, 0.025)),
            conditional_patient_bootstrap_high=float(np.quantile(bs, 0.975)),
        )
    )
    print(name, v.mean(), flush=True)
pd.DataFrame(rows).to_csv(OUT / "MMR_leave_pair_out_auc.csv", index=False)
pd.DataFrame(details).to_csv(OUT / "MMR_leave_pair_out_predictions.csv", index=False)
a = wins["clinical_mean_R2_plus_delta"] - wins["clinical_mean_R2"]
bs = (pairweights @ a) / den
(OUT / "MMR_pairout_increment.json").write_text(
    json.dumps(
        dict(
            AUC_difference=float(a.mean()),
            conditional_bootstrap_low=float(np.quantile(bs, 0.025)),
            conditional_bootstrap_high=float(np.quantile(bs, 0.975)),
            bootstrap_repetitions=4000,
            scope="Expanded 62-patient full integrin graph; fixed C=1 models",
            audit="Pooled LOO AUC was excluded because it compares different class priors; LOO log loss and Brier score remain primary proper scoring rules. No graph, endpoint or model penalty was selected using AUC.",
        ),
        indent=2,
    )
    + "\n"
)
