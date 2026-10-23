"""Patient-level exploratory MMR analysis; fixed models, training-fold scaling."""

from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
P = ROOT.parents[1]
features = pd.read_csv(OUT / "expanded_patient_features.csv").set_index("patient")
scores = pd.read_csv(OUT / "expanded_graph_scores.csv")
old = np.load(P / "data/development_counts.npz")["patients"]
rng = np.random.default_rng(26101878)
results = []
predrows = []
associations = []
contrasts = []
for graph in ["integrin26", "selected5", "common4"]:
    wide = scores[scores.graph == graph].pivot(index="patient", columns="k", values="R")
    d = features.join(wide)
    d["delta"] = d[500] - d[2]
    genes = [c for c in d.columns if str(c).startswith("mean_")]
    if graph == "selected5":
        genes = ["mean_" + g for g in ["ITGA5", "ITGAV", "ITGB1", "ITGB3", "ITGB6"]]
    if graph == "common4":
        genes = ["mean_" + g for g in ["ITGA5", "ITGAV", "ITGB1", "ITGB6"]]
    for cohort, ids in [("original23", old), ("expanded62", d.index.to_numpy())]:
        a = d.loc[ids]
        y = (a.MMR == "MMRd").to_numpy(int)
        n = len(a)
        clinical = np.column_stack(
            [a.age.astype(float), (a.sex == "M").astype(float), (a.site == "right").astype(float)]
        )
        mean = np.log10(a[genes].to_numpy() + 1e-8)
        base = np.column_stack([clinical, mean, a[2]])
        models = {
            "mean_abundance": np.log10(a[genes].sum(axis=1).to_numpy()[:, None] + 1e-8),
            "mean_profile": mean,
            "R2": a[[2]].to_numpy(),
            "R500": a[[500]].to_numpy(),
            "R2_plus_delta": a[[2, "delta"]].to_numpy(),
            "clinical_mean_R2": base,
            "clinical_mean_R2_plus_delta": np.column_stack([base, a["delta"]]),
        }
        predictions = {}
        for name, X in models.items():
            assert np.isfinite(X).all()
            pr = np.zeros(n)
            for i in range(n):
                train = np.arange(n) != i
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
                )
                model.fit(X[train], y[train])
                pr[i] = model.predict_proba(X[i : i + 1])[0, 1]
            predictions[name] = pr
            results.append(
                dict(
                    cohort=cohort,
                    graph=graph,
                    model=name,
                    patients=n,
                    MMRd=int(y.sum()),
                    MMRp=int(n - y.sum()),
                    loo_AUC=roc_auc_score(y, pr),
                    loo_log_loss=log_loss(y, pr),
                    loo_Brier=brier_score_loss(y, pr),
                )
            )
            for i, pid in enumerate(ids):
                predrows.append(
                    dict(
                        cohort=cohort,
                        graph=graph,
                        model=name,
                        patient=pid,
                        MMRd=y[i],
                        heldout_probability=pr[i],
                    )
                )
        for reduced, expanded in [
            ("R2", "R2_plus_delta"),
            ("clinical_mean_R2", "clinical_mean_R2_plus_delta"),
        ]:
            pr0, pr1 = predictions[reduced], predictions[expanded]
            loss0 = -(y * np.log(pr0) + (1 - y) * np.log1p(-pr0))
            loss1 = -(y * np.log(pr1) + (1 - y) * np.log1p(-pr1))
            delta = loss1 - loss0
            # Intervals resample fixed held-out predictions, not the training procedure.
            ii = rng.integers(0, n, (4000, n))
            lossdiff = delta[ii].mean(1)
            aucdiff = []
            for ix in ii:
                if np.unique(y[ix]).size == 2:
                    aucdiff.append(roc_auc_score(y[ix], pr1[ix]) - roc_auc_score(y[ix], pr0[ix]))
            contrasts.append(
                dict(
                    cohort=cohort,
                    graph=graph,
                    reduced=reduced,
                    expanded=expanded,
                    log_loss_difference=float(delta.mean()),
                    loss_ci_low=float(np.quantile(lossdiff, 0.025)),
                    loss_ci_high=float(np.quantile(lossdiff, 0.975)),
                    AUC_difference=roc_auc_score(y, pr1) - roc_auc_score(y, pr0),
                    AUC_ci_low=float(np.quantile(aucdiff, 0.025)),
                    AUC_ci_high=float(np.quantile(aucdiff, 0.975)),
                    interval_interpretation="patient bootstrap of fixed LOO predictions; conditional on fitted models",
                )
            )
        for target in [2, 500, "delta"]:
            v = a[target].to_numpy()
            test = mannwhitneyu(v[y == 1], v[y == 0], alternative="two-sided", method="auto")
            associations.append(
                dict(
                    cohort=cohort,
                    graph=graph,
                    target=str(target),
                    MMRd_median=np.median(v[y == 1]),
                    MMRp_median=np.median(v[y == 0]),
                    rank_AUC=test.statistic / (y.sum() * (n - y.sum())),
                    two_sided_p=test.pvalue,
                )
            )
ass = pd.DataFrame(associations)
ass["holm_p_within_graph_cohort"] = np.nan
for key, group in ass.groupby(["cohort", "graph"]):
    order = group.sort_values("two_sided_p").index
    adj = np.minimum(
        1, np.maximum.accumulate(ass.loc[order, "two_sided_p"].to_numpy() * np.arange(3, 0, -1))
    )
    ass.loc[order, "holm_p_within_graph_cohort"] = adj
pd.DataFrame(results).to_csv(OUT / "MMR_fixed_model_cv.csv", index=False)
pd.DataFrame(predrows).to_csv(OUT / "MMR_heldout_predictions.csv", index=False)
pd.DataFrame(contrasts).to_csv(OUT / "MMR_incremental_comparison.csv", index=False)
ass.to_csv(OUT / "MMR_associations.csv", index=False)
(OUT / "phenotype_protocol.json").write_text(
    json.dumps(
        dict(
            endpoint="author MMRStatus",
            primary_cohort="expanded62",
            primary_graph="integrin26",
            primary_resolution=500,
            primary_increment="clinical_mean_R2_plus_delta versus clinical_mean_R2",
            model="L2 logistic C=1; no tuning; leave-one-original-patient-out; training-only scaling",
            gene_features="log10(mean count fraction+1e-8)",
            clinical_features=["age", "sex M indicator", "right-sided location indicator"],
            covariate_rationale="assess expression and molecular-resolution increment beyond basic patient context",
            selection="No selection based on MMR effect size or significance",
            independence="Original 23 patients are included in expanded 62; no external phenotype replication",
            confidence_intervals="descriptive patient bootstrap of fixed held-out predictions",
            seed=26101878,
        ),
        indent=2,
    )
    + "\n"
)
print(
    pd.DataFrame(results)
    .query('cohort=="expanded62" and graph=="integrin26"')
    .to_string(index=False)
)
print(
    pd.DataFrame(contrasts)
    .query('cohort=="expanded62" and graph=="integrin26"')
    .to_string(index=False)
)
