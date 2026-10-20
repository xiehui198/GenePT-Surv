"""Leave-one-dataset-out evaluation of the final model.

Each cohort is held out in turn; modules, screening, hyper-parameter selection
and all normalisation statistics come from the other five cohorts only.
"""
import numpy as np
import pandas as pd
from sksurv.util import Surv

import config
import data
import metrics
import model
import modules


def main():
    clinical = data.load_clinical()
    ranks, genes = data.load_rank_matrix(clinical)
    activity, labels, gene_index = modules.build_modules(ranks, genes)
    print(f"{len(clinical)} patients, {int(clinical.event.sum())} events, "
          f"{len(genes)} genes, {len(gene_index)} with an embedding")

    risk = np.full(len(clinical), np.nan)
    coefficients = {}
    for held_out in config.COHORTS:
        test = (clinical.cohort == held_out).values
        train_df = clinical[~test]

        y_train = Surv.from_arrays(train_df.event.astype(bool).values,
                                   train_df.time.values)
        clin_train, median = data.clinical_matrix(train_df)
        clin_test, _ = data.clinical_matrix(clinical[test], median)
        mu, sd = clin_train.mean(0), clin_train.std(0) + 1e-8

        x_train, stats = model.zscore_fit(activity[~test])
        x_test = model.zscore_apply(activity[test], stats)

        risk[test], coefficients[held_out] = model.fit_predict(
            x_train, y_train, x_test, train_df,
            (clin_train - mu) / sd, (clin_test - mu) / sd)
        print(f"  {held_out} done")

    cohort = clinical.cohort.values
    time = clinical.time.values
    event = clinical.event.values

    pooled = metrics.stratified_cindex(cohort, time, event, risk)
    low, high = metrics.bootstrap_ci(cohort, time, event, risk)
    print(f"\npooled C-index {pooled:.4f} (95% CI {low:.4f}-{high:.4f})")
    for g in config.COHORTS:
        idx = np.where(cohort == g)[0]
        print(f"  {g:12s} {metrics.stratified_cindex(cohort, time, event, risk, idx):.4f}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"sample": clinical["sample"], "cohort": cohort,
                  "time": time, "event": event, "risk": risk}).to_csv(
        config.OUTPUT_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame({"gene": genes[gene_index], "module": labels}).to_csv(
        config.OUTPUT_DIR / "module_membership.csv", index=False)
    pd.DataFrame(coefficients, index=[f"module_{k}" for k in
                                      range(config.N_MODULES)]).to_csv(
        config.OUTPUT_DIR / "fold_coefficients.csv")


if __name__ == "__main__":
    main()
