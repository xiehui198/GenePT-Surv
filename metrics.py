"""Cohort-stratified concordance index and its bootstrap interval."""
import numpy as np
from sksurv.metrics import concordance_index_censored


def stratified_cindex(cohort, time, event, risk, index=None):
    """Pairs are counted within cohorts and summed across them: every fold
    produces its own model, so risks are not comparable between cohorts."""
    index = np.arange(len(risk)) if index is None else index
    num = den = 0.0
    for group in np.unique(cohort[index]):
        m = index[cohort[index] == group]
        if len(m) < 3 or event[m].sum() < 1:
            continue
        _, conc, disc, tied, _ = concordance_index_censored(
            event[m].astype(bool), time[m], risk[m])
        num += conc + 0.5 * tied
        den += conc + disc + tied
    return num / den if den > 0 else np.nan


def bootstrap_ci(cohort, time, event, risk, n_boot=2000, seed=0):
    """Patients are resampled with replacement within each cohort."""
    rng = np.random.default_rng(seed)
    by_cohort = [np.where(cohort == g)[0] for g in np.unique(cohort)]
    values = []
    for _ in range(n_boot):
        index = np.concatenate(
            [rng.choice(idx, len(idx), replace=True) for idx in by_cohort])
        values.append(stratified_cindex(cohort, time, event, risk, index))
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)
