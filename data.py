"""Loading, follow-up truncation and within-sample rank normalisation."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config


def load_clinical():
    df = pd.read_csv(config.CLINICAL_CSV)
    df["time"] = df.os_months.clip(upper=config.FOLLOWUP_MONTHS)
    df["event"] = np.where(df.os_months > config.FOLLOWUP_MONTHS, 0,
                           df.os_event).astype(int)
    return df[df.time > 0].reset_index(drop=True)


def load_rank_matrix(clinical):
    """Percentile rank of every gene within each patient, over the genes shared
    by all cohorts. Uses no statistic from other patients or cohorts."""
    keep = set(clinical["sample"])
    frames = {}
    for cohort in config.COHORTS:
        expr = pd.read_csv(str(config.EXPR_CSV).format(cohort=cohort),
                           index_col=0)
        frames[cohort] = expr[[s for s in expr.columns if s in keep]]

    genes = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    parts = []
    for expr in frames.values():
        x = expr.loc[genes].T.values.astype(float)
        ranks = np.vstack([rankdata(row) / len(genes) for row in x])
        parts.append(pd.DataFrame(ranks, index=expr.columns, columns=genes))

    ranks = pd.concat(parts).loc[clinical["sample"]]
    return ranks.values.astype(float), np.array(genes)


def clinical_matrix(df, median=None):
    """Age, stage and sex; missing values filled with the training median."""
    mat = pd.DataFrame({"age": df.age.values,
                        "stage": df.stage.values,
                        "sex_M": (df.sex == "M").astype(float).values},
                       index=df.index)
    median = mat.median() if median is None else median
    return mat.fillna(median).values.astype(float), median
