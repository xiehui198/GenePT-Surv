"""ElasticNet-Cox on module activity plus clinical covariates.

Hyper-parameters are chosen by a leave-one-cohort-out loop inside the training
cohorts; the held-out cohort is never used for model selection.
"""
import itertools

import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

K_GRID = (50, 100, 200, 500)
L1_GRID = (0.05, 0.3, 0.7, 0.95)
MIN_TRAIN_EVENTS = 15
MIN_VALID_EVENTS = 5


def zscore_fit(x):
    mu, sd = x.mean(0), x.std(0) + 1e-8
    return np.nan_to_num((x - mu) / sd), (mu, sd)


def zscore_apply(x, stats):
    mu, sd = stats
    return np.nan_to_num((x - mu) / sd)


def screen(x, y, k):
    """Univariate Cox score test; returns the k features with the largest |z|."""
    order = np.argsort(-y["time"].astype(float))
    z, event = x[order], y["event"].astype(bool)[order]
    n = np.arange(1, len(z) + 1, dtype=float)[:, None]
    mean = np.cumsum(z, 0) / n
    var = np.cumsum(z ** 2, 0) / n - mean ** 2
    u = (z - mean)[event].sum(0)
    v = var[event].sum(0)
    score = np.nan_to_num(np.abs(u) / np.sqrt(np.maximum(v, 1e-12)))
    return np.argsort(-score)[: min(k, x.shape[1])]


def inner_folds(train):
    for cohort in train.cohort.unique():
        mask = (train.cohort != cohort).values
        if (train[mask].event.sum() >= MIN_TRAIN_EVENTS
                and train[~mask].event.sum() >= MIN_VALID_EVENTS):
            yield mask


def cindex(y, risk):
    return concordance_index_censored(y["event"].astype(bool), y["time"],
                                      risk)[0]


def _inner_score(x, y, train, clinical, k, l1, alphas):
    """Mean inner-fold C-index along the penalty path."""
    rows = []
    for mask in inner_folds(train):
        y_fit = Surv.from_arrays(train[mask].event.astype(bool).values,
                                 train[mask].time.values)
        y_val = Surv.from_arrays(train[~mask].event.astype(bool).values,
                                 train[~mask].time.values)
        x_fit, stats = zscore_fit(x[mask])
        x_val = zscore_apply(x[~mask], stats)
        sel = screen(x_fit, y_fit, k)
        fit = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=alphas,
                                     max_iter=50000).fit(
            np.hstack([x_fit[:, sel], clinical[mask]]), y_fit)
        design = np.hstack([x_val[:, sel], clinical[~mask]])
        rows.append([cindex(y_val, fit.predict(design, alpha=a))
                     if j < fit.coef_.shape[1] else np.nan
                     for j, a in enumerate(alphas)])
    if not rows:
        return None
    mean = np.nanmean(np.array(rows, dtype=float), axis=0)
    return None if np.all(np.isnan(mean)) else mean


def fit_predict(x_train, y_train, x_test, train, clinical_train,
                clinical_test):
    best = (-np.inf, None)
    for k, l1 in itertools.product(K_GRID, L1_GRID):
        sel = screen(x_train, y_train, k)
        alphas = CoxnetSurvivalAnalysis(
            l1_ratio=l1, alpha_min_ratio=0.005, n_alphas=40,
            max_iter=40000).fit(
            np.hstack([x_train[:, sel], clinical_train]), y_train).alphas_

        mean = _inner_score(x_train, y_train, train, clinical_train, k, l1,
                            alphas)
        if mean is None:
            continue
        j = int(np.nanargmax(mean))
        if mean[j] > best[0]:
            best = (mean[j], (k, l1, alphas, j))

    if best[1] is None:
        raise RuntimeError("no hyper-parameter setting could be evaluated")

    k, l1, alphas, j = best[1]
    sel = screen(x_train, y_train, k)
    fit = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=alphas,
                                 max_iter=50000).fit(
        np.hstack([x_train[:, sel], clinical_train]), y_train)
    alpha = fit.alphas_[min(j, fit.coef_.shape[1] - 1)]
    risk = fit.predict(np.hstack([x_test[:, sel], clinical_test]), alpha=alpha)

    coef = np.zeros(x_train.shape[1])
    coef[sel] = fit.coef_[:len(sel), min(j, fit.coef_.shape[1] - 1)]
    return risk, coef
