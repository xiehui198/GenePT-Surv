"""GenePT semantic modules.

Genes are clustered by their GenePT text embedding, so the partition uses no
expression, outcome or cohort information and is identical in every fold.
"""
import pickle

import numpy as np
from sklearn.cluster import KMeans

import config


def load_embeddings(genes):
    """Returns the embedding matrix and the indices of the genes it covers."""
    with open(config.EMBEDDING_PICKLE, "rb") as fh:
        emb = pickle.load(fh)
    index = np.array([i for i, g in enumerate(genes) if g in emb])
    matrix = np.vstack([np.asarray(emb[genes[i]], dtype=float) for i in index])
    return matrix, index


def build_modules(ranks, genes, n_modules=config.N_MODULES, seed=config.SEED):
    """Module activity of a patient is the mean within-sample rank of the genes
    assigned to that module."""
    matrix, index = load_embeddings(genes)
    labels = KMeans(n_modules, n_init=4, random_state=seed).fit(matrix).labels_

    activity = np.zeros((ranks.shape[0], n_modules))
    for k in range(n_modules):
        members = index[labels == k]
        if len(members):
            activity[:, k] = ranks[:, members].mean(1)
    return activity, labels, index
