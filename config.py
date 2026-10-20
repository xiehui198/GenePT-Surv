"""Paths and constants. Replace the paths with your own."""
from pathlib import Path

DATA_DIR = Path("data")

CLINICAL_CSV = DATA_DIR / "clinical.csv"
EXPR_CSV = DATA_DIR / "expr_{cohort}.csv"
EMBEDDING_PICKLE = DATA_DIR / "genept_3l_embeddings.pickle"
OUTPUT_DIR = Path("output")

COHORTS = ["GSE68465", "TCGA-LUAD", "GSE31210", "GSE50081", "GSE30219",
           "GSE42127"]

FOLLOWUP_MONTHS = 60.0
N_MODULES = 100
SEED = 0
