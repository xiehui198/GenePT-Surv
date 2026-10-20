# GenePT semantic modules for cross-cohort survival prediction in LUAD

Reference implementation of the final model: genes are grouped into 100 semantic
modules by clustering their GenePT text embeddings, each patient is described by
the mean within-sample expression rank of each module, and an ElasticNet-Cox
model is fitted on the module activities together with age, sex and stage.
Performance is measured under leave-one-dataset-out (LODO) validation over six
lung adenocarcinoma cohorts.

## Install

```bash
pip install -r requirements.txt
```

## Data

Place the following in `data/` (paths are set in `src/config.py`):

| File | Content |
| --- | --- |
| `clinical.csv` | one row per patient: `sample, cohort, age, sex, stage, os_months, os_event` |
| `expr_<cohort>.csv` | expression matrix of one cohort, genes in rows, samples in columns |
| `genept_3l_embeddings.pickle` | `{gene_symbol: vector}` |

`sex` is `M`/`F`, `stage` is 1–4, `os_event` is 1 for death. Expression values
may be on any scale; only the within-patient ranking is used.

GenePT embeddings are available from the authors of GenePT
(https://github.com/yiqunchen/GenePT); this work uses the `text-embedding-3-large`
release. The six cohorts are public: GSE68465, GSE31210, GSE50081, GSE30219,
GSE42127 and TCGA-LUAD.

## Run

```bash
cd src && python run_lodo.py
```

Writes to `output/`:

- `oof_predictions.csv` — out-of-fold risk score of every patient
- `module_membership.csv` — module assignment of every gene
- `fold_coefficients.csv` — module coefficients in each fold

## Files

| File | Purpose |
| --- | --- |
| `config.py` | paths and constants |
| `data.py` | loading, 60-month truncation, within-sample rank normalisation |
| `modules.py` | k-means modules from the embeddings and their activity |
| `model.py` | univariate screen and ElasticNet-Cox with inner leave-one-cohort-out tuning |
| `metrics.py` | cohort-stratified C-index and bootstrap interval |
| `run_lodo.py` | LODO loop and entry point |
