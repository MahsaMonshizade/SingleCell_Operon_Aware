I downloaded the  E coli data from Pountain paper's github : ranscription–replication interactions reveal bacterial genome regulation
downloaded the operon data from RegulonDB 


# OperonAware scVI

A biologically-informed extension of [scVI](https://docs.scvi-tools.org/) for microbial single-cell RNA-seq data that incorporates **operon structure** as a regularization signal during training.

---

## Background

Standard scVI was designed for eukaryotic single-cell data. In bacteria, genes within the same **operon** are co-transcribed and should therefore show correlated expression. This project modifies the scVI VAE loss function to penalize differences in the decoded expression of operon-neighbor gene pairs, encouraging the model to learn latent representations that respect prokaryotic transcriptional organization.

The current penalty added to the reconstruction loss is a **correlation-style neighbor loss**:

```
L_operon = λ · mean(1 - corr(log(1 + px_rate[:, g1]), log(1 + px_rate[:, g2])))
```

where `g1, g2` are operon-neighbor gene index pairs and `λ` is a tunable hyperparameter (`lambda_val`).

The key modeling change is that neighboring genes are now encouraged to **co-vary across cells**, rather than being forced to have the same absolute decoded count. This is a better match to operon biology: genes in the same operon often move together, but they do not necessarily have identical expression magnitude.

---

## Project Structure

```
project/
├── config.py                        # All paths and hyperparameters
├── experiment_utils.py              # Shared experiment folder/path helpers
│
├── operon_aware_lib/                # Shared utility library
│   ├── __init__.py
│   ├── load_neighbor_indices.py     # Load operon pairs CSV → tensor
│   ├── gene_mapping.py              # GFF parsing: gene symbol → adata index
│   ├── operon_utils.py              # RegulonDB TSV parsing and filtering
│   └── generate_neighbors.py        # GFF parsing: find neighbor genes → csv file
│
├── scripts/                         # Executable pipeline scripts
│   ├── train_standard.py            # Train baseline scVI
│   ├── train_operon.py              # Train OperonAwareSCVI
│   ├── evaluate_regulondb.py        # RegulonDB ground-truth evaluation
│   ├── evaluate_genome_proximity.py # Local genomic specificity evaluation
│   └── evaluate_corruption_recovery.py # Direct denoising benchmark
│
└── notebooks/
    ├── results.ipynb                # Results exploration
    ├── initial_processing.ipynb     # Raw data processing
    └── adata_distribution_overview.ipynb # Pre-model distribution checks
```

The scVI module modifications live in the cloned `scvi-tools` repo:
- `scvi/module/_vae_new.py` — `OperonAwareVAE` (adds operon penalty to loss)
- `scvi/model/_scvi_new.py` — `OperonAwareSCVI` (model class)

Generated outputs are organized under:

```
Results/
└── experiments/
    └── <experiment_name>/
        ├── models/
        │   ├── standard/
        │   └── operon/
        ├── metrics/
        ├── plots/
        └── run_config.json
```

## Neighbor Loss Design

The project has gone through a few versions of the operon regularizer.

### 1. Original loss: raw-rate MSE

```
L_operon = λ · mean((px_rate[:, g1] - px_rate[:, g2])²)
```

**Idea:** directly force neighboring genes to have similar decoded expression.

**Downside:** this is very sensitive to high-expression genes. A small number of genes with large decoded rates can dominate the penalty. It also assumes operon neighbors should have the same absolute expression level, which is often too strong biologically.

### 2. Intermediate loss: Huber on log1p rates

```
L_operon = λ · Huber(log(1 + px_rate[:, g1]) - log(1 + px_rate[:, g2]))
```

**Idea:** reduce the effect of extreme counts by working on `log1p(px_rate)` and replacing pure squared error with a robust Huber penalty.

**Downside:** this is more stable than raw MSE, but it still pushes neighboring genes toward similar absolute decoded levels. That is still stronger than the biological claim we really want.

### 3. Current loss: correlation-style loss on log1p rates

For each neighbor pair, the model:
- takes `log1p(px_rate)` for both genes
- centers each gene across the minibatch
- computes a cosine / Pearson-like similarity across cells
- penalizes `1 - similarity`

**Why this is preferred:** it asks whether neighboring genes rise and fall together across cells, rather than whether they have the same magnitude. That is closer to the idea of operon co-expression.

**Practical note:** because this loss behaves differently from the older MSE-style penalties, `lambda_val` usually needs to be retuned. A smaller sweep such as `0.001`, `0.005`, `0.01`, `0.02` is a sensible starting point.

---

## Installation

Clone and install `scvi-tools` in editable mode so your module changes are picked up:

```bash
git clone https://github.com/YosefLab/scvi-tools.git
cd scvi-tools
pip install -e . --break-system-packages
cd ..
```

Install additional dependencies:

```bash
micromamba install -c conda-forge python-igraph leidenalg -y
pip install scanpy scikit-learn scipy --break-system-packages
```

---

## Data Provenance
 
### Source dataset
This project builds on the *E. coli* K-12 MG1655 single-bacterium RNA-seq dataset
from Pountain et al., generated using the **PETRI-seq** protocol:
 
> Pountain AW, Jiang P, Podkowik M, Shopsin B, Torres VJ, Yanai I.
> *A quantitative model for the transcriptional landscape of the bacterial cell cycle.*
> bioRxiv, 2022. https://www.biorxiv.org/content/10.1101/2022.10.22.513359v1
 
The raw count matrices were processed using `notebooks/initial_processing.ipynb`,
the Yanai Lab's notebook from the
[TRIPs repository](https://github.com/yanailab/TRIPs/blob/main/Ecoli_D1/initial_processing.ipynb),
stored locally in this project for reproducibility.
 
**What the notebook does (~1 hr runtime):**
- Imports raw PETRI-seq count matrices from `pountain_data/outputs/count_matrices/`
- Quality control filtering — removes low-quality cells and lowly expressed genes,
  reducing from ~4,300+ GFF genes down to the **3,070 genes** used in this project
- Normalizes and runs standard scVI denoising
- Generates global gene correlation patterns used for exploratory analysis
 
**Output:** `pountain_data/outputs/lb_adata.h5ad` (~500 MB) with `var_names` as
Blattner locus tags (e.g. `b0001`). This file is too large for the repo — run the
notebook to generate it.
 


---
generate neighboring genes using following:
```python 
python operon_aware_lib/generate_neighbors.py 
```
---

## Input Files

| File | Description |
|---|---|
| `pountain_data/outputs/lb_adata.h5ad` | AnnData object with Blattner locus tags (`b0001`...) as `var_names` |
| `operon_aware_data/operon_neighbors.csv` | Two-column CSV (`gene_1`, `gene_2`) of operon-adjacent gene pairs in Blattner format |
| `operon_aware_data/OperonSet.tsv` | RegulonDB operon set (downloaded from [regulondb.ccg.unam.mx](https://regulondb.ccg.unam.mx/datasets)) |
| `pountain_data/reference/*.gff` | RefSeq GFF3 annotation for *E. coli* K-12 MG1655 — used to map gene symbols to Blattner tags |

### Note on gene naming
`adata.var_names` uses **Blattner locus tags** (e.g. `b0002`) while RegulonDB uses **gene symbols** (e.g. `thrA`). The `gene_mapping.py` utility bridges this using the GFF file, which contains both `Name=thrA` and `locus_tag=b0002` in the same record.

---

## Configuration

Global defaults live in **`config.py`**. The important defaults are:

```python
ADATA_PATH       = "pountain_data/outputs/lb_adata.h5ad"
OPERON_NEIGHBORS = "operon_aware_data/operon_neighbors.csv"
GFF_PATH         = "pountain_data/reference/GCF_000005845.2_ASM584v2_genomic.gff"
OPERON_TSV       = "operon_aware_data/OperonSet.tsv"

DEFAULT_EXPERIMENT_NAME = "corr_log1p_lambda0p5"
DEFAULT_NEIGHBOR_LOSS   = "corr_log1p"
LAMBDA_VAL              = 0.5
HUBER_DELTA             = 0.1
```

Implemented neighbor-loss options:
- `"mse_raw"` — original raw decoded-rate MSE
- `"huber_log1p"` — robust absolute-level penalty on log-transformed decoded rates
- `"corr_log1p"` — current default, correlation-style penalty on log-transformed decoded rates

The intended workflow is:
- keep stable defaults in `config.py`
- change experiment-specific settings from the command line
- let each run write into its own experiment folder

---

## Usage

All scripts are run from the **project root** and should use the same `--experiment` name:

```bash
# Example experiment name
EXP=corr_log1p_lambda0p01

# 1. Train both models
python scripts/train_standard.py --experiment $EXP
python scripts/train_operon.py --experiment $EXP --neighbor-loss corr_log1p --lambda-val 0.01

# 2. Run evaluations
python scripts/evaluate_regulondb.py --experiment $EXP
python scripts/evaluate_genome_proximity.py --experiment $EXP
python scripts/evaluate_corruption_recovery.py --experiment $EXP
```

Outputs for that run will be saved under:

```text
Results/experiments/$EXP/
```

This keeps models, metrics, and plots from different runs separated and makes it much easier to compare many loss / lambda combinations without manual renaming.

---

## Evaluation Metrics

<!-- ### General (`compare_models.py`)

| Metric | Description |
|---|---|
| **ELBO** | Evidence lower bound — overall model fit (higher = better) |
| **Marginal LL** | Marginal log-likelihood via 50 MC samples — generalization quality |
| **Operon Consistency** | Mean Pearson r between denoised expression of training operon pairs |
| **Silhouette Score** | Cluster separability in latent space using Leiden communities | -->

### RegulonDB (`evaluate_regulondb.py`)

| Metric | Description |
|---|---|
| **Intra-operon r** | Mean pairwise Pearson r between genes in the same operon — stratified by confidence (Strong / Weak) |
| **Intra-inter contrast** | Difference between intra-operon and random gene-pair correlation — tests whether the model specifically respects operon structure |
| **Within-operon gene similarity** | Gene-level correlation structure within each operon |
| **% operons improved** | Fraction of RegulonDB operons showing higher co-expression under OperonAwareSCVI |

> **Note:** RegulonDB v14.5 (release 03-04-2026) contains no Confirmed-confidence operons — only Strong (61) and Weak (787). The Strong operon result is the most meaningful ground-truth signal.

---

<!-- ## Lambda Tuning

The `lambda_val` hyperparameter controls the strength of the operon penalty. Results from the sweep on this dataset:

| `lambda_val` | Mean Δ intra-operon r | % operons improved | Verdict |
|---|---|---|---|
| 0.01 | +0.0031 | 52.9% | Operon slightly better |
| **0.05** | **+0.0038** | **53.4%** | **Best — current setting** |
| 0.1 | −0.0025 | 49.6% | Over-penalized |

Too high a lambda distorts the reconstruction loss. The optimum for this dataset is around **0.05**. -->

---

<!-- ## Results Summary (lambda = 0.05)

| Metric | Standard SCVI | Operon SCVI | Winner |
|---|---|---|---|
| ELBO | −376.39 | −376.37 | ✓ Operon |
| Marginal LL | −0.22 | −0.22 | ✓ Operon |
| Operon Consistency | 0.3377 | 0.3415 | ✓ Operon |
| Silhouette Score | 0.0646 | 0.0945 | ✓ Operon |
| Intra-operon r (all) | 0.3377 | 0.3415 | ✓ Operon |
| Within-operon similarity | 0.3427 | 0.3498 | ✓ Operon | -->

---

## Citation

If you use this work, please cite the original scVI paper:

> Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15(12), 1053–1058.

And RegulonDB:

> Salgado H., Gama-Castro S., et al. RegulonDB v12.0: a comprehensive resource of transcriptional regulation in E. coli K-12. *Nucleic Acids Research*, 2023. https://doi.org/10.1093/nar/gkad1072
