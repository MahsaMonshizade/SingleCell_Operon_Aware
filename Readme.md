I downloaded the  E coli data from Pountain paper's github : ranscription–replication interactions reveal bacterial genome regulation
downloaded the operon data from RegulonDB 


# OperonAware scVI

A biologically-informed extension of [scVI](https://docs.scvi-tools.org/) for microbial single-cell RNA-seq data that incorporates **operon structure** as a regularization signal during training.

---

## Background

Standard scVI was designed for eukaryotic single-cell data. In bacteria, genes within the same **operon** are co-transcribed and should therefore show correlated expression. This project modifies the scVI VAE loss function to penalize differences in the decoded expression of operon-neighbor gene pairs, encouraging the model to learn latent representations that respect prokaryotic transcriptional organization.

The penalty added to the reconstruction loss is:

```
L_operon = λ · mean( (px_rate[:, g1] - px_rate[:, g2])² )
```

where `g1, g2` are operon-neighbor gene index pairs and `λ` is a tunable hyperparameter (`lambda_val`).

---

## Project Structure

```
project/
├── config.py                        # All paths and hyperparameters
│
├── operon_aware_lib/                # Shared utility library
│   ├── __init__.py
│   ├── load_neighbor_indices.py     # Load operon pairs CSV → tensor
│   ├── gene_mapping.py              # GFF parsing: gene symbol → adata index
│   └── operon_utils.py              # RegulonDB TSV parsing and filtering
│
├── scripts/                         # Executable pipeline scripts
│   ├── train_standard.py            # Train baseline scVI
│   ├── train_operon.py              # Train OperonAwareSCVI
│   ├── compare_models.py            # General model comparison
│   └── evaluate_regulondb.py        # RegulonDB ground-truth evaluation
│
└── notebooks/
    └── results.ipynb                # Results exploration (no heavy compute)
```

The scVI module modifications live in the cloned `scvi-tools` repo:
- `scvi/module/_vae_new.py` — `OperonAwareVAE` (adds operon penalty to loss)
- `scvi/model/_scvi_new.py` — `OperonAwareSCVI` (model class)

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

## Input Files

| File | Description |
|---|---|
| `pountain_data/outputs/lb_adata.h5ad` | AnnData object with Blattner locus tags (`b0001`...) as `var_names` |
| `operon_neighbors.csv` | Two-column CSV (`gene_1`, `gene_2`) of operon-adjacent gene pairs in Blattner format |
| `OperonSet.tsv` | RegulonDB operon set (downloaded from [regulondb.ccg.unam.mx](https://regulondb.ccg.unam.mx)) |
| `pountain_data/reference/*.gff` | RefSeq GFF3 annotation for *E. coli* K-12 MG1655 — used to map gene symbols to Blattner tags |

### Note on gene naming
`adata.var_names` uses **Blattner locus tags** (e.g. `b0002`) while RegulonDB uses **gene symbols** (e.g. `thrA`). The `gene_mapping.py` utility bridges this using the GFF file, which contains both `Name=thrA` and `locus_tag=b0002` in the same record.

---

## Configuration

All paths and hyperparameters are set in **`config.py`**. Edit this file before running any scripts — nothing else needs changing.

```python
# Key settings to review:
ADATA_PATH       = "pountain_data/outputs/lb_adata.h5ad"
OPERON_NEIGHBORS = "operon_neighbors.csv"
GFF_PATH         = "pountain_data/reference/GCF_000005845.2_ASM584v2_genomic.gff"
OPERON_TSV       = "OperonSet.tsv"

LAMBDA_VAL       = 0.05   # operon regularization strength
MODEL_KWARGS     = dict(n_layers=2, n_latent=5, n_hidden=64, ...)
```

---

## Usage

All scripts are run from the **project root**:

```bash
# 1. Train both models (GPU recommended, ~20–30 min each)
python scripts/train_standard.py
python scripts/train_operon.py

# 2. General comparison (ELBO, MLL, silhouette, UMAP)
python scripts/compare_models.py

# 3. RegulonDB operon-specific evaluation
python scripts/evaluate_regulondb.py
```

Outputs are saved to `pountain_data/outputs/`:
- `model_comparison.png`
- `regulondb_operon_evaluation.png`
- `per_operon_results.csv`

---

## Evaluation Metrics

### General (`compare_models.py`)

| Metric | Description |
|---|---|
| **ELBO** | Evidence lower bound — overall model fit (higher = better) |
| **Marginal LL** | Marginal log-likelihood via 50 MC samples — generalization quality |
| **Operon Consistency** | Mean Pearson r between denoised expression of training operon pairs |
| **Silhouette Score** | Cluster separability in latent space using Leiden communities |

### RegulonDB (`evaluate_regulondb.py`)

| Metric | Description |
|---|---|
| **Intra-operon r** | Mean pairwise Pearson r between genes in the same operon — stratified by confidence (Strong / Weak) |
| **Intra-inter contrast** | Difference between intra-operon and random gene-pair correlation — tests whether the model specifically respects operon structure |
| **Within-operon gene similarity** | Gene-level correlation structure within each operon |
| **% operons improved** | Fraction of RegulonDB operons showing higher co-expression under OperonAwareSCVI |

> **Note:** RegulonDB v14.5 (release 03-04-2026) contains no Confirmed-confidence operons — only Strong (61) and Weak (787). The Strong operon result is the most meaningful ground-truth signal.

---

## Lambda Tuning

The `lambda_val` hyperparameter controls the strength of the operon penalty. Results from the sweep on this dataset:

| `lambda_val` | Mean Δ intra-operon r | % operons improved | Verdict |
|---|---|---|---|
| 0.01 | +0.0031 | 52.9% | Operon slightly better |
| **0.05** | **+0.0038** | **53.4%** | **Best — current setting** |
| 0.1 | −0.0025 | 49.6% | Over-penalized |

Too high a lambda distorts the reconstruction loss. The optimum for this dataset is around **0.05**.

---

## Results Summary (lambda = 0.05)

| Metric | Standard SCVI | Operon SCVI | Winner |
|---|---|---|---|
| ELBO | −376.39 | −376.37 | ✓ Operon |
| Marginal LL | −0.22 | −0.22 | ✓ Operon |
| Operon Consistency | 0.3377 | 0.3415 | ✓ Operon |
| Silhouette Score | 0.0646 | 0.0945 | ✓ Operon |
| Intra-operon r (all) | 0.3377 | 0.3415 | ✓ Operon |
| Within-operon similarity | 0.3427 | 0.3498 | ✓ Operon |

---

## Citation

If you use this work, please cite the original scVI paper:

> Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15(12), 1053–1058.

And RegulonDB:

> Salgado H., Gama-Castro S., et al. RegulonDB v12.0: a comprehensive resource of transcriptional regulation in E. coli K-12. *Nucleic Acids Research*, 2023. https://doi.org/10.1093/nar/gkad1072