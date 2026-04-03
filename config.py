"""
config.py
=========
Default configuration for data paths, model hyperparameters, and experiment
naming. Scripts can override experiment settings from the command line.
"""

# ── Data ─────────────────────────────────────────────────
ADATA_PATH = "pountain_data/outputs/lb_adata.h5ad"
OPERON_NEIGHBORS = "operon_aware_data/operon_neighbors.csv"
GFF_PATH = "pountain_data/reference/GCF_000005845.2_ASM584v2_genomic.gff"
OPERON_TSV = "operon_aware_data/OperonSet.tsv"

# ── Experiment defaults ──────────────────────────────────
RESULTS_ROOT = "Results"
EXPERIMENTS_ROOT = f"{RESULTS_ROOT}/experiments"
ARCHIVE_ROOT = f"{RESULTS_ROOT}/archive"

DEFAULT_NEIGHBOR_LOSS = "corr_log1p"
LAMBDA_VAL = 0.5
HUBER_DELTA = 0.1
DEFAULT_EXPERIMENT_NAME = "corr_log1p_lambda0p5"

# ── Shared model architecture ────────────────────────────
MODEL_KWARGS = dict(
    n_layers=2,
    n_latent=5,
    n_hidden=64,
    dropout_rate=0.1,
    gene_likelihood="zinb",
    dispersion="gene",
)

# ── Training ─────────────────────────────────────────────
TRAIN_KWARGS = dict(
    check_val_every_n_epoch=1,
)

# ── Evaluation ───────────────────────────────────────────
EVAL_N_CELLS = None
EVAL_N_MC_SAMPLES = 50
EVAL_RANDOM_SEED = 42
