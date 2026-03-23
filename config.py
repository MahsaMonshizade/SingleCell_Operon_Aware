"""
config.py
=========
Single source of truth for all paths and hyperparameters.
Every script imports from here — never hardcode paths elsewhere.
"""

# ── Data ─────────────────────────────────────────────────
ADATA_PATH       = "pountain_data/outputs/lb_adata.h5ad"
OPERON_NEIGHBORS = "operon_neighbors.csv"
GFF_PATH         = "pountain_data/reference/GCF_000005845.2_ASM584v2_genomic.gff"
OPERON_TSV       = "OperonSet.tsv"

# ── Model checkpoints ────────────────────────────────────
MODEL_STANDARD   = "pountain_data/outputs/lb_scVI_model_benchmark"
MODEL_OPERON     = "pountain_data/outputs/lb_scVI_model_operon_aware"

# ── Evaluation outputs ───────────────────────────────────
OUT_DIR              = "pountain_data/outputs"
OUT_COMPARISON_PLOT  = f"{OUT_DIR}/model_comparison.png"
OUT_REGULONDB_PLOT   = f"{OUT_DIR}/regulondb_operon_evaluation.png"
OUT_REGULONDB_CSV    = f"{OUT_DIR}/per_operon_results.csv"

# ── Shared model architecture ─────────────────────────────
# Used by both train_standard.py and train_operon.py
MODEL_KWARGS = dict(
    n_layers        = 2,
    n_latent        = 5,
    n_hidden        = 64,
    dropout_rate    = 0.1,
    gene_likelihood = "zinb",
    dispersion      = "gene",
)

# ── OperonAwareSCVI-specific ──────────────────────────────
LAMBDA_VAL = 1.0   # best from sweep: {0.01: +0.0031, 0.05: +0.0038, 0.1: -0.0025}

# ── Training ──────────────────────────────────────────────
TRAIN_KWARGS = dict(
    check_val_every_n_epoch = 1,
)

# ── Evaluation ────────────────────────────────────────────
EVAL_N_CELLS      = 3000   # cells to subsample for expression computation
EVAL_N_MC_SAMPLES = 50     # MC samples for marginal log-likelihood
EVAL_RANDOM_SEED  = 42