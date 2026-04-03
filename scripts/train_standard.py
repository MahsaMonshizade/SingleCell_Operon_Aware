"""
scripts/train_standard.py
=========================
Train and save the standard SCVI baseline model for a named experiment.

Usage (from project root):
    python scripts/train_standard.py --experiment corr_log1p_lambda0p01
"""

import argparse
import sys
sys.path.insert(0, ".")

import scanpy as sc
import scvi
from scvi.model import SCVI as StandardSCVI

from config import ADATA_PATH, MODEL_KWARGS, TRAIN_KWARGS
from experiment_utils import (
    add_experiment_args,
    build_experiment_paths,
    ensure_experiment_dirs,
    load_run_config,
    save_run_config,
)


parser = argparse.ArgumentParser()
add_experiment_args(parser)
args = parser.parse_args()

paths = build_experiment_paths(args.experiment)
ensure_experiment_dirs(paths)

print("=" * 50)
print("Training Standard SCVI")
print("=" * 50)
print(f"  Experiment:   {args.experiment}")
print(f"  Save path:    {paths['model_standard']}")

adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

model = StandardSCVI(adata, **MODEL_KWARGS)
print(f"\n{model}")

print("\nTraining...")
model.train(**TRAIN_KWARGS)

model.save(str(paths["model_standard"]), overwrite=True)
print(f"\nSaved → {paths['model_standard']}")

run_config = load_run_config(paths)
run_config.update(
    {
        "experiment": args.experiment,
        "adata_path": ADATA_PATH,
        "model_kwargs": MODEL_KWARGS,
        "train_kwargs": TRAIN_KWARGS,
        "standard_model_path": str(paths["model_standard"]),
    }
)
save_run_config(paths, run_config)
