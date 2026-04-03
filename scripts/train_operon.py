"""
scripts/train_operon.py
=======================
Train and save the OperonAwareSCVI model for a named experiment.

Usage (from project root):
    python scripts/train_operon.py --experiment corr_log1p_lambda0p01 --lambda-val 0.01
"""

import argparse
import sys
sys.path.insert(0, ".")

import scanpy as sc
import scvi
from scvi.model import OperonAwareSCVI

from config import ADATA_PATH, MODEL_KWARGS, OPERON_NEIGHBORS, TRAIN_KWARGS
from experiment_utils import (
    add_experiment_args,
    build_experiment_paths,
    ensure_experiment_dirs,
    load_run_config,
    save_run_config,
)
from operon_aware_lib import load_neighbor_indices


parser = argparse.ArgumentParser()
add_experiment_args(parser, include_model_args=True)
args = parser.parse_args()

paths = build_experiment_paths(args.experiment)
ensure_experiment_dirs(paths)

print("=" * 50)
print("Training OperonAware SCVI")
print("=" * 50)
print(f"  Experiment:    {args.experiment}")
print(f"  Save path:     {paths['model_operon']}")
print(f"  Neighbor loss: {args.neighbor_loss}")
print(f"  Lambda:        {args.lambda_val}")
print(f"  Huber delta:   {args.huber_delta}")

adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

neighbor_indices = load_neighbor_indices(OPERON_NEIGHBORS, adata)
print(f"  Operon pairs: {neighbor_indices.shape[1]:,}")

model = OperonAwareSCVI(
    adata,
    neighbor_indices=neighbor_indices,
    lambda_val=args.lambda_val,
    neighbor_loss=args.neighbor_loss,
    huber_delta=args.huber_delta,
    **MODEL_KWARGS,
)
print(f"\n{model}")

print("\nTraining...")
model.train(**TRAIN_KWARGS)

model.save(str(paths["model_operon"]), overwrite=True)
print(f"\nSaved → {paths['model_operon']}")

run_config = load_run_config(paths)
run_config.update(
    {
        "experiment": args.experiment,
        "adata_path": ADATA_PATH,
        "neighbor_csv": OPERON_NEIGHBORS,
        "neighbor_loss": args.neighbor_loss,
        "lambda_val": args.lambda_val,
        "huber_delta": args.huber_delta,
        "model_kwargs": MODEL_KWARGS,
        "train_kwargs": TRAIN_KWARGS,
        "operon_model_path": str(paths["model_operon"]),
        "standard_model_path": str(paths["model_standard"]),
    }
)
save_run_config(paths, run_config)
