"""
scripts/train_operon.py
=======================
Train and save the OperonAwareSCVI model.

Usage (from project root):
    python scripts/train_operon.py
"""

import sys
sys.path.insert(0, ".")

import scanpy as sc
import scvi
from scvi.model import OperonAwareSCVI

from config import (ADATA_PATH, MODEL_OPERON, OPERON_NEIGHBORS,
                    MODEL_KWARGS, TRAIN_KWARGS, LAMBDA_VAL)
from operon_aware_lib import load_neighbor_indices

print("=" * 50)
print("Training OperonAware SCVI")
print("=" * 50)

adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

neighbor_indices = load_neighbor_indices(OPERON_NEIGHBORS, adata)
print(f"  Operon pairs: {neighbor_indices.shape[1]:,}")
print(f"  Lambda:       {LAMBDA_VAL}")

model = OperonAwareSCVI(
    adata,
    neighbor_indices = neighbor_indices,
    lambda_val       = LAMBDA_VAL,
    **MODEL_KWARGS,
)
print(f"\n{model}")

print("\nTraining...")
model.train(**TRAIN_KWARGS)

model.save(MODEL_OPERON, overwrite=True)
print(f"\nSaved → {MODEL_OPERON}")