"""
scripts/train_standard.py
=========================
Train and save the standard SCVI baseline model.

Usage (from project root):
    python scripts/train_standard.py
"""

import sys
sys.path.insert(0, ".")

import scanpy as sc
import scvi
from scvi.model import SCVI as StandardSCVI

from config import ADATA_PATH, MODEL_STANDARD, MODEL_KWARGS, TRAIN_KWARGS

print("=" * 50)
print("Training Standard SCVI")
print("=" * 50)

adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

model = StandardSCVI(adata, **MODEL_KWARGS)
print(f"\n{model}")

print("\nTraining...")
model.train(**TRAIN_KWARGS)

model.save(MODEL_STANDARD, overwrite=True)
print(f"\nSaved → {MODEL_STANDARD}")