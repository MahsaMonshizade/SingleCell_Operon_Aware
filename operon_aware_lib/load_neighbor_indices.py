"""
load_neighbor_indices.py
========================
Convert a CSV of gene-name pairs (Blattner locus tags) to integer
index pairs usable as neighbor_indices in OperonAwareSCVI.
"""

import pandas as pd
import torch


def load_neighbor_indices(csv_path: str, adata) -> torch.Tensor:
    """
    Parameters
    ----------
    csv_path
        Path to CSV with columns gene_1, gene_2 (Blattner locus tags).
    adata
        AnnData object registered with scvi — var_names must be Blattner tags.

    Returns
    -------
    torch.LongTensor of shape (2, n_pairs)
        Row 0: indices of gene_1
        Row 1: indices of gene_2
    """
    df = pd.read_csv(csv_path)

    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}

    mask = df["gene_1"].isin(gene_to_idx) & df["gene_2"].isin(gene_to_idx)
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"[load_neighbor_indices] Warning: dropping {n_dropped} pairs "
              f"with genes not in adata.var_names")
    df = df[mask].reset_index(drop=True)

    idx1 = torch.LongTensor([gene_to_idx[g] for g in df["gene_1"]])
    idx2 = torch.LongTensor([gene_to_idx[g] for g in df["gene_2"]])

    print(f"[load_neighbor_indices] Loaded {len(idx1)} operon neighbor pairs")
    return torch.stack([idx1, idx2], dim=0)   # shape: (2, n_pairs)