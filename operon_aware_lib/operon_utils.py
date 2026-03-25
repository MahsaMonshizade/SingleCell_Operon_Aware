"""
operon_utils.py
===============
Load and filter RegulonDB operon data against an AnnData object.
"""

import pandas as pd


def parse_regulondb_operons(filepath: str) -> pd.DataFrame:
    """
    Parse RegulonDB operonset.tsv.
    Returns a DataFrame of multi-gene operons (≥2 genes) with columns:
        operon_id, operon_name, genes (list), confidence_label
    """
    conf_map = {"C": "Confirmed", "S": "Strong", "W": "Weak"}
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("1)operonId"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            genes = [g.strip() for g in parts[6].split(";") if g.strip()]
            if len(genes) < 2:
                continue
            confidence = parts[8].strip() if len(parts) > 8 else "W"
            rows.append({
                "operon_id":        parts[0].strip(),
                "operon_name":      parts[1].strip(),
                "genes":            genes,
                "confidence_label": conf_map.get(confidence, "Weak"),
            })

    df = pd.DataFrame(rows)
    print(f"[operon_utils] Parsed {len(df)} multi-gene operons from RegulonDB")
    print(df["confidence_label"].value_counts().to_string())
    return df



def load_operons(filepath: str, gene_to_idx: dict):
    """
    Parse RegulonDB operons and filter to genes present in adata.
    Returns (operons_valid, operons_all) where operons_valid has ≥2 genes in adata.

    Parameters
    ----------
    filepath
        Path to RegulonDB operonset.tsv
    gene_to_idx
        {gene_symbol: adata_column_index} from gene_mapping.build_gene_to_idx()

    Returns
    -------
    operons_valid : pd.DataFrame  — operons with ≥2 genes found in adata
    operons_all   : pd.DataFrame  — full parsed operon table (for inspection)
    """
    df = parse_regulondb_operons(filepath)

    df["genes_found"] = df["genes"].apply(
        lambda gl: [g for g in gl if g in gene_to_idx])
    df["idx_found"] = df["genes_found"].apply(
        lambda gl: [gene_to_idx[g] for g in gl])

    valid   = df[df["genes_found"].apply(len) >= 2].copy().reset_index(drop=True)
    n_pairs = valid["genes_found"].apply(lambda g: len(g) * (len(g)-1) // 2).sum()

    print(f"\n[operon_utils] {len(valid)} operons with ≥2 genes in adata "
          f"({n_pairs} total gene pairs)")
    print("[operon_utils] Coverage by confidence level:")
    for conf in ["Confirmed", "Strong", "Weak"]:
        sub     = df[df["confidence_label"] == conf]
        n_valid = (sub["genes_found"].apply(len) >= 2).sum()
        print(f"  {conf:9s}: {n_valid}/{len(sub)} operons have ≥2 genes in adata")

    return valid, df