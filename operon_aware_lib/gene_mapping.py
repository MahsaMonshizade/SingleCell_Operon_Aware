"""
gene_mapping.py
===============
Build a gene symbol → adata column index mapping from a RefSeq GFF3 file.
Needed because RegulonDB uses gene symbols (e.g. thrA) while adata
uses Blattner locus tags (e.g. b0002).
"""

import re


def parse_gff_symbol_to_blattner(gff_path: str) -> dict:
    """
    Parse a RefSeq GFF3 file and return {gene_symbol: blattner_tag}.
    e.g. {'thrL': 'b0001', 'thrA': 'b0002', ...}
    """
    symbol_to_blattner = {}
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            name  = re.search(r"Name=([^;]+)", attrs)
            locus = re.search(r"locus_tag=(b\d+)", attrs)
            if name and locus:
                symbol_to_blattner[name.group(1)] = locus.group(1)

    print(f"[gene_mapping] Parsed {len(symbol_to_blattner)} "
          f"symbol→blattner pairs from GFF")
    return symbol_to_blattner


def build_gene_to_idx(gff_path: str, adata) -> dict:
    """
    Build {gene_symbol: adata_column_index} via GFF + adata.var_names.

    Parameters
    ----------
    gff_path
        Path to the RefSeq GFF3 annotation file.
    adata
        AnnData whose var_names are Blattner locus tags (b0001 etc.)

    Returns
    -------
    dict  {gene_symbol: int}
    """
    symbol_to_blattner = parse_gff_symbol_to_blattner(gff_path)
    blattner_to_idx    = {b: i for i, b in enumerate(adata.var_names)}

    gene_to_idx = {
        symbol: blattner_to_idx[blattner]
        for symbol, blattner in symbol_to_blattner.items()
        if blattner in blattner_to_idx
    }

    n_total   = len(symbol_to_blattner)
    n_mapped  = len(gene_to_idx)
    print(f"[gene_mapping] {n_mapped}/{n_total} symbols mapped to adata "
          f"({n_total - n_mapped} not found — likely filtered in preprocessing)")
    return gene_to_idx