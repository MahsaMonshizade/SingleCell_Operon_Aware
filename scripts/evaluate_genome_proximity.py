"""
scripts/evaluate_genome_proximity.py
===================================
Evaluate whether OperonAwareSCVI preferentially strengthens local genomic
co-expression rather than globally smoothing all genes.

This script uses genome annotation and the heuristic training pairs to compare:
  1. adjacent same-strand training-neighbor pairs
  2. adjacent same-strand non-neighbor pairs
  3. adjacent opposite-strand pairs
  4. random gene pairs
  5. correlation-vs-distance decay for same- vs opposite-strand pairs

Usage (from project root):
    python scripts/evaluate_genome_proximity.py --experiment corr_log1p_lambda0p01 --baseline-experiment shared_baseline
"""

import argparse
import sys
sys.path.insert(0, ".")

import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.stats import pearsonr

from scvi.model import SCVI as StandardSCVI
from scvi.model import OperonAwareSCVI

from config import (
    ADATA_PATH,
    GFF_PATH,
    OPERON_NEIGHBORS,
    EVAL_N_CELLS,
    EVAL_RANDOM_SEED,
)
from experiment_utils import (
    add_experiment_args,
    build_experiment_paths,
    ensure_experiment_dirs,
    metrics_path,
    plot_path,
)


DARK = "#0f1117"
PANEL = "#1a1d27"
TEXT = "#e0e0e0"
ACC_STD = "#4fc3f7"
ACC_OP = "#f06292"

DIST_BINS = [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000)]
MAX_PAIRS_PER_GROUP = 1500


def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)


def safe_corr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return pearsonr(a, b)[0]


def load_gene_table(gff_path, adata):
    rows = []
    adata_genes = set(adata.var_names)
    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            locus = re.search(r"locus_tag=(b\d+)", attrs)
            if locus is None:
                continue
            locus_tag = locus.group(1)
            if locus_tag not in adata_genes:
                continue
            rows.append(
                {
                    "locus_tag": locus_tag,
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                }
            )
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    df["adata_idx"] = df["locus_tag"].map({g: i for i, g in enumerate(adata.var_names)})
    return df


def compute_pair_corrs(expr, pairs):
    vals = []
    for i, j in pairs:
        r = safe_corr(expr[:, i], expr[:, j])
        if not np.isnan(r):
            vals.append(r)
    return np.asarray(vals)


print("=" * 60)
print("Genome proximity evaluation")
print("=" * 60)

parser = argparse.ArgumentParser()
add_experiment_args(parser, include_baseline_arg=True)
args = parser.parse_args()

paths = build_experiment_paths(args.experiment)
baseline_paths = build_experiment_paths(args.baseline_experiment)
ensure_experiment_dirs(paths)
out_plot = plot_path(paths, "genome_proximity_evaluation.png")
out_csv = metrics_path(paths, "genome_proximity_metrics.csv")
out_distance_csv = metrics_path(paths, "genome_proximity_distance_bins.csv")

print("\nLoading data and models...")
print(f"  Experiment: {args.experiment}")
print(f"  Baseline:   {args.baseline_experiment}")
adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
model_std = StandardSCVI.load(str(baseline_paths["model_standard"]), adata=adata)
model_op = OperonAwareSCVI.load(str(paths["model_operon"]), adata=adata)

np.random.seed(EVAL_RANDOM_SEED)
if EVAL_N_CELLS is None or EVAL_N_CELLS >= adata.n_obs:
    adata_sub = adata.copy()
else:
    cell_idx = np.random.choice(adata.n_obs, size=EVAL_N_CELLS, replace=False)
    adata_sub = adata[cell_idx].copy()
scvi.data.setup_anndata(adata_sub)

expr_std = model_std.get_normalized_expression(adata=adata_sub).values
expr_op = model_op.get_normalized_expression(adata=adata_sub).values

print(f"  Cells used: {adata_sub.n_obs:,}")
print(f"  Genes used: {adata_sub.n_vars:,}")

print("\nLoading genome annotation...")
gene_df = load_gene_table(GFF_PATH, adata_sub)
neighbor_df = pd.read_csv(OPERON_NEIGHBORS)
neighbor_set = {
    tuple(sorted((row.gene_1, row.gene_2)))
    for row in neighbor_df.itertuples(index=False)
}

print(f"  Genes with GFF + adata coverage: {len(gene_df):,}")
print(f"  Training neighbor pairs in CSV: {len(neighbor_set):,}")

adj_same_neighbor = []
adj_same_nonneighbor = []
adj_opposite = []
distance_groups = {
    "same_0_200": [],
    "same_200_500": [],
    "same_500_1000": [],
    "same_1000_2000": [],
    "same_2000_5000": [],
    "opp_0_200": [],
    "opp_200_500": [],
    "opp_500_1000": [],
    "opp_1000_2000": [],
    "opp_2000_5000": [],
}

for idx in range(1, len(gene_df)):
    prev = gene_df.iloc[idx - 1]
    curr = gene_df.iloc[idx]
    pair = (int(prev["adata_idx"]), int(curr["adata_idx"]))
    pair_tag = tuple(sorted((prev["locus_tag"], curr["locus_tag"])))
    distance = int(curr["start"] - prev["end"])

    if distance < 0:
        continue

    if prev["strand"] == curr["strand"]:
        if pair_tag in neighbor_set:
            adj_same_neighbor.append(pair)
        else:
            adj_same_nonneighbor.append(pair)
    else:
        adj_opposite.append(pair)

    for lo, hi in DIST_BINS:
        if lo <= distance < hi:
            prefix = "same" if prev["strand"] == curr["strand"] else "opp"
            distance_groups[f"{prefix}_{lo}_{hi}"].append(pair)
            break


all_gene_idx = gene_df["adata_idx"].tolist()
random_pairs = []
seen_random = set()
target_random = min(
    MAX_PAIRS_PER_GROUP,
    len(adj_same_neighbor),
    len(adj_same_nonneighbor) if adj_same_nonneighbor else len(adj_same_neighbor),
    len(adj_opposite) if adj_opposite else len(adj_same_neighbor),
)
while len(random_pairs) < target_random:
    i, j = np.random.choice(all_gene_idx, 2, replace=False)
    key = tuple(sorted((int(i), int(j))))
    if key in seen_random:
        continue
    seen_random.add(key)
    random_pairs.append((int(i), int(j)))


def downsample_pairs(pairs, limit):
    if len(pairs) <= limit:
        return pairs
    idx = np.random.choice(len(pairs), size=limit, replace=False)
    return [pairs[i] for i in idx]


pair_groups = {
    "adjacent_same_training_neighbor": downsample_pairs(adj_same_neighbor, MAX_PAIRS_PER_GROUP),
    "adjacent_same_nonneighbor": downsample_pairs(adj_same_nonneighbor, MAX_PAIRS_PER_GROUP),
    "adjacent_opposite_strand": downsample_pairs(adj_opposite, MAX_PAIRS_PER_GROUP),
    "random_pairs": random_pairs,
}

print("\nPair-group sizes:")
for name, pairs in pair_groups.items():
    print(f"  {name}: {len(pairs):,}")

summary_rows = []
distribution_rows = []

print("\nComputing pairwise correlations...")
for group_name, pairs in pair_groups.items():
    corr_std = compute_pair_corrs(expr_std, pairs)
    corr_op = compute_pair_corrs(expr_op, pairs)

    mean_std = float(np.mean(corr_std))
    mean_op = float(np.mean(corr_op))
    delta = mean_op - mean_std

    summary_rows.append(
        {
            "group": group_name,
            "n_pairs": len(corr_std),
            "mean_standard": mean_std,
            "mean_operon": mean_op,
            "delta": delta,
            "winner": "operon" if delta > 0 else "standard",
        }
    )

    for a, b in zip(corr_std, corr_op):
        distribution_rows.append(
            {"group": group_name, "r_standard": a, "r_operon": b, "delta": b - a}
        )

distance_rows = []
for group_name, pairs in distance_groups.items():
    if len(pairs) < 10:
        continue
    pairs_sub = downsample_pairs(pairs, MAX_PAIRS_PER_GROUP)
    corr_std = compute_pair_corrs(expr_std, pairs_sub)
    corr_op = compute_pair_corrs(expr_op, pairs_sub)
    distance_rows.append(
        {
            "distance_group": group_name,
            "n_pairs": len(corr_std),
            "mean_standard": float(np.mean(corr_std)),
            "mean_operon": float(np.mean(corr_op)),
            "delta": float(np.mean(corr_op) - np.mean(corr_std)),
        }
    )

summary_df = pd.DataFrame(summary_rows)
dist_df = pd.DataFrame(distribution_rows)
distance_df = pd.DataFrame(distance_rows)

summary_df.to_csv(out_csv, index=False)
distance_df.to_csv(out_distance_csv, index=False)

print(f"\nSaved summary      → {out_csv}")
print(f"Saved distance bins → {out_distance_csv}")

print("\nSummary:")
for _, row in summary_df.iterrows():
    print(
        f"  {row['group']}: "
        f"Std={row['mean_standard']:.4f}  "
        f"Op={row['mean_operon']:.4f}  "
        f"Δ={row['delta']:+.4f}  "
        f"({row['winner']})"
    )

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(summary_df))
width = 0.36
ax1.bar(x - width / 2, summary_df["mean_standard"], width=width, color=ACC_STD, label="Standard")
ax1.bar(x + width / 2, summary_df["mean_operon"], width=width, color=ACC_OP, label="Operon")
ax1.set_xticks(x)
ax1.set_xticklabels(
    [
        "Train\nneighbor",
        "Same-strand\nnon-neighbor",
        "Adjacent\nopposite",
        "Random",
    ],
    rotation=0,
)
ax1.set_ylabel("Mean pair correlation")
ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax1, "Pair-Group Mean Correlation")

ax2 = fig.add_subplot(gs[0, 1])
for i, group_name in enumerate(summary_df["group"]):
    grp = dist_df[dist_df["group"] == group_name]
    ax2.scatter(
        np.full(len(grp), i) + np.random.uniform(-0.12, 0.12, len(grp)),
        grp["delta"],
        s=8,
        alpha=0.35,
        color=ACC_OP,
        rasterized=True,
    )
ax2.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.set_xticks(np.arange(len(summary_df)))
ax2.set_xticklabels(
    ["Train\nneighbor", "Same-strand\nnon-neighbor", "Adjacent\nopposite", "Random"]
)
ax2.set_ylabel("Per-pair delta correlation (operon - standard)")
style_ax(ax2, "Delta Distribution by Pair Group")

ax3 = fig.add_subplot(gs[1, 0])
same_df = distance_df[distance_df["distance_group"].str.startswith("same_")].copy()
opp_df = distance_df[distance_df["distance_group"].str.startswith("opp_")].copy()

def midpoint(label):
    lo, hi = label.split("_")[1:]
    return (int(lo) + int(hi)) / 2

same_df["mid"] = same_df["distance_group"].apply(midpoint)
opp_df["mid"] = opp_df["distance_group"].apply(midpoint)
same_df = same_df.sort_values("mid")
opp_df = opp_df.sort_values("mid")

ax3.plot(same_df["mid"], same_df["mean_standard"], "-o", color=ACC_STD, label="Standard same-strand")
ax3.plot(same_df["mid"], same_df["mean_operon"], "-o", color=ACC_OP, label="Operon same-strand")
ax3.plot(opp_df["mid"], opp_df["mean_standard"], "--o", color="#7ec8ff", label="Standard opposite-strand")
ax3.plot(opp_df["mid"], opp_df["mean_operon"], "--o", color="#ff9ab8", label="Operon opposite-strand")
ax3.set_xlabel("Intergenic distance bin midpoint (bp)")
ax3.set_ylabel("Mean pair correlation")
ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7)
style_ax(ax3, "Correlation-vs-Distance Decay")

ax4 = fig.add_subplot(gs[1, 1])
delta_same = same_df[["mid", "delta"]].rename(columns={"delta": "same_delta"})
delta_opp = opp_df[["mid", "delta"]].rename(columns={"delta": "opp_delta"})
ax4.plot(delta_same["mid"], delta_same["same_delta"], "-o", color=ACC_OP, label="Same-strand Δ")
ax4.plot(delta_opp["mid"], delta_opp["opp_delta"], "--o", color="#ffd54f", label="Opposite-strand Δ")
ax4.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.6)
ax4.set_xlabel("Intergenic distance bin midpoint (bp)")
ax4.set_ylabel("Mean delta correlation")
ax4.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax4, "Where Does Operon Help?")

plt.tight_layout()
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
print(f"Saved plot         → {out_plot}")
