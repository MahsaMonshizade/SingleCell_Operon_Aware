"""
scripts/evaluate_regulondb_v2.py
================================
Evaluate Standard SCVI vs OperonAwareSCVI using RegulonDB ground-truth operons.

This script:
  1. Loads the processed AnnData object and both trained denoising models
  2. Loads RegulonDB operons and maps their genes to the AnnData gene space
  3. Computes denoised expression from both models on the same sampled cells
  4. Measures within-operon gene-gene correlation for each valid operon
  5. Compares the two models at the operon level and by confidence category
  6. Computes an intra-operon vs inter-operon contrast as a negative-control check
  7. Reports summary statistics, top improved/degraded operons, and saves plots/CSV

Outputs:
  - A CSV file with per-operon scores for both models
  - A summary figure comparing the two models across RegulonDB-based metrics

Usage:
    python scripts/evaluate_regulondb_v2.py
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scanpy as sc
import scvi

from scipy.stats import pearsonr, wilcoxon

from scvi.model import SCVI as StandardSCVI
from scvi.model import OperonAwareSCVI

from config import (
    ADATA_PATH,
    MODEL_STANDARD,
    MODEL_OPERON,
    GFF_PATH,
    OPERON_TSV,
    OUT_REGULONDB_PLOT,
    OUT_REGULONDB_CSV,
    EVAL_N_CELLS,
    EVAL_RANDOM_SEED,
)
from operon_aware_lib import build_gene_to_idx, load_operons


# ── Plot style ───────────────────────────────
DARK  = "#0f1117"
PANEL = "#1a1d27"
ACC1  = "#4fc3f7"   # standard
ACC2  = "#f06292"   # operon
ACC3  = "#81c784"   # confirmed
TEXT  = "#e0e0e0"


def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)


def safe_pearsonr(a, b):
    """Pearson correlation across cells for two genes."""
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return pearsonr(a, b)[0]


def mean_pairwise_corr(expr, gidx):
    """Mean within-operon pairwise correlation for one operon."""
    gidx = sorted(set(gidx))
    if len(gidx) < 2:
        return np.nan

    rs = [safe_pearsonr(expr[:, i], expr[:, j]) for i, j in combinations(gidx, 2)]
    rs = [r for r in rs if not np.isnan(r)]
    return np.mean(rs) if rs else np.nan


def pooled_pairwise_corr(expr, pair_list):
    """Average correlation across a list of gene pairs."""
    rs = [safe_pearsonr(expr[:, i], expr[:, j]) for i, j in pair_list]
    rs = np.array([r for r in rs if not np.isnan(r)], dtype=float)
    return rs


def unique_intra_operon_pairs(operons_df):
    """
    Build a unique set of within-operon gene pairs across all valid operons.
    """
    pairs = set()
    for _, row in operons_df.iterrows():
        gidx = sorted(set(row["idx_found"]))
        if len(gidx) < 2:
            continue
        for i, j in combinations(gidx, 2):
            pairs.add((i, j))
    return sorted(pairs)


def sample_unique_inter_pairs(all_gene_idx, intra_pair_set, n_pairs, rng):
    """
    Sample unique inter-operon control pairs not in the intra-operon set.
    """
    inter_pairs = set()
    all_gene_idx = np.array(sorted(set(all_gene_idx)))

    max_tries = max(100000, n_pairs * 50)
    tries = 0

    while len(inter_pairs) < n_pairs and tries < max_tries:
        i, j = sorted(rng.choice(all_gene_idx, 2, replace=False))
        pair = (i, j)
        if pair not in intra_pair_set:
            inter_pairs.add(pair)
        tries += 1

    if len(inter_pairs) < n_pairs:
        print(f"Warning: only sampled {len(inter_pairs)} inter-operon pairs out of requested {n_pairs}.")
    return sorted(inter_pairs)


def paired_wilcoxon_from_delta(delta_values, alternative="greater"):
    """
    Run Wilcoxon signed-rank on per-operon deltas.
    Tests whether median(delta) > 0 when alternative='greater'.
    """
    delta_values = np.asarray(delta_values, dtype=float)
    delta_values = delta_values[~np.isnan(delta_values)]

    if len(delta_values) == 0:
        return np.nan, np.nan

    # If all deltas are exactly zero, scipy can fail / be meaningless.
    if np.allclose(delta_values, 0):
        return 0.0, 1.0

    try:
        stat, pval = wilcoxon(delta_values, alternative=alternative, zero_method="wilcox")
    except ValueError:
        stat, pval = np.nan, np.nan
    return stat, pval


def winner(a, b):
    if np.isnan(a) or np.isnan(b):
        return "—"
    if b > a:
        return "✓ Operon"
    if a > b:
        return "✓ Standard"
    return "Tie"


# ═══════════════════════════════════════════
# 0. Load data and models
# ═══════════════════════════════════════════
print("=" * 60)
print("Loading data, models, and RegulonDB...")

adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)

model_std = StandardSCVI.load(MODEL_STANDARD, adata=adata)
model_op  = OperonAwareSCVI.load(MODEL_OPERON, adata=adata)

print(f"  Cells: {adata.n_obs:,} | Genes: {adata.n_vars:,}")

gene_to_idx = build_gene_to_idx(GFF_PATH, adata)
operons_valid, operons_all = load_operons(OPERON_TSV, gene_to_idx)

# Deduplicate indices per operon and keep only operons with >=2 mapped genes
operons_valid = operons_valid.copy()
operons_valid["idx_found"] = operons_valid["idx_found"].apply(lambda x: sorted(set(x)))
operons_valid = operons_valid[operons_valid["idx_found"].apply(len) >= 2].reset_index(drop=True)

print(f"  Valid operons with >=2 mapped genes: {len(operons_valid):,}")


# ═══════════════════════════════════════════
# 1. Denoised expression on same cell subset
# ═══════════════════════════════════════════
print("\nComputing denoised expression...")
rng = np.random.default_rng(EVAL_RANDOM_SEED)

n_cells_use = adata.n_obs
adata_sub = adata.copy()
scvi.data.setup_anndata(adata_sub)

expr_std = model_std.get_normalized_expression(adata=adata_sub).values
expr_op  = model_op.get_normalized_expression(adata=adata_sub).values

print(f"  Using {n_cells_use:,} cells for evaluation")


# ═══════════════════════════════════════════
# 2. Per-operon intra-operon correlation
# ═══════════════════════════════════════════
print("\nComputing per-operon intra-operon correlations...")

results = []
for _, row in operons_valid.iterrows():
    gidx = row["idx_found"]

    results.append({
        "operon_name": row["operon_name"],
        "n_genes": len(gidx),
        "confidence": row["confidence_label"],
        "r_standard": mean_pairwise_corr(expr_std, gidx),
        "r_operon": mean_pairwise_corr(expr_op, gidx),
    })

res_df = pd.DataFrame(results)

print(
    f"  Raw: {len(res_df)} operons | "
    f"NaN std: {res_df['r_standard'].isna().sum()} | "
    f"NaN op: {res_df['r_operon'].isna().sum()}"
)

res_df = res_df.dropna().reset_index(drop=True)
res_df["delta"] = res_df["r_operon"] - res_df["r_standard"]
pct_improved = 100 * (res_df["delta"] > 0).mean()

wilcoxon_stat, wilcoxon_p = paired_wilcoxon_from_delta(res_df["delta"].values, alternative="greater")

print(f"\n  Overall ({len(res_df)} operons):")
print(f"    Standard mean ± sd : {res_df['r_standard'].mean():.4f} ± {res_df['r_standard'].std():.4f}")
print(f"    Operon   mean ± sd : {res_df['r_operon'].mean():.4f} ± {res_df['r_operon'].std():.4f}")
print(f"    Mean Δ             : {res_df['delta'].mean():+.4f}")
print(f"    Median Δ           : {res_df['delta'].median():+.4f}")
print(f"    Wilcoxon p         : {wilcoxon_p:.3e}")
print(f"    % improved         : {pct_improved:.1f}%")

print("\n  By confidence level:")
confidence_order = ["Confirmed", "Strong", "Weak"]
for conf in confidence_order:
    sub = res_df[res_df["confidence"] == conf]
    if len(sub) == 0:
        continue

    _, p_conf = paired_wilcoxon_from_delta(sub["delta"].values, alternative="greater")
    print(
        f"    [{conf:9s}] n={len(sub):4d} | "
        f"Std={sub['r_standard'].mean():.4f}  "
        f"Op={sub['r_operon'].mean():.4f}  "
        f"MeanΔ={sub['delta'].mean():+.4f}  "
        f"MedΔ={sub['delta'].median():+.4f}  "
        f"p={p_conf:.3e}  "
        f"({(sub['delta'] > 0).mean() * 100:.0f}% improved)"
    )

primary = res_df[res_df["confidence"].isin(["Confirmed", "Strong"])]
if len(primary):
    _, p_primary = paired_wilcoxon_from_delta(primary["delta"].values, alternative="greater")
    print(f"\n  ── Primary subset: Confirmed + Strong (n={len(primary)}) ──")
    print(f"    Mean Δ   = {primary['delta'].mean():+.4f}")
    print(f"    Median Δ = {primary['delta'].median():+.4f}")
    print(f"    p        = {p_primary:.3e}")
    print(f"    % improved = {(primary['delta'] > 0).mean() * 100:.1f}%")

strong = res_df[res_df["confidence"] == "Strong"]
if len(strong):
    _, p_strong = paired_wilcoxon_from_delta(strong["delta"].values, alternative="greater")
    print(f"\n  ── Strong operons only (n={len(strong)}) ──")
    print(f"    Mean Δ   = {strong['delta'].mean():+.4f}")
    print(f"    Median Δ = {strong['delta'].median():+.4f}")
    print(f"    p        = {p_strong:.3e}")
    print(f"    % improved = {(strong['delta'] > 0).mean() * 100:.1f}%")


# ═══════════════════════════════════════════
# 3. Intra- vs inter-operon contrast
# ═══════════════════════════════════════════
print("\nComputing intra- vs inter-operon contrast...")

intra_pairs = unique_intra_operon_pairs(operons_valid)
intra_pair_set = set(intra_pairs)

all_gene_idx = sorted(set(gene_to_idx.values()))
inter_pairs = sample_unique_inter_pairs(
    all_gene_idx=all_gene_idx,
    intra_pair_set=intra_pair_set,
    n_pairs=len(intra_pairs),
    rng=rng,
)

intra_std = pooled_pairwise_corr(expr_std, intra_pairs)
intra_op  = pooled_pairwise_corr(expr_op,  intra_pairs)
inter_std = pooled_pairwise_corr(expr_std, inter_pairs)
inter_op  = pooled_pairwise_corr(expr_op,  inter_pairs)

contrast_std = np.mean(intra_std) - np.mean(inter_std)
contrast_op  = np.mean(intra_op)  - np.mean(inter_op)

print(f"  Unique intra-operon pairs : {len(intra_pairs):,}")
print(f"  Unique inter-operon pairs : {len(inter_pairs):,}")
print(f"  Standard contrast         : {contrast_std:.4f}")
print(f"  Operon contrast           : {contrast_op:.4f}")
print(f"  Δ contrast                : {contrast_op - contrast_std:+.4f}")


# ═══════════════════════════════════════════
# 4. Pooled within-operon pairwise score
# ═══════════════════════════════════════════
print("\nComputing pooled within-operon pairwise score...")
pooled_within_std = np.mean(intra_std)
pooled_within_op  = np.mean(intra_op)

print(
    f"  Std={pooled_within_std:.4f}  "
    f"Op={pooled_within_op:.4f}  "
    f"({winner(pooled_within_std, pooled_within_op)})"
)


# ═══════════════════════════════════════════
# 5. Top improved / degraded operons
# ═══════════════════════════════════════════
print("\nTop 10 most improved operons:")
print(
    res_df.nlargest(10, "delta")[
        ["operon_name", "n_genes", "confidence", "r_standard", "r_operon", "delta"]
    ].to_string(index=False)
)

print("\nTop 10 where standard was better:")
print(
    res_df.nsmallest(10, "delta")[
        ["operon_name", "n_genes", "confidence", "r_standard", "r_operon", "delta"]
    ].to_string(index=False)
)


# ═══════════════════════════════════════════
# 6. Save detailed CSV
# ═══════════════════════════════════════════
res_df_out = res_df.copy()
res_df_out["eval_seed"] = EVAL_RANDOM_SEED
res_df_out["n_cells_used"] = n_cells_use
res_df_out.to_csv(OUT_REGULONDB_CSV, index=False)
print(f"\nSaved CSV → {OUT_REGULONDB_CSV}")


# ═══════════════════════════════════════════
# 7. Plot
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

conf_colors = {"Confirmed": ACC3, "Strong": ACC1, "Weak": "#888"}

# 7a. Per-operon scatter
ax1 = fig.add_subplot(gs[0, 0])
for conf, grp in res_df.groupby("confidence"):
    ax1.scatter(
        grp["r_standard"],
        grp["r_operon"],
        s=12,
        alpha=0.6,
        color=conf_colors.get(conf, "#888"),
        label=conf,
        rasterized=True,
    )

lims = [
    min(res_df[["r_standard", "r_operon"]].min()) - 0.02,
    max(res_df[["r_standard", "r_operon"]].max()) + 0.02,
]
ax1.plot(lims, lims, "w--", lw=0.8, alpha=0.5)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("Standard SCVI intra-operon r")
ax1.set_ylabel("OperonAware SCVI intra-operon r")
ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7, markerscale=1.5)
style_ax(ax1, "Per-Operon Intra-Correlation\n(above diagonal = OperonAware better)")

# 7b. Delta by confidence
ax2 = fig.add_subplot(gs[0, 1])
conf_groups = [c for c in confidence_order if len(res_df[res_df["confidence"] == c]) > 0]
bp = ax2.boxplot(
    [res_df[res_df["confidence"] == c]["delta"].values for c in conf_groups],
    patch_artist=True,
    notch=False,
    medianprops=dict(color="white", lw=2),
    whiskerprops=dict(color=TEXT),
    capprops=dict(color=TEXT),
    flierprops=dict(marker=".", color="#555", markersize=3),
)
for patch, conf in zip(bp["boxes"], conf_groups):
    patch.set_facecolor(conf_colors.get(conf, "#888"))
    patch.set_alpha(0.7)
ax2.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
ax2.set_xticklabels(conf_groups)
ax2.set_ylabel("Δ intra-operon r (OperonAware − Standard)")
style_ax(ax2, "Improvement by Confidence Level")

# 7c. Intra vs inter contrast
ax3 = fig.add_subplot(gs[0, 2])
x, w = np.arange(2), 0.35

for offset, means, errs, col, label in [
    (
        -w / 2,
        [np.mean(intra_std), np.mean(inter_std)],
        [
            np.std(intra_std) / np.sqrt(len(intra_std)),
            np.std(inter_std) / np.sqrt(len(inter_std)),
        ],
        ACC1,
        "Standard SCVI",
    ),
    (
        +w / 2,
        [np.mean(intra_op), np.mean(inter_op)],
        [
            np.std(intra_op) / np.sqrt(len(intra_op)),
            np.std(inter_op) / np.sqrt(len(inter_op)),
        ],
        ACC2,
        "OperonAware SCVI",
    ),
]:
    ax3.bar(
        x + offset,
        means,
        w,
        yerr=errs,
        color=col,
        alpha=0.8,
        label=label,
        capsize=3,
        edgecolor="none",
    )

ax3.set_xticks(x)
ax3.set_xticklabels(["Intra-operon", "Inter-operon"])
ax3.set_ylabel("Mean pairwise r")
ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax3, "Intra vs Inter Gene-Pair Correlation")

# 7d. Delta histogram
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(res_df["delta"], bins=35, color=ACC2, alpha=0.75, edgecolor="none")
ax4.axvline(0, color="white", lw=0.8, ls="--", alpha=0.6)
ax4.set_xlabel("Δ intra-operon r")
ax4.set_ylabel("Count")
style_ax(ax4, "Distribution of Per-Operon Improvements")

# 7e. Delta vs operon size
ax5 = fig.add_subplot(gs[1, 1])
for conf, grp in res_df.groupby("confidence"):
    ax5.scatter(
        grp["n_genes"],
        grp["delta"],
        s=16,
        alpha=0.65,
        color=conf_colors.get(conf, "#888"),
        label=conf,
        rasterized=True,
    )
ax5.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
ax5.set_xlabel("Operon size (# genes)")
ax5.set_ylabel("Δ intra-operon r")
style_ax(ax5, "Improvement vs Operon Size")

# 7f. Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
ax6.set_facecolor(PANEL)

table_data = [
    ["Metric", "Standard", "OperonAware", "Δ", "p-value", "Winner"],
    [
        "Per-operon mean intra-r",
        f"{res_df['r_standard'].mean():.4f}",
        f"{res_df['r_operon'].mean():.4f}",
        f"{res_df['delta'].mean():+.4f}",
        f"{wilcoxon_p:.3e}" if not np.isnan(wilcoxon_p) else "N/A",
        winner(res_df["r_standard"].mean(), res_df["r_operon"].mean()),
    ],
    [
        "Per-operon median Δ",
        "—",
        "—",
        f"{res_df['delta'].median():+.4f}",
        "—",
        "✓ Operon" if res_df["delta"].median() > 0 else "✓ Standard",
    ],
    [
        "Intra-inter contrast",
        f"{contrast_std:.4f}",
        f"{contrast_op:.4f}",
        f"{contrast_op - contrast_std:+.4f}",
        "—",
        winner(contrast_std, contrast_op),
    ],
    [
        "% operons improved",
        "—",
        f"{pct_improved:.1f}%",
        "—",
        "—",
        "✓ Operon" if pct_improved > 50 else "✓ Standard",
    ],
    [
        "Pooled within-operon r",
        f"{pooled_within_std:.4f}",
        f"{pooled_within_op:.4f}",
        f"{pooled_within_op - pooled_within_std:+.4f}",
        "—",
        winner(pooled_within_std, pooled_within_op),
    ],
]

tbl = ax6.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#23263a" if r == 0 else PANEL)
    cell.set_edgecolor("#2a2d3a")
    cell.set_text_props(color=TEXT)
    if r > 0 and c == 5 and r <= len(table_data) - 1:
        txt = table_data[r][5]
        if "Operon" in txt:
            cell.set_facecolor("#1e3a2f")
        elif "Standard" in txt:
            cell.set_facecolor("#3a1e1e")

# 7g. Top improved
ax7 = fig.add_subplot(gs[2, :])
ax7.axis("off")
ax7.set_facecolor(PANEL)

top_show = res_df.nlargest(8, "delta")[
    ["operon_name", "confidence", "n_genes", "r_standard", "r_operon", "delta"]
].copy()

for col in ["r_standard", "r_operon", "delta"]:
    top_show[col] = top_show[col].map(lambda x: f"{x:.4f}")

tbl2 = ax7.table(
    cellText=top_show.values,
    colLabels=top_show.columns,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_facecolor("#23263a" if r == 0 else PANEL)
    cell.set_edgecolor("#2a2d3a")
    cell.set_text_props(color=TEXT)

fig.suptitle(
    "OperonAware SCVI vs Standard SCVI — RegulonDB Evaluation (v2)",
    color=TEXT,
    fontsize=14,
    fontweight="bold",
    y=0.99,
)

plt.savefig(OUT_REGULONDB_PLOT, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved plot → {OUT_REGULONDB_PLOT}")


# ═══════════════════════════════════════════
# 8. Final verdict
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("Verdict summary:")
print(f"  {'✓' if res_df['delta'].mean() > 0 else '✗'} Mean per-operon Δ: "
      f"{'OperonAware better' if res_df['delta'].mean() > 0 else 'Standard better'}")
print(f"  {'✓' if res_df['delta'].median() > 0 else '✗'} Median per-operon Δ: "
      f"{'OperonAware better' if res_df['delta'].median() > 0 else 'Standard better'}")
print(f"  {'✓' if contrast_op > contrast_std else '✗'} Intra-vs-inter contrast: "
      f"{'OperonAware better' if contrast_op > contrast_std else 'Standard better'}")
print(f"  {'✓' if pct_improved > 50 else '✗'} % improved operons: "
      f"{'OperonAware better' if pct_improved > 50 else 'Standard better'}")
print(f"  {'✓' if pooled_within_op > pooled_within_std else '✗'} Pooled within-operon r: "
      f"{'OperonAware better' if pooled_within_op > pooled_within_std else 'Standard better'}")
print("=" * 60)