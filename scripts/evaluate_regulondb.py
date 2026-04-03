"""
scripts/evaluate_regulondb.py
==============================
Evaluate Standard SCVI vs OperonAwareSCVI using RegulonDB ground-truth operons.
Run after compare_models.py.

Metrics:
  1. Per-operon intra-operon correlation (all, Strong, Weak, NaN)
  2. Intra- vs inter-operon contrast
  3. Within-operon gene similarity
  4. Top improved / degraded operons table

Usage (from project root):
    python scripts/evaluate_regulondb.py --experiment corr_log1p_lambda0p01 --baseline-experiment shared_baseline
"""

import argparse
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, mannwhitneyu
from itertools import combinations
import scanpy as sc
import scvi
import warnings
warnings.filterwarnings("ignore")

from scvi.model import SCVI as StandardSCVI
from scvi.model import OperonAwareSCVI

from config import ADATA_PATH, GFF_PATH, OPERON_TSV, EVAL_N_CELLS, EVAL_RANDOM_SEED
from experiment_utils import (
    add_experiment_args,
    build_experiment_paths,
    ensure_experiment_dirs,
    metrics_path,
    plot_path,
)
from operon_aware_lib import build_gene_to_idx, load_operons

# ── Plot style ───────────────────────────────
DARK  = "#0f1117"
PANEL = "#1a1d27"
ACC1  = "#4fc3f7"   # cyan   – Strong
ACC2  = "#f06292"   # pink   – all operons / operon model
ACC3  = "#81c784"   # green  – Confirmed
ACC4  = "#ffd54f"   # amber  – NaN / unknown confidence
TEXT  = "#e0e0e0"

# Colour and display label for every possible confidence value including NaN
CONF_COLORS  = {"Confirmed": ACC3, "Strong": ACC1, "Weak": "#888888", None: ACC4}
CONF_LABEL   = {"Confirmed": "Confirmed", "Strong": "Strong",
                "Weak": "Weak", None: "Unknown"}
# Ordered for consistent display
CONF_ORDER   = ["Confirmed", "Strong", "Weak", None]

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)


# ═══════════════════════════════════════════
# 0.  Load
# ═══════════════════════════════════════════
parser = argparse.ArgumentParser()
add_experiment_args(parser, include_baseline_arg=True)
args = parser.parse_args()

paths = build_experiment_paths(args.experiment)
baseline_paths = build_experiment_paths(args.baseline_experiment)
ensure_experiment_dirs(paths)
out_regulondb_plot = plot_path(paths, "regulondb_operon_evaluation.png")
out_regulondb_csv = metrics_path(paths, "per_operon_results.csv")

print("=" * 50)
print("Loading data, models, and RegulonDB...")
print(f"  Experiment: {args.experiment}")
print(f"  Baseline:   {args.baseline_experiment}")
adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
model_std = StandardSCVI.load(str(baseline_paths["model_standard"]), adata=adata)
model_op  = OperonAwareSCVI.load(str(paths["model_operon"]), adata=adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}")

gene_to_idx            = build_gene_to_idx(GFF_PATH, adata)
operons_valid, operons_all = load_operons(OPERON_TSV, gene_to_idx)


# ═══════════════════════════════════════════
# 1.  Denoised expression
# ═══════════════════════════════════════════
print("\nComputing denoised expression...")
np.random.seed(EVAL_RANDOM_SEED)
if EVAL_N_CELLS is None or EVAL_N_CELLS >= adata.n_obs:
    adata_sub = adata.copy()
else:
    cell_idx = np.random.choice(adata.n_obs, size=EVAL_N_CELLS, replace=False)
    adata_sub = adata[cell_idx].copy()
scvi.data.setup_anndata(adata_sub)
expr_std = model_std.get_normalized_expression(adata=adata_sub).values
expr_op  = model_op.get_normalized_expression(adata=adata_sub).values


# ═══════════════════════════════════════════
# 2.  Per-operon intra-operon correlation
# ═══════════════════════════════════════════
print("Computing per-operon intra-operon correlations...")

def safe_pearsonr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return pearsonr(a, b)[0]

def mean_pairwise_corr(expr, gidx):
    rs = [safe_pearsonr(expr[:, i], expr[:, j])
          for i, j in combinations(gidx, 2)]
    rs = [r for r in rs if not np.isnan(r)]
    return np.mean(rs) if rs else np.nan

results = []
for _, row in operons_valid.iterrows():
    gidx = row["idx_found"]
    results.append({
        "operon_name": row["operon_name"],
        "n_genes":     len(gidx),
        "confidence":  row["confidence_label"],   # may be None/NaN
        "r_standard":  mean_pairwise_corr(expr_std, gidx),
        "r_operon":    mean_pairwise_corr(expr_op,  gidx),
    })

res_df = pd.DataFrame(results)
print(f"  Raw: {len(res_df)} operons | "
      f"NaN r_std: {res_df['r_standard'].isna().sum()} | "
      f"NaN r_op: {res_df['r_operon'].isna().sum()} | "
      f"NaN confidence: {res_df['confidence'].isna().sum()}")

# ── Only drop rows where the CORRELATION itself couldn't be computed.
#    Rows with NaN confidence are valid operons and are kept.
res_df = res_df.dropna(subset=["r_standard", "r_operon"]).reset_index(drop=True)
res_df["delta"] = res_df["r_operon"] - res_df["r_standard"]
pct_improved    = (res_df["delta"] > 0).mean() * 100

_, pval = mannwhitneyu(res_df["r_operon"], res_df["r_standard"],
                       alternative="greater")

print(f"\n  Overall ({len(res_df)} operons, "
      f"NaN-confidence kept: {res_df['confidence'].isna().sum()}):")
print(f"    Standard : {res_df['r_standard'].mean():.4f} ± "
      f"{res_df['r_standard'].std():.4f}")
print(f"    Operon   : {res_df['r_operon'].mean():.4f} ± "
      f"{res_df['r_operon'].std():.4f}")
print(f"    Δ        : {res_df['delta'].mean():+.4f}  "
      f"(Mann-Whitney p={pval:.3e})")
print(f"    % improved: {pct_improved:.1f}%")

print("\n  By confidence level (NaN = confidence field missing/unrecognised in TSV):")
for conf in ["Confirmed", "Strong", "Weak", None]:
    if conf is None:
        sub = res_df[res_df["confidence"].isna()]
        label = "NaN"
    else:
        sub   = res_df[res_df["confidence"] == conf]
        label = conf
    if len(sub):
        print(f"    [{label:9s}] n={len(sub):4d} | "
              f"Std={sub['r_standard'].mean():.4f}  "
              f"Op={sub['r_operon'].mean():.4f}  "
              f"Δ={sub['delta'].mean():+.4f}  "
              f"({(sub['delta']>0).mean()*100:.0f}% improved)")

# Strong-only (most reliable ground truth)
strong = res_df[res_df["confidence"] == "Strong"]
if len(strong):
    print(f"\n  ── Strong operons only (n={len(strong)}) ──")
    print(f"    Δ = {strong['delta'].mean():+.4f}  |  "
          f"{(strong['delta']>0).mean()*100:.1f}% improved")

weak = res_df[res_df["confidence"] == "Weak"]


# ═══════════════════════════════════════════
# 3.  Intra- vs inter-operon contrast
# ═══════════════════════════════════════════
print("\nComputing intra- vs inter-operon contrast...")

intra_pairs = [(i, j)
               for _, row in operons_valid.iterrows()
               for i, j in combinations(row["idx_found"], 2)]
intra_set = set(intra_pairs)

np.random.seed(0)
all_idx, inter_pairs = list(range(adata.n_vars)), []
while len(inter_pairs) < len(intra_pairs):
    i, j = np.random.choice(all_idx, 2, replace=False)
    if (i, j) not in intra_set and (j, i) not in intra_set:
        inter_pairs.append((i, j))

def batch_corr(expr, pairs):
    rs = [safe_pearsonr(expr[:, i], expr[:, j]) for i, j in pairs]
    return np.array([r for r in rs if not np.isnan(r)])

intra_std    = batch_corr(expr_std, intra_pairs)
intra_op     = batch_corr(expr_op,  intra_pairs)
inter_std    = batch_corr(expr_std, inter_pairs)
inter_op     = batch_corr(expr_op,  inter_pairs)
contrast_std = intra_std.mean() - inter_std.mean()
contrast_op  = intra_op.mean()  - inter_op.mean()

print(f"  Intra | Std={intra_std.mean():.4f}  Op={intra_op.mean():.4f}")
print(f"  Inter | Std={inter_std.mean():.4f}  Op={inter_op.mean():.4f}")
print(f"  Δr    | Std={contrast_std:.4f}  Op={contrast_op:.4f}  "
      f"({'✓ operon' if contrast_op > contrast_std else '✗ standard'})")


# ═══════════════════════════════════════════
# 4.  Within-operon gene similarity
# ═══════════════════════════════════════════
print("\nComputing within-operon gene similarity...")

prox_std, prox_op = [], []
for _, row in operons_valid.head(200).iterrows():
    gidx = row["idx_found"]
    for expr, store in [(expr_std, prox_std), (expr_op, prox_op)]:
        sub  = expr[:, gidx].T
        corr = np.corrcoef(sub)
        triu = np.triu_indices(len(gidx), k=1)
        store.append(corr[triu].mean())

print(f"  Std={np.mean(prox_std):.4f}  Op={np.mean(prox_op):.4f}  "
      f"({'✓ operon' if np.mean(prox_op) > np.mean(prox_std) else '✗ standard'})")


# ═══════════════════════════════════════════
# 5.  Top improved / degraded operons
# ═══════════════════════════════════════════
print(f"\nTop 10 most improved operons:")
print(res_df.nlargest(10, "delta")[
    ["operon_name", "n_genes", "confidence", "r_standard", "r_operon", "delta"]
].to_string(index=False))

print(f"\nTop 10 where standard was better:")
print(res_df.nsmallest(10, "delta")[
    ["operon_name", "n_genes", "confidence", "r_standard", "r_operon", "delta"]
].to_string(index=False))


# ═══════════════════════════════════════════
# 6.  Plot
# ═══════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

# 6a. Per-operon scatter — groupby with dropna=False so NaN group appears
ax1 = fig.add_subplot(gs[0, 0])
for conf, grp in res_df.groupby("confidence", dropna=False):
    label = CONF_LABEL.get(conf, "Unknown") if conf is not None else "Unknown"
    color = CONF_COLORS.get(conf, "#aaaaaa")
    ax1.scatter(grp["r_standard"], grp["r_operon"], s=12, alpha=0.6,
                color=color, label=label, rasterized=True)
lims = [min(res_df[["r_standard","r_operon"]].min())-0.02,
        max(res_df[["r_standard","r_operon"]].max())+0.02]
ax1.plot(lims, lims, "w--", lw=0.8, alpha=0.5)
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("Standard SCVI intra-operon r")
ax1.set_ylabel("Operon SCVI intra-operon r")
ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=7, markerscale=1.5)
style_ax(ax1, "Per-Operon Intra-Correlation\n(above diagonal = operon better)")

# 6b. Boxplot by confidence — include NaN as its own box
ax2 = fig.add_subplot(gs[0, 1])
# Build groups in display order, including NaN
box_data, box_labels, box_colors = [], [], []
for conf in CONF_ORDER:
    if conf is None:
        sub = res_df[res_df["confidence"].isna()]["delta"].values
        lbl = "Unknown"
    else:
        sub = res_df[res_df["confidence"] == conf]["delta"].values
        lbl = conf
    if len(sub) > 0:
        box_data.append(sub)
        box_labels.append(lbl)
        box_colors.append(CONF_COLORS[conf])

bp = ax2.boxplot(box_data, patch_artist=True, notch=False,
                 medianprops=dict(color="white", lw=2),
                 whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                 flierprops=dict(marker=".", color="#555", markersize=3))
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax2.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
ax2.set_xticklabels(box_labels, fontsize=8)
ax2.set_ylabel("Δ intra-operon r (Operon − Standard)")
style_ax(ax2, "Improvement by Confidence Level\n(Strong = most reliable)")

# 6c. Intra vs inter contrast
ax3 = fig.add_subplot(gs[0, 2])
x, w = np.arange(2), 0.35
for offset, means, errs, col, label in [
    (-w/2, [intra_std.mean(), inter_std.mean()],
           [intra_std.std()/np.sqrt(len(intra_std)),
            inter_std.std()/np.sqrt(len(inter_std))],
           ACC1, "Standard SCVI"),
    (+w/2, [intra_op.mean(), inter_op.mean()],
           [intra_op.std()/np.sqrt(len(intra_op)),
            inter_op.std()/np.sqrt(len(inter_op))],
           ACC2, "Operon SCVI"),
]:
    ax3.bar(x+offset, means, w, color=col, alpha=0.85, label=label,
            yerr=errs, capsize=4, error_kw=dict(color=TEXT))
ax3.set_xticks(x)
ax3.set_xticklabels(["Intra-operon", "Inter-operon\n(random)"])
ax3.set_ylabel("Mean Pearson r")
ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax3, "Intra- vs Inter-Operon Contrast\n(larger gap = better)")

# 6d. Delta histogram — separate overlay for NaN group
ax4 = fig.add_subplot(gs[1, :2])
bins = np.linspace(res_df["delta"].min()-0.01, res_df["delta"].max()+0.01, 60)
nan_mask = res_df["confidence"].isna()

ax4.hist(res_df["delta"], bins=bins, color=ACC2, alpha=0.7,
         edgecolor="none", label="All operons")
if len(strong):
    ax4.hist(strong["delta"], bins=bins, color=ACC3, alpha=0.7,
             edgecolor="none", label="Strong only")
if len(weak):
    ax4.hist(weak["delta"], bins=bins, color="#888888", alpha=0.7,
             edgecolor="none", label=f"Weak (n={len(weak)})")
    
if nan_mask.sum() > 0:
    ax4.hist(res_df.loc[nan_mask, "delta"], bins=bins, color=ACC4, alpha=0.7,
             edgecolor="none", label=f"Unknown confidence (n={nan_mask.sum()})")

ax4.axvline(0, color="white", lw=1.2, ls="--")
ax4.axvline(res_df["delta"].mean(), color=ACC2, lw=2, ls="-",
            label=f"All mean = {res_df['delta'].mean():+.4f}")
if len(strong):
    ax4.axvline(strong["delta"].mean(), color=ACC3, lw=2, ls="-",
                label=f"Strong mean = {strong['delta'].mean():+.4f}")
if len(weak):
    weak_mean = weak["delta"].mean()
    ax4.axvline(weak_mean, color="#888888", lw=2, ls="-",
                label=f"Weak mean = {weak_mean:+.4f}")
if nan_mask.sum() > 0:
    nan_mean = res_df.loc[nan_mask, "delta"].mean()
    ax4.axvline(nan_mean, color=ACC4, lw=2, ls="-",
                label=f"Unknown mean = {nan_mean:+.4f}")

ax4.text(0.97, 0.88, f"{pct_improved:.1f}% of operons improved",
         transform=ax4.transAxes, ha="right", color=TEXT, fontsize=9,
         bbox=dict(facecolor=PANEL, edgecolor="#2a2d3a", pad=4))
ax4.set_xlabel("Δ intra-operon r (Operon − Standard)")
ax4.set_ylabel("Number of operons")
ax4.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax4, "Distribution of Per-Operon Improvement")

# 6e. Within-operon gene similarity
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(prox_std, prox_op, s=15, alpha=0.5, color=ACC2, rasterized=True)
lims2 = [min(min(prox_std), min(prox_op))-0.02,
         max(max(prox_std), max(prox_op))+0.02]
ax5.plot(lims2, lims2, "w--", lw=0.8, alpha=0.5)
ax5.set_xlabel("Standard SCVI within-operon similarity")
ax5.set_ylabel("Operon SCVI within-operon similarity")
style_ax(ax5, "Within-Operon Gene Similarity\n(above diagonal = operon better)")

# 6f. Summary table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis("off"); style_ax(ax6)

def winner(a, b): return "✓ Operon" if b > a else "✓ Standard"

s_std = strong["r_standard"].mean() if len(strong) else float("nan")
s_op  = strong["r_operon"].mean()   if len(strong) else float("nan")
s_d   = strong["delta"].mean()      if len(strong) else float("nan")

nan_sub  = res_df[res_df["confidence"].isna()]
nan_std  = nan_sub["r_standard"].mean() if len(nan_sub) else float("nan")
nan_op   = nan_sub["r_operon"].mean()   if len(nan_sub) else float("nan")
nan_d    = nan_sub["delta"].mean()      if len(nan_sub) else float("nan")

table_data = [
    ["Metric", "Standard SCVI", "Operon SCVI", "Δ", "p-value", "Winner"],
    ["Mean intra-operon r (all)",
     f"{res_df['r_standard'].mean():.4f}", f"{res_df['r_operon'].mean():.4f}",
     f"{res_df['delta'].mean():+.4f}", f"{pval:.3e}",
     winner(res_df['r_standard'].mean(), res_df['r_operon'].mean())],
    ["Mean intra-operon r (Strong only)",
     f"{s_std:.4f}" if not np.isnan(s_std) else "N/A",
     f"{s_op:.4f}"  if not np.isnan(s_op)  else "N/A",
     f"{s_d:+.4f}"  if not np.isnan(s_d)   else "N/A", "—",
     winner(s_std, s_op) if not np.isnan(s_std) else "N/A"],
    ["Mean intra-operon r (Unknown conf.)",
     f"{nan_std:.4f}" if not np.isnan(nan_std) else "N/A",
     f"{nan_op:.4f}"  if not np.isnan(nan_op)  else "N/A",
     f"{nan_d:+.4f}"  if not np.isnan(nan_d)   else "N/A", "—",
     winner(nan_std, nan_op) if not np.isnan(nan_std) else "N/A"],
    ["Intra-inter contrast (Δr)",
     f"{contrast_std:.4f}", f"{contrast_op:.4f}",
     f"{contrast_op-contrast_std:+.4f}", "—",
     winner(contrast_std, contrast_op)],
    ["% operons improved", "—", f"{pct_improved:.1f}%", "—", "—",
     "✓ Operon" if pct_improved > 50 else "✓ Standard"],
    ["Within-operon gene similarity",
     f"{np.mean(prox_std):.4f}", f"{np.mean(prox_op):.4f}",
     f"{np.mean(prox_op)-np.mean(prox_std):+.4f}", "—",
     winner(np.mean(prox_std), np.mean(prox_op))],
]
tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#23263a" if r == 0 else PANEL)
    cell.set_edgecolor("#2a2d3a")
    cell.set_text_props(color=TEXT)
    if r > 0 and c == 5 and r <= len(table_data) - 1:
        txt = table_data[r][5]
        cell.set_facecolor(
            "#1e3a2f" if "Operon" in txt else
            "#3a1e1e" if "Standard" in txt else PANEL)

fig.suptitle("OperonAware SCVI vs Standard SCVI — RegulonDB Evaluation",
             color=TEXT, fontsize=14, fontweight="bold", y=0.99)
plt.savefig(out_regulondb_plot, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\nSaved → {out_regulondb_plot}")
res_df.to_csv(out_regulondb_csv, index=False)
print(f"Saved → {out_regulondb_csv}")


# ═══════════════════════════════════════════
# 7.  Verdict
# ═══════════════════════════════════════════
print("\n" + "=" * 50)
wins = {
    "Intra-operon correlation":      res_df["delta"].mean() > 0,
    "Intra-inter contrast":          contrast_op > contrast_std,
    "% operons improved (>50%)":     pct_improved > 50,
    "Within-operon gene similarity": np.mean(prox_op) > np.mean(prox_std),
}
for metric, operon_wins in wins.items():
    print(f"  {'✓' if operon_wins else '✗'} {metric}: "
          f"{'Operon' if operon_wins else 'Standard'} wins")
print(f"\n  Operon SCVI wins {sum(wins.values())}/{len(wins)} RegulonDB metrics")
print("=" * 50)
