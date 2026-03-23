"""
scripts/compare_models.py
=========================
Compare Standard SCVI vs OperonAwareSCVI on general metrics.

Metrics:
  1. ELBO
  2. Marginal log-likelihood
  3. Operon consistency score (training pairs)
  4. Latent space silhouette score
  5. UMAP visualisation

Usage (from project root):
    python scripts/compare_models.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scanpy as sc
import scvi
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from scvi.model import SCVI as StandardSCVI
from scvi.model import OperonAwareSCVI

from config import (ADATA_PATH, MODEL_STANDARD, MODEL_OPERON, OPERON_NEIGHBORS,
                    OUT_COMPARISON_PLOT, EVAL_N_CELLS, EVAL_N_MC_SAMPLES,
                    EVAL_RANDOM_SEED)
from operon_aware_lib import load_neighbor_indices

# ── Plot style ───────────────────────────────
DARK  = "#0f1117"
PANEL = "#1a1d27"
ACC1  = "#4fc3f7"   # cyan  – standard
ACC2  = "#f06292"   # pink  – operon
TEXT  = "#e0e0e0"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)


# ═══════════════════════════════════════════
# 0.  Load
# ═══════════════════════════════════════════
print("=" * 50)
print("Loading data and models...")
adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
neighbor_indices = load_neighbor_indices(OPERON_NEIGHBORS, adata)
model_std = StandardSCVI.load(MODEL_STANDARD, adata=adata)
model_op  = OperonAwareSCVI.load(MODEL_OPERON, adata=adata)
print(f"  Cells: {adata.n_obs:,}  |  Genes: {adata.n_vars:,}  |  "
      f"Operon pairs: {neighbor_indices.shape[1]:,}")


# ═══════════════════════════════════════════
# 1.  ELBO
# ═══════════════════════════════════════════
print("\n── 1. ELBO ──")
elbo_std = model_std.get_elbo()
elbo_op  = model_op.get_elbo()
print(f"  Standard : {elbo_std:.2f}")
print(f"  Operon   : {elbo_op:.2f}")
print(f"  Δ        : {elbo_op - elbo_std:+.2f}  "
      f"({'✓ operon' if elbo_op > elbo_std else '✗ standard'})")


# ═══════════════════════════════════════════
# 2.  Marginal log-likelihood
# ═══════════════════════════════════════════
print(f"\n── 2. Marginal Log-Likelihood ({EVAL_N_MC_SAMPLES} MC samples) ──")
mll_std = model_std.get_marginal_ll(n_mc_samples=EVAL_N_MC_SAMPLES)
mll_op  = model_op.get_marginal_ll(n_mc_samples=EVAL_N_MC_SAMPLES)
print(f"  Standard : {mll_std:.2f}")
print(f"  Operon   : {mll_op:.2f}")
print(f"  Δ        : {mll_op - mll_std:+.2f}  "
      f"({'✓ operon' if mll_op > mll_std else '✗ standard'})")


# ═══════════════════════════════════════════
# 3.  Operon consistency score (training pairs)
# ═══════════════════════════════════════════
print("\n── 3. Operon Consistency Score ──")

def operon_consistency(model, neighbor_indices):
    idx    = np.random.choice(model.adata.n_obs,
                              size=min(EVAL_N_CELLS, model.adata.n_obs),
                              replace=False)
    subset = model.adata[idx].copy()
    scvi.data.setup_anndata(subset)
    expr   = model.get_normalized_expression(adata=subset).values
    g1, g2 = neighbor_indices[0].numpy(), neighbor_indices[1].numpy()
    rs = []
    for i in range(len(g1)):
        a, b = expr[:, g1[i]], expr[:, g2[i]]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        r, _ = pearsonr(a, b)
        if not np.isnan(r):
            rs.append(r)
    return np.mean(rs), np.std(rs), rs

np.random.seed(EVAL_RANDOM_SEED)
mean_std, std_std, corrs_std = operon_consistency(model_std, neighbor_indices)
mean_op,  std_op,  corrs_op  = operon_consistency(model_op,  neighbor_indices)
print(f"  Standard : {mean_std:.4f} ± {std_std:.4f}")
print(f"  Operon   : {mean_op:.4f} ± {std_op:.4f}")
print(f"  Δ        : {mean_op - mean_std:+.4f}  "
      f"({'✓ operon' if mean_op > mean_std else '✗ standard'})")


# ═══════════════════════════════════════════
# 4.  Latent space silhouette
# ═══════════════════════════════════════════
print("\n── 4. Latent Space Silhouette Score ──")

def latent_silhouette(model, obsm_key):
    z = model.get_latent_representation()
    adata.obsm[obsm_key] = z
    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=15)
    sc.tl.leiden(adata, key_added=f"{obsm_key}_leiden", resolution=0.5)
    labels = LabelEncoder().fit_transform(adata.obs[f"{obsm_key}_leiden"])
    idx    = np.random.choice(len(z), size=min(5000, len(z)), replace=False)
    return silhouette_score(z[idx], labels[idx], metric="euclidean"), z

sil_std, z_std = latent_silhouette(model_std, "X_scvi_std")
sil_op,  z_op  = latent_silhouette(model_op,  "X_scvi_op")
print(f"  Standard : {sil_std:.4f}")
print(f"  Operon   : {sil_op:.4f}")
print(f"  Δ        : {sil_op - sil_std:+.4f}  "
      f"({'✓ operon' if sil_op > sil_std else '✗ standard'})")


# ═══════════════════════════════════════════
# 5.  Plot
# ═══════════════════════════════════════════
print("\n── 5. Generating plots ──")

# UMAPs
sc.tl.umap(adata, neighbors_key="neighbors")
adata.obsm["X_umap_op"] = adata.obsm["X_umap"].copy()
sc.pp.neighbors(adata, use_rep="X_scvi_std", n_neighbors=15)
sc.tl.umap(adata)
adata.obsm["X_umap_std"] = adata.obsm["X_umap"].copy()

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: UMAPs
for span, umap_key, cluster_key, label, sil in [
    (gs[0, :2], "X_umap_std", "X_scvi_std_leiden", "Standard SCVI",    sil_std),
    (gs[0, 2:], "X_umap_op",  "X_scvi_op_leiden",  "OperonAware SCVI", sil_op),
]:
    ax       = fig.add_subplot(span)
    coords   = adata.obsm[umap_key]
    labels_u = adata.obs[cluster_key].astype("category")
    cmap     = plt.cm.get_cmap("tab20", labels_u.nunique())
    for i, cl in enumerate(labels_u.cat.categories):
        mask = labels_u == cl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=0.4, alpha=0.5, color=cmap(i), rasterized=True)
    style_ax(ax, f"{label}  (silhouette = {sil:.3f})")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

# Row 1: consistency distribution + metric bars
ax_c = fig.add_subplot(gs[1, :2])
bins = np.linspace(-0.2, 1.0, 50)
ax_c.hist(corrs_std, bins=bins, alpha=0.7, color=ACC1,
          label=f"Standard  μ={mean_std:.3f}", density=True)
ax_c.hist(corrs_op,  bins=bins, alpha=0.7, color=ACC2,
          label=f"Operon    μ={mean_op:.3f}",  density=True)
ax_c.axvline(mean_std, color=ACC1, lw=1.5, ls="--")
ax_c.axvline(mean_op,  color=ACC2, lw=1.5, ls="--")
style_ax(ax_c, "Operon Pair Correlation (training pairs)")
ax_c.set_xlabel("Pearson r"); ax_c.set_ylabel("Density")
ax_c.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)

ax_b = fig.add_subplot(gs[1, 2:])
metrics = {
    "ELBO":               (elbo_std, elbo_op),
    "Marginal LL":        (mll_std,  mll_op),
    "Operon Consistency": (mean_std, mean_op),
    "Silhouette":         (sil_std,  sil_op),
}
x, w = np.arange(len(metrics)), 0.35
ax_b.bar(x - w/2, [v[0] for v in metrics.values()], w,
         color=ACC1, alpha=0.85, label="Standard SCVI")
ax_b.bar(x + w/2, [v[1] for v in metrics.values()], w,
         color=ACC2, alpha=0.85, label="Operon SCVI")
ax_b.set_xticks(x)
ax_b.set_xticklabels(list(metrics.keys()), fontsize=8)
style_ax(ax_b, "Summary Metric Comparison")
ax_b.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
ax_b.set_ylabel("Score")

# Row 2: summary table
ax_t = fig.add_subplot(gs[2, :])
ax_t.axis("off"); style_ax(ax_t)

def winner(a, b): return "✓ Operon" if b > a else "✓ Standard"

table_data = [
    ["Metric", "Standard SCVI", "Operon SCVI", "Δ (operon−std)", "Winner"],
    ["ELBO",
     f"{elbo_std:.2f}", f"{elbo_op:.2f}",
     f"{elbo_op-elbo_std:+.2f}", winner(elbo_std, elbo_op)],
    ["Marginal LL",
     f"{mll_std:.2f}", f"{mll_op:.2f}",
     f"{mll_op-mll_std:+.2f}", winner(mll_std, mll_op)],
    ["Operon Consistency",
     f"{mean_std:.4f}", f"{mean_op:.4f}",
     f"{mean_op-mean_std:+.4f}", winner(mean_std, mean_op)],
    ["Silhouette Score",
     f"{sil_std:.4f}", f"{sil_op:.4f}",
     f"{sil_op-sil_std:+.4f}", winner(sil_std, sil_op)],
]
tbl = ax_t.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#23263a" if r == 0 else PANEL)
    cell.set_edgecolor("#2a2d3a")
    cell.set_text_props(color=TEXT)
    if r > 0 and c == 4 and r <= len(table_data) - 1:
        cell.set_facecolor(
            "#1e3a2f" if "Operon" in table_data[r][4] else "#3a1e1e")

fig.suptitle("SCVI vs OperonAware SCVI — Model Comparison",
             color=TEXT, fontsize=15, fontweight="bold", y=0.98)
plt.savefig(OUT_COMPARISON_PLOT, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"  Saved → {OUT_COMPARISON_PLOT}")


# ═══════════════════════════════════════════
# 6.  Verdict
# ═══════════════════════════════════════════
print("\n" + "=" * 50)
scores = {
    "ELBO":               elbo_op  > elbo_std,
    "Marginal LL":        mll_op   > mll_std,
    "Operon Consistency": mean_op  > mean_std,
    "Silhouette Score":   sil_op   > sil_std,
}
for metric, operon_better in scores.items():
    print(f"  {'✓' if operon_better else '✗'} {metric}: "
          f"{'Operon' if operon_better else 'Standard'} wins")
n_wins = sum(scores.values())
print(f"\n  Operon SCVI wins {n_wins}/4 metrics")
if n_wins >= 3:
    print("  → OperonAwareSCVI is the better model.")
elif n_wins == 2:
    print("  → Mixed — consider tuning lambda_val in config.py.")
else:
    print("  → Standard SCVI performs better overall.")
print("=" * 50)