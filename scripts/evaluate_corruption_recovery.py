"""
scripts/evaluate_corruption_recovery.py
======================================
Direct denoising benchmark by corrupting observed counts and testing which model
better recovers the original signal.

Corruptions:
  1. Random masking of a fraction of nonzero entries to zero
  2. Binomial downsampling of observed counts

Metrics:
  1. Masked-entry recovery correlation
  2. Masked-entry recovery MSE on log1p normalized scale
  3. Downsampled-entry recovery correlation
  4. Downsampled-entry recovery MSE on log1p normalized scale

Usage (from project root):
    python scripts/evaluate_corruption_recovery.py --experiment corr_log1p_lambda0p01
"""

import argparse
import sys
sys.path.insert(0, ".")

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse
from scipy.stats import pearsonr

from scvi.model import SCVI as StandardSCVI
from scvi.model import OperonAwareSCVI

from config import (
    ADATA_PATH,
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

MASK_FRAC = 0.10
DOWNSAMPLE_KEEP_PROB = 0.50
MAX_EVAL_POINTS = 500_000
TARGET_SUM = 1e4


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


def sample_points(a, b, limit, seed):
    n = len(a)
    if n <= limit:
        return a, b
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=limit, replace=False)
    return a[idx], b[idx]


def normalize_log1p(x):
    x = np.asarray(x, dtype=np.float64)
    lib = x.sum(axis=1, keepdims=True)
    scale = np.divide(
        TARGET_SUM,
        lib,
        out=np.zeros_like(lib, dtype=np.float64),
        where=lib > 0,
    )
    return np.log1p(x * scale)


def make_masked_copy(adata, mask_frac, seed):
    x = adata.X.tocsr(copy=True) if sparse.issparse(adata.X) else np.array(adata.X, copy=True)
    rng = np.random.default_rng(seed)

    if sparse.issparse(x):
        coo = x.tocoo(copy=True)
        nnz = coo.data.size
        n_mask = int(mask_frac * nnz)
        mask_idx = rng.choice(nnz, size=n_mask, replace=False)

        masked_rows = coo.row[mask_idx]
        masked_cols = coo.col[mask_idx]
        original_vals = coo.data[mask_idx].copy()

        coo.data[mask_idx] = 0
        x_masked = coo.tocsr()
        x_masked.eliminate_zeros()
    else:
        nz = np.argwhere(x > 0)
        n_mask = int(mask_frac * len(nz))
        chosen = rng.choice(len(nz), size=n_mask, replace=False)
        masked_rows = nz[chosen, 0]
        masked_cols = nz[chosen, 1]
        original_vals = x[masked_rows, masked_cols].copy()
        x[masked_rows, masked_cols] = 0
        x_masked = x

    adata_masked = adata.copy()
    adata_masked.X = x_masked
    return adata_masked, masked_rows, masked_cols, original_vals


def make_downsampled_copy(adata, keep_prob, seed):
    x = adata.X.tocsr(copy=True) if sparse.issparse(adata.X) else np.array(adata.X, copy=True)
    rng = np.random.default_rng(seed)

    if sparse.issparse(x):
        x.data = rng.binomial(x.data.astype(np.int64), keep_prob).astype(x.data.dtype)
        x.eliminate_zeros()
    else:
        x = rng.binomial(x.astype(np.int64), keep_prob).astype(x.dtype)

    adata_down = adata.copy()
    adata_down.X = x
    return adata_down


def eval_against_original(original_vals, pred_vals):
    y_true = np.log1p(np.asarray(original_vals, dtype=np.float64))
    y_pred = np.log1p(np.asarray(pred_vals, dtype=np.float64))
    mse = float(np.mean((y_true - y_pred) ** 2))
    corr = float(safe_corr(y_true, y_pred))
    return corr, mse, y_true, y_pred


print("=" * 60)
print("Corruption recovery evaluation")
print("=" * 60)

parser = argparse.ArgumentParser()
add_experiment_args(parser)
args = parser.parse_args()

paths = build_experiment_paths(args.experiment)
ensure_experiment_dirs(paths)
out_plot = plot_path(paths, "corruption_recovery_evaluation.png")
out_csv = metrics_path(paths, "corruption_recovery_metrics.csv")

print("\nLoading data and models...")
print(f"  Experiment: {args.experiment}")
adata = sc.read_h5ad(ADATA_PATH)
scvi.data.setup_anndata(adata)
model_std = StandardSCVI.load(str(paths["model_standard"]), adata=adata)
model_op = OperonAwareSCVI.load(str(paths["model_operon"]), adata=adata)

if EVAL_N_CELLS is None or EVAL_N_CELLS >= adata.n_obs:
    adata_eval = adata.copy()
else:
    np.random.seed(EVAL_RANDOM_SEED)
    cell_idx = np.random.choice(adata.n_obs, size=EVAL_N_CELLS, replace=False)
    adata_eval = adata[cell_idx].copy()

scvi.data.setup_anndata(adata_eval)
print(f"  Cells used: {adata_eval.n_obs:,}")
print(f"  Genes used: {adata_eval.n_vars:,}")

orig_dense = adata_eval.X.toarray() if sparse.issparse(adata_eval.X) else np.asarray(adata_eval.X)
orig_norm_log = normalize_log1p(orig_dense)


print("\n1. Random masking corruption...")
adata_masked, mask_rows, mask_cols, mask_true = make_masked_copy(
    adata_eval, MASK_FRAC, EVAL_RANDOM_SEED
)
scvi.data.setup_anndata(adata_masked)

mask_std = model_std.get_normalized_expression(adata=adata_masked).values
mask_op = model_op.get_normalized_expression(adata=adata_masked).values

mask_true_norm = orig_norm_log[mask_rows, mask_cols]
mask_pred_std = np.log1p(mask_std[mask_rows, mask_cols])
mask_pred_op = np.log1p(mask_op[mask_rows, mask_cols])

mask_corr_std, mask_mse_std, mask_true_log, mask_pred_std_log = eval_against_original(
    np.expm1(mask_true_norm), np.expm1(mask_pred_std)
)
mask_corr_op, mask_mse_op, _, mask_pred_op_log = eval_against_original(
    np.expm1(mask_true_norm), np.expm1(mask_pred_op)
)

print(f"  Standard masked-entry corr: {mask_corr_std:.4f}")
print(f"  Operon   masked-entry corr: {mask_corr_op:.4f}")
print(f"  Standard masked-entry MSE : {mask_mse_std:.4f}")
print(f"  Operon   masked-entry MSE : {mask_mse_op:.4f}")


print("\n2. Count downsampling corruption...")
adata_down = make_downsampled_copy(adata_eval, DOWNSAMPLE_KEEP_PROB, EVAL_RANDOM_SEED + 1)
scvi.data.setup_anndata(adata_down)

down_std = model_std.get_normalized_expression(adata=adata_down).values
down_op = model_op.get_normalized_expression(adata=adata_down).values

down_dense = adata_down.X.toarray() if sparse.issparse(adata_down.X) else np.asarray(adata_down.X)
changed_mask = (orig_dense > 0) & (down_dense != orig_dense)

orig_log = orig_norm_log[changed_mask]
down_std_log = np.log1p(down_std[changed_mask])
down_op_log = np.log1p(down_op[changed_mask])

orig_log_s, down_std_log_s = sample_points(orig_log, down_std_log, MAX_EVAL_POINTS, EVAL_RANDOM_SEED)
_, down_op_log_s = sample_points(orig_log, down_op_log, MAX_EVAL_POINTS, EVAL_RANDOM_SEED)

down_corr_std = float(safe_corr(orig_log_s, down_std_log_s))
down_corr_op = float(safe_corr(orig_log_s, down_op_log_s))
down_mse_std = float(np.mean((orig_log_s - down_std_log_s) ** 2))
down_mse_op = float(np.mean((orig_log_s - down_op_log_s) ** 2))

print(f"  Standard downsample corr: {down_corr_std:.4f}")
print(f"  Operon   downsample corr: {down_corr_op:.4f}")
print(f"  Standard downsample MSE : {down_mse_std:.4f}")
print(f"  Operon   downsample MSE : {down_mse_op:.4f}")


metrics_df = pd.DataFrame(
    [
        {
            "metric": "masked_entry_corr",
            "standard": mask_corr_std,
            "operon": mask_corr_op,
            "winner": "operon" if mask_corr_op > mask_corr_std else "standard",
            "direction": "higher_better",
        },
        {
            "metric": "masked_entry_mse",
            "standard": mask_mse_std,
            "operon": mask_mse_op,
            "winner": "operon" if mask_mse_op < mask_mse_std else "standard",
            "direction": "lower_better",
        },
        {
            "metric": "downsample_recovery_corr",
            "standard": down_corr_std,
            "operon": down_corr_op,
            "winner": "operon" if down_corr_op > down_corr_std else "standard",
            "direction": "higher_better",
        },
        {
            "metric": "downsample_recovery_mse",
            "standard": down_mse_std,
            "operon": down_mse_op,
            "winner": "operon" if down_mse_op < down_mse_std else "standard",
            "direction": "lower_better",
        },
    ]
)
metrics_df["delta_operon_minus_standard"] = metrics_df["operon"] - metrics_df["standard"]

metrics_df.to_csv(out_csv, index=False)
print(f"\nSaved metrics → {out_csv}")


fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(DARK)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
true_s, pred_std_s = sample_points(mask_true_log, mask_pred_std_log, 80000, EVAL_RANDOM_SEED)
_, pred_op_s = sample_points(mask_true_log, mask_pred_op_log, 80000, EVAL_RANDOM_SEED)
ax1.scatter(true_s, pred_std_s, s=6, alpha=0.2, color=ACC_STD, label="Standard", rasterized=True)
ax1.scatter(true_s, pred_op_s, s=6, alpha=0.15, color=ACC_OP, label="Operon", rasterized=True)
lims = [0, max(true_s.max(), pred_std_s.max(), pred_op_s.max())]
ax1.plot(lims, lims, "w--", linewidth=0.8, alpha=0.6)
ax1.set_xlabel("log1p(original masked counts)")
ax1.set_ylabel("log1p(recovered expression)")
ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax1, "Masked-Entry Recovery")

ax2 = fig.add_subplot(gs[0, 1])
labels = ["Std", "Operon"]
ax2.bar(labels, [mask_corr_std, mask_corr_op], color=[ACC_STD, ACC_OP], alpha=0.9)
ax2.set_ylabel("Correlation")
style_ax(ax2, "Masked Recovery Correlation")

ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(labels, [mask_mse_std, mask_mse_op], color=[ACC_STD, ACC_OP], alpha=0.9)
ax3.set_ylabel("MSE on log1p scale")
style_ax(ax3, "Masked Recovery MSE")

ax4 = fig.add_subplot(gs[1, 1])
x = np.arange(2)
width = 0.36
ax4.bar(x - width / 2, [down_corr_std, down_mse_std], width=width, color=ACC_STD, alpha=0.9, label="Standard")
ax4.bar(x + width / 2, [down_corr_op, down_mse_op], width=width, color=ACC_OP, alpha=0.9, label="Operon")
ax4.set_xticks(x)
ax4.set_xticklabels(["Changed-entry corr", "Changed-entry MSE"])
ax4.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
style_ax(ax4, "Downsampling Recovery")

plt.tight_layout()
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
print(f"Saved plot    → {out_plot}")
