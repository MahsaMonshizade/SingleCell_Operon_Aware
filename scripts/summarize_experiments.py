"""
scripts/summarize_experiments.py
================================
Collect evaluation outputs across experiment folders into one summary table.

Usage (from project root):
    python scripts/summarize_experiments.py --baseline-experiment shared_baseline
"""

import argparse
import json
import sys
sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd

from config import EXPERIMENTS_ROOT, RESULTS_ROOT
from experiment_utils import build_experiment_paths


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--baseline-experiment",
    default="shared_baseline",
    help="Shared baseline experiment name. Used only for labeling the summary.",
)
args = parser.parse_args()

experiments_root = Path(EXPERIMENTS_ROOT)
summary_rows = []

for exp_dir in sorted(experiments_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    experiment = exp_dir.name
    if experiment == args.baseline_experiment:
        continue

    paths = build_experiment_paths(experiment)
    run_cfg = read_json(paths["run_config"])

    regulondb = read_csv(Path(paths["metrics_dir"]) / "per_operon_results.csv")
    proximity = read_csv(Path(paths["metrics_dir"]) / "genome_proximity_metrics.csv")
    corruption = read_csv(Path(paths["metrics_dir"]) / "corruption_recovery_metrics.csv")

    row = {
        "experiment": experiment,
        "baseline_experiment": args.baseline_experiment,
        "neighbor_loss": run_cfg.get("neighbor_loss"),
        "lambda_val": run_cfg.get("lambda_val"),
        "huber_delta": run_cfg.get("huber_delta"),
    }

    if regulondb is not None and "delta" in regulondb.columns:
        row["regulondb_mean_delta"] = regulondb["delta"].mean()
        row["regulondb_median_delta"] = regulondb["delta"].median()
        row["regulondb_pct_improved"] = (regulondb["delta"] > 0).mean() * 100
        strong = regulondb[regulondb["confidence"] == "Strong"]
        if len(strong):
            row["regulondb_strong_mean_delta"] = strong["delta"].mean()
    else:
        row["regulondb_mean_delta"] = None
        row["regulondb_median_delta"] = None
        row["regulondb_pct_improved"] = None
        row["regulondb_strong_mean_delta"] = None

    if proximity is not None:
        prox_idx = proximity.set_index("group")
        for group in [
            "adjacent_same_training_neighbor",
            "adjacent_same_nonneighbor",
            "adjacent_opposite_strand",
            "random_pairs",
        ]:
            if group in prox_idx.index:
                row[f"proximity_delta__{group}"] = prox_idx.loc[group, "delta"]
    else:
        row["proximity_delta__adjacent_same_training_neighbor"] = None

    if corruption is not None:
        corr_idx = corruption.set_index("metric")
        for metric in [
            "masked_entry_corr",
            "masked_entry_mse",
            "downsample_recovery_corr",
            "downsample_recovery_mse",
        ]:
            if metric in corr_idx.index:
                row[f"corruption__{metric}"] = corr_idx.loc[metric, "delta_operon_minus_standard"]
    else:
        row["corruption__masked_entry_corr"] = None

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty and "lambda_val" in summary_df.columns:
    summary_df = summary_df.sort_values(["neighbor_loss", "lambda_val", "experiment"], na_position="last")

out_csv = Path(RESULTS_ROOT) / "experiment_summary.csv"
summary_df.to_csv(out_csv, index=False)

print(f"Saved summary → {out_csv}")
if summary_df.empty:
    print("No evaluated experiments found.")
else:
    display_cols = [
        col for col in [
            "experiment",
            "neighbor_loss",
            "lambda_val",
            "regulondb_mean_delta",
            "regulondb_pct_improved",
            "regulondb_strong_mean_delta",
            "proximity_delta__adjacent_same_training_neighbor",
            "proximity_delta__random_pairs",
            "corruption__masked_entry_corr",
            "corruption__masked_entry_mse",
        ] if col in summary_df.columns
    ]
    print(summary_df[display_cols].to_string(index=False))
