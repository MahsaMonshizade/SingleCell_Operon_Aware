"""
experiment_utils.py
===================
Shared helpers for keeping experiments organized and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    ARCHIVE_ROOT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_NEIGHBOR_LOSS,
    EXPERIMENTS_ROOT,
    HUBER_DELTA,
    LAMBDA_VAL,
    RESULTS_ROOT,
)


def build_experiment_paths(experiment_name: str) -> dict:
    experiment_dir = Path(EXPERIMENTS_ROOT) / experiment_name
    paths = {
        "results_root": Path(RESULTS_ROOT),
        "archive_root": Path(ARCHIVE_ROOT),
        "experiment_dir": experiment_dir,
        "models_dir": experiment_dir / "models",
        "metrics_dir": experiment_dir / "metrics",
        "plots_dir": experiment_dir / "plots",
        "model_standard": experiment_dir / "models" / "standard",
        "model_operon": experiment_dir / "models" / "operon",
        "run_config": experiment_dir / "run_config.json",
        "notes": experiment_dir / "notes.txt",
    }
    return paths


def ensure_experiment_dirs(paths: dict) -> None:
    paths["results_root"].mkdir(parents=True, exist_ok=True)
    paths["archive_root"].mkdir(parents=True, exist_ok=True)
    paths["experiment_dir"].mkdir(parents=True, exist_ok=True)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    paths["plots_dir"].mkdir(parents=True, exist_ok=True)


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    include_model_args: bool = False,
    include_baseline_arg: bool = False,
) -> None:
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Experiment folder name under {EXPERIMENTS_ROOT}.",
    )
    if include_model_args:
        parser.add_argument(
            "--neighbor-loss",
            default=DEFAULT_NEIGHBOR_LOSS,
            choices=["corr_log1p", "huber_log1p", "mse_raw"],
            help="Neighbor regularization used by OperonAwareSCVI.",
        )
        parser.add_argument(
            "--lambda-val",
            type=float,
            default=LAMBDA_VAL,
            help="Strength of the operon regularizer.",
        )
        parser.add_argument(
            "--huber-delta",
            type=float,
            default=HUBER_DELTA,
            help="Huber delta used when neighbor_loss=huber_log1p.",
        )
    if include_baseline_arg:
        parser.add_argument(
            "--baseline-experiment",
            default="shared_baseline",
            help=f"Experiment folder name for the shared standard-model baseline under {EXPERIMENTS_ROOT}.",
        )


def save_run_config(paths: dict, config_data: dict) -> None:
    ensure_experiment_dirs(paths)
    payload = dict(config_data)
    payload["experiment_dir"] = str(paths["experiment_dir"])
    with open(paths["run_config"], "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_run_config(paths: dict) -> dict:
    if not paths["run_config"].exists():
        return {}
    with open(paths["run_config"]) as handle:
        return json.load(handle)


def metrics_path(paths: dict, filename: str) -> str:
    return str(paths["metrics_dir"] / filename)


def plot_path(paths: dict, filename: str) -> str:
    return str(paths["plots_dir"] / filename)
