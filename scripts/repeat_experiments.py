"""Run the full PWT-CMMoE pipeline over independent seeds and aggregate test metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _absolute_from_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def _seed_config(base: dict, seed: int, run_dir: Path) -> dict:
    config = deepcopy(base)
    config["seed"] = int(seed)
    config.setdefault("data", {}).setdefault("split", {})["seed"] = int(seed)
    config.setdefault("gan", {})["output_dir"] = str(run_dir / "gan")
    config.setdefault("pretrain", {})["output_dir"] = str(run_dir / "pretrain")
    config["pretrain"]["gan_synthetic_path"] = str(run_dir / "gan" / "gan_synthetic.npz")
    config.setdefault("adapter_finetune", {})["output_dir"] = str(run_dir / "adapter")
    return config


def _aggregate(records: list[dict]) -> dict:
    metric_names = ("mae", "rmse", "r2", "mape")
    result: dict[str, dict[str, dict[str, float]]] = {}
    targets = sorted({target for record in records for target in record["metrics"]})
    for target in targets:
        result[target] = {}
        for metric in metric_names:
            values = [record["metrics"][target][metric] for record in records if metric in record["metrics"].get(target, {})]
            if values:
                result[target][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat complete PWT-CMMoE experiments and aggregate test metrics.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="outputs/repeated_runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = _absolute_from_root(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}
    seeds = args.seeds or base_config.get("experiment", {}).get("repeat_seeds", [])
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("Provide --seeds or experiment.repeat_seeds in the configuration")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique")

    root_output = _absolute_from_root(args.output_dir)
    records: list[dict] = []
    for seed in seeds:
        run_dir = root_output / f"seed_{seed:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_config = _seed_config(base_config, seed, run_dir)
        seed_config_path = run_dir / "config.yaml"
        with seed_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(seed_config, handle, allow_unicode=True, sort_keys=False)
        command = [
            sys.executable, "scripts/train.py", "--config", str(seed_config_path),
            "--pretrain-dir", str(run_dir / "pretrain"),
            "--adapter-dir", str(run_dir / "adapter"),
        ]
        print("$", " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, check=True)
        metric_path = run_dir / "adapter" / "metrics_test.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Seed {seed} completed without {metric_path}")
        with metric_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records.append({"seed": seed, "metrics": payload["metrics"]})

    if not args.dry_run:
        summary = {"seeds": seeds, "n_runs": len(records), "test_metrics": _aggregate(records), "per_seed": records}
        summary_path = root_output / "test_metrics_mean_std.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
