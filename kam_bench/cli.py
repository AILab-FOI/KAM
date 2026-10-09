from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .config import AlgorithmConfig, BenchmarkConfig, ObjectiveWeights
from .experiments import run_benchmark


def _load_config(path: Path) -> BenchmarkConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    objective = ObjectiveWeights(**data.get("objective", {}))
    algorithm = AlgorithmConfig(**data.get("algorithm", {}))
    base = {k: v for k, v in data.items() if k not in {"objective", "algorithm"}}
    return BenchmarkConfig(**base, objective=objective, algorithm=algorithm)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KAM multi-domain reproducible benchmark simulator")
    p.add_argument("--config", type=Path, help="JSON benchmark configuration")
    p.add_argument("--out", type=Path, default=Path("kam_benchmark_output"))
    p.add_argument("--quick", action="store_true", help="Run a small smoke benchmark")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = BenchmarkConfig()
    if args.quick:
        cfg.steps = 12
        cfg.n_agents = 24
        cfg.n_units = 4
        cfg.seeds = [11]
    paths = run_benchmark(cfg, args.out)
    print("KAM benchmark complete")
    for key, path in paths.items():
        if key == "assignments" and not cfg.save_assignments:
            continue
        print(f"  {key:20s} {path}")


if __name__ == "__main__":
    main()
