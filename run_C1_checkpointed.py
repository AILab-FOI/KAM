from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy

from kam_bench.cli import _load_config
from kam_bench.experiments import (
    _aggregate_runs,
    _optional_backend_info,
    _preflight_dependencies,
    _relative_to_static,
    _source_hashes,
    _trace_sha256,
)
from kam_bench.scenarios import SCENARIO_SPECS, generate_trace
from kam_bench.simulation import run_one
from kam_bench.statistics import ckam_dkam_comparison, friedman_omnibus, paired_method_comparisons

PROTOCOL_ID = "C1.1-confirmatory-hotfix"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_config_hash(cfg) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def checkpoint_dir(root: Path, scenario: str, seed: int, method: str) -> Path:
    return root / "checkpoints" / scenario / str(seed) / method


def checkpoint_complete(path: Path, context_hash: str, trace_hash: str) -> bool:
    marker = path / "complete.json"
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("protocol_id") == PROTOCOL_ID
        and data.get("context_hash") == context_hash
        and data.get("trace_sha256") == trace_hash
        and (path / "steps.csv").exists()
        and (path / "summary.json").exists()
    )


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_checkpoint(path: Path, steps: pd.DataFrame, events: pd.DataFrame, summary: Dict[str, object], *, protocol_id: str, context_hash: str, trace_hash: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    steps_tmp = path / "steps.csv.tmp"
    steps.to_csv(steps_tmp, index=False)
    os.replace(steps_tmp, path / "steps.csv")
    events_path = path / "events.csv"
    if events.empty:
        if events_path.exists():
            events_path.unlink()
    else:
        events_tmp = path / "events.csv.tmp"
        events.to_csv(events_tmp, index=False)
        os.replace(events_tmp, events_path)
    atomic_write_text(path / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    marker = {
        "protocol_id": protocol_id,
        "context_hash": context_hash,
        "trace_sha256": trace_hash,
        "status": "complete",
    }
    atomic_write_text(path / "complete.json", json.dumps(marker, indent=2, sort_keys=True))


def load_checkpoint(path: Path):
    steps = pd.read_csv(path / "steps.csv")
    events_path = path / "events.csv"
    events = pd.read_csv(events_path) if events_path.exists() and events_path.stat().st_size > 0 else pd.DataFrame()
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return steps, events, summary


def finalize(cfg, out_dir: Path, context: Dict[str, object], trace_manifest: List[Dict[str, object]]) -> Dict[str, Path]:
    all_steps: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, object]] = []

    for scenario in cfg.scenarios:
        methods = cfg.scenario_methods.get(scenario, cfg.methods)
        for seed in cfg.seeds:
            for method in methods:
                cp = checkpoint_dir(out_dir, scenario, seed, method)
                if not (cp / "complete.json").exists():
                    raise RuntimeError(f"missing completed checkpoint: {cp}")
                steps, events, summary = load_checkpoint(cp)
                all_steps.append(steps)
                if not events.empty:
                    all_events.append(events)
                run_summaries.append({"scenario": scenario, "seed": seed, "method": method, **summary})

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    steps_df = pd.concat(all_steps, ignore_index=True) if all_steps else pd.DataFrame()
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    run_summary_df = pd.DataFrame(run_summaries)
    aggregate_df = _aggregate_runs(run_summary_df)
    relative_df = _relative_to_static(run_summary_df)
    stats_df = paired_method_comparisons(run_summary_df)
    omnibus_df = friedman_omnibus(run_summary_df)
    c_vs_d_df = ckam_dkam_comparison(run_summary_df)

    paths = {
        "steps": raw_dir / "step_metrics.csv",
        "events": raw_dir / "events.csv",
        "run_summary": out_dir / "run_summary.csv",
        "aggregate_summary": out_dir / "aggregate_summary.csv",
        "relative_to_static": out_dir / "relative_to_static.csv",
        "statistical_comparisons": out_dir / "statistical_comparisons.csv",
        "omnibus_statistics": out_dir / "omnibus_statistics.csv",
        "ckam_vs_dkam": out_dir / "ckam_vs_dkam.csv",
        "manifest": out_dir / "manifest.json",
    }
    steps_df.to_csv(paths["steps"], index=False)
    events_df.to_csv(paths["events"], index=False)
    run_summary_df.to_csv(paths["run_summary"], index=False)
    aggregate_df.to_csv(paths["aggregate_summary"], index=False)
    relative_df.to_csv(paths["relative_to_static"], index=False)
    stats_df.to_csv(paths["statistical_comparisons"], index=False)
    omnibus_df.to_csv(paths["omnibus_statistics"], index=False)
    c_vs_d_df.to_csv(paths["ckam_vs_dkam"], index=False)

    manifest = {
        "confirmatory_protocol": PROTOCOL_ID,
        "base_protocol": "C1-confirmatory-v1",
        "amendment": "Implementation hotfix only: accept empty physical ParMETIS target partitions and checkpoint each scenario/seed/method run. No algorithm parameter, metric, generator, method panel, or seed changed.",
        "config": cfg.to_dict(),
        "traces": trace_manifest,
        "source_sha256": _source_hashes(),
        "checkpoint_context": context,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "optional_backends": _optional_backend_info(),
        },
        "notes": {
            "paired_design": "All methods receive identical exogenous arrivals, communication, capacity and disturbances for a scenario/seed; backlog is endogenous to each method.",
            "empty_parmetis_parts": "A ParMETIS target part may contain zero tasks. The corresponding physical resource pool remains present with its original positive capacity share, so unused capacity is retained rather than deleted.",
            "checkpointing": "A complete.json marker is written only after steps/summary/event files for one scenario-seed-method run are safely persisted. Re-running resumes only checkpoints whose context and trace hashes match.",
            "change_control": "C1.1 changes execution robustness and ParMETIS result admissibility only; frozen parameters and confirmatory seeds remain C1-identical.",
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpointed C1.1 confirmatory runner")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    _preflight_dependencies(cfg)
    for scenario in cfg.scenarios:
        if scenario not in SCENARIO_SPECS:
            raise KeyError(f"unknown scenario {scenario!r}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_config_hash(cfg)
    source_hashes = _source_hashes()
    context_material = json.dumps({"protocol": PROTOCOL_ID, "config_hash": config_hash, "source_hashes": source_hashes}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    context_hash = hashlib.sha256(context_material).hexdigest()
    context = {
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config_hash,
        "source_sha256": source_hashes,
        "context_sha256": context_hash,
    }
    context_path = out / "checkpoint_context.json"
    if context_path.exists():
        old = json.loads(context_path.read_text(encoding="utf-8"))
        if old != context:
            raise RuntimeError("existing checkpoint context does not match current frozen C1.1 code/config; use a different output directory")
    else:
        atomic_write_text(context_path, json.dumps(context, indent=2, sort_keys=True))

    total = 0
    for scenario in cfg.scenarios:
        total += len(cfg.seeds) * len(cfg.scenario_methods.get(scenario, cfg.methods))
    completed = 0
    trace_manifest: List[Dict[str, object]] = []

    for scenario in cfg.scenarios:
        methods = cfg.scenario_methods.get(scenario, cfg.methods)
        for seed in cfg.seeds:
            override = cfg.scenario_overrides.get(scenario, {})
            trace = generate_trace(
                scenario,
                seed,
                int(override.get("steps", cfg.steps)),
                int(override.get("n_agents", cfg.n_agents)),
                int(override.get("n_units", cfg.n_units)),
                supervisor_fanout=cfg.algorithm.supervisor_fanout,
            )
            trace_hash = _trace_sha256(trace)
            trace_manifest.append({
                "scenario": scenario,
                "seed": seed,
                "description": trace.metadata.get("description", ""),
                "generator_status": trace.metadata.get("generator_status", "unknown"),
                "n_agents": trace.n_agents,
                "initial_n_units": trace.n_units,
                "steps": len(trace.snapshots),
                "trace_sha256": trace_hash,
            })
            for method in methods:
                completed += 1
                cp = checkpoint_dir(out, scenario, seed, method)
                if checkpoint_complete(cp, context_hash, trace_hash):
                    print(f"[{completed}/{total}] resume-skip {scenario} seed={seed} method={method}", flush=True)
                    continue
                print(f"[{completed}/{total}] RUN {scenario} seed={seed} method={method}", flush=True)
                try:
                    steps, events, assignments, summary = run_one(
                        trace,
                        method,
                        cfg.algorithm,
                        cfg.objective,
                        save_assignments=False,
                    )
                    write_checkpoint(cp, steps, events, summary, protocol_id=PROTOCOL_ID, context_hash=context_hash, trace_hash=trace_hash)
                except Exception as exc:
                    cp.mkdir(parents=True, exist_ok=True)
                    failure = {
                        "protocol_id": PROTOCOL_ID,
                        "context_hash": context_hash,
                        "trace_sha256": trace_hash,
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                    atomic_write_text(cp / "failure.json", json.dumps(failure, indent=2, sort_keys=True))
                    raise

    print("All per-run checkpoints complete; finalizing aggregate outputs...", flush=True)
    paths = finalize(cfg, out, context, trace_manifest)
    print("C1.1 confirmatory run complete", flush=True)
    for key, path in paths.items():
        print(f"  {key:24s} {path}", flush=True)


if __name__ == "__main__":
    main()
