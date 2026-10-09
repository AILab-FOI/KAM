from __future__ import annotations

import hashlib
import json
import math
import platform
import shlex
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t

from .config import BenchmarkConfig
from .scenarios import SCENARIO_SPECS, generate_trace
from .simulation import run_one
from .statistics import ckam_dkam_comparison, friedman_omnibus, paired_method_comparisons


PROTOCOL_VERSION = "1.1-rc2"

KEY_METRICS = [
    "completion_ratio",
    "disturbance_backlog_auc_mean",
    "disturbance_service_loss_auc_mean",
    "mean_service_ratio",
    "mean_backlog_to_capacity",
    "final_backlog",
    "mean_raw_load_imbalance",
    "mean_kam_capacity_mismatch",
    "mean_overload_penalty",
    "mean_makespan_proxy",
    "mean_cross_communication_ratio",
    "mean_coordination_cost_normalized",
    "mean_management_overhead_normalized",
    "mean_objective",
    "total_reorganization_cost",
    "total_moved_agents",
    "total_control_messages",
    "total_control_message_hops",
    "total_candidate_evaluations",
    "mean_decision_time_ms",
]


def _aggregate_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (scenario, method), group in run_summary.groupby(["scenario", "method"], sort=False):
        row: Dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "runs": int(len(group)),
        }
        for metric in KEY_METRICS:
            if metric not in group:
                continue
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if len(values) > 1:
                crit = float(student_t.ppf(0.975, df=len(values) - 1))
                ci = crit * sd / math.sqrt(len(values))
            else:
                ci = 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_t"] = float(ci)
        rows.append(row)
    return pd.DataFrame(rows)


def _relative_to_static(run_summary: pd.DataFrame) -> pd.DataFrame:
    lower_better = {
        "mean_backlog_to_capacity",
        "mean_raw_load_imbalance",
        "mean_kam_capacity_mismatch",
        "mean_overload_penalty",
        "mean_makespan_proxy",
        "mean_cross_communication_ratio",
        "mean_coordination_cost_normalized",
    }
    higher_better = {"completion_ratio", "mean_service_ratio"}
    rows: List[Dict[str, object]] = []
    for scenario, sg in run_summary.groupby("scenario"):
        static = sg[sg["method"] == "static"]
        if static.empty:
            continue
        static_by_seed = static.set_index("seed")
        for method, mg in sg.groupby("method"):
            if method == "static":
                continue
            common = sorted(set(mg["seed"]) & set(static_by_seed.index))
            if not common:
                continue
            mg_idx = mg.set_index("seed")
            for metric in sorted(lower_better | higher_better):
                if metric not in mg_idx or metric not in static_by_seed:
                    continue
                deltas, rel = [], []
                for seed in common:
                    base = float(static_by_seed.loc[seed, metric])
                    value = float(mg_idx.loc[seed, metric])
                    improvement = (base - value) if metric in lower_better else (value - base)
                    deltas.append(improvement)
                    rel.append(improvement / max(abs(base), 1e-12))
                rows.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "metric": metric,
                        "paired_runs": len(common),
                        "mean_absolute_improvement_vs_static": float(np.mean(deltas)),
                        "mean_relative_improvement_vs_static": float(np.mean(rel)),
                    }
                )
    return pd.DataFrame(rows)


def _trace_sha256(trace) -> str:
    """Hash every trace/input field that can influence an algorithm or metric."""
    h = hashlib.sha256()
    h.update(trace.name.encode("utf-8"))
    h.update(str(trace.seed).encode("ascii"))
    for arr in [
        trace.initial_assignment,
        trace.q_impact,
        trace.c_impact,
        trace.ks,
        trace.migration_cost,
    ]:
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps(trace.initial_capacity_shares, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(trace.initial_unit_supervisor, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(trace.initial_supervisor_parent, sort_keys=True).encode("utf-8"))
    h.update(np.asarray([trace.root_supervisor, trace.remote_comm_cost, trace.internal_comm_base, trace.management_scale], dtype=np.float64).tobytes())
    h.update(json.dumps(trace.metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    for snap in trace.snapshots:
        h.update(np.ascontiguousarray(snap.work).tobytes())
        h.update(np.ascontiguousarray(snap.communication).tobytes())
        h.update(np.asarray([snap.total_capacity, snap.phase], dtype=np.float64).tobytes())
        h.update(str(snap.disturbance).encode("utf-8"))
    return h.hexdigest()


def _source_hashes() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    out = {}
    for p in sorted(root.glob("*.py")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _optional_backend_info() -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        import networkx as nx
        info["networkx"] = nx.__version__
    except Exception as exc:
        info["networkx"] = f"unavailable: {type(exc).__name__}"
    try:
        import igraph as ig
        info["igraph"] = getattr(ig, "__version__", "unknown")
    except Exception as exc:
        info["igraph"] = f"unavailable: {type(exc).__name__}"
    return info


def _all_requested_methods(config: BenchmarkConfig) -> set[str]:
    methods: set[str] = set()
    for scenario in config.scenarios:
        methods.update(config.scenario_methods.get(scenario, config.methods))
    return methods


def _preflight_dependencies(config: BenchmarkConfig) -> None:
    methods = _all_requested_methods(config)
    if "dynamic_leiden" in methods:
        backend = str(config.algorithm.leiden_backend).lower().strip()
        if backend == "igraph":
            try:
                import igraph  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "dynamic_leiden is requested with leiden_backend='igraph', but python-igraph is unavailable. "
                    "Install the optional Leiden dependency before a confirmatory run."
                ) from exc
        elif backend not in {"auto", "networkx"}:
            raise ValueError(f"unknown leiden_backend: {config.algorithm.leiden_backend!r}")
    if "parmetis_adaptive" in methods:
        template = config.algorithm.hpc_external_command.strip()
        if not template:
            raise RuntimeError(
                "parmetis_adaptive is requested, but hpc_external_command is empty. "
                "Build/configure the external ParMETIS driver first."
            )
        first = shlex.split(template)[0]
        # If the first token contains a path, check it directly; otherwise use PATH.
        if "{" not in first:
            exists = Path(first).exists() if ("/" in first or "\\" in first) else shutil.which(first) is not None
            if not exists:
                raise RuntimeError(f"external HPC launcher/executable not found: {first!r}")


def run_benchmark(config: BenchmarkConfig, out_dir: Path) -> Dict[str, Path]:
    _preflight_dependencies(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    all_steps: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []
    all_assignments: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, object]] = []
    trace_manifest: List[Dict[str, object]] = []

    for scenario in config.scenarios:
        if scenario not in SCENARIO_SPECS:
            raise KeyError(f"unknown scenario {scenario!r}")
        for seed in config.seeds:
            override = config.scenario_overrides.get(scenario, {})
            trace = generate_trace(
                scenario,
                seed,
                int(override.get("steps", config.steps)),
                int(override.get("n_agents", config.n_agents)),
                int(override.get("n_units", config.n_units)),
                supervisor_fanout=config.algorithm.supervisor_fanout,
            )
            trace_manifest.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "description": trace.metadata.get("description", ""),
                    "generator_status": trace.metadata.get("generator_status", "unknown"),
                    "n_agents": trace.n_agents,
                    "initial_n_units": trace.n_units,
                    "steps": len(trace.snapshots),
                    "trace_sha256": _trace_sha256(trace),
                }
            )
            methods = config.scenario_methods.get(scenario, config.methods)
            for method in methods:
                steps, events, assignments, summary = run_one(
                    trace,
                    method,
                    config.algorithm,
                    config.objective,
                    save_assignments=config.save_assignments,
                )
                all_steps.append(steps)
                if not events.empty:
                    all_events.append(events)
                if config.save_assignments and not assignments.empty:
                    all_assignments.append(assignments)
                run_summaries.append(
                    {"scenario": scenario, "seed": seed, "method": method, **summary}
                )

    steps_df = pd.concat(all_steps, ignore_index=True) if all_steps else pd.DataFrame()
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    assignments_df = (
        pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()
    )
    run_summary_df = pd.DataFrame(run_summaries)
    aggregate_df = _aggregate_runs(run_summary_df)
    relative_df = _relative_to_static(run_summary_df)
    stats_df = paired_method_comparisons(run_summary_df)
    omnibus_df = friedman_omnibus(run_summary_df)
    c_vs_d_df = ckam_dkam_comparison(run_summary_df)

    paths = {
        "steps": raw_dir / "step_metrics.csv",
        "events": raw_dir / "events.csv",
        "assignments": raw_dir / "assignments.csv",
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
    if config.save_assignments:
        assignments_df.to_csv(paths["assignments"], index=False)
    run_summary_df.to_csv(paths["run_summary"], index=False)
    aggregate_df.to_csv(paths["aggregate_summary"], index=False)
    relative_df.to_csv(paths["relative_to_static"], index=False)
    stats_df.to_csv(paths["statistical_comparisons"], index=False)
    omnibus_df.to_csv(paths["omnibus_statistics"], index=False)
    c_vs_d_df.to_csv(paths["ckam_vs_dkam"], index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "freeze_status": "core algorithm/objective RC; built-in scenario generators are PILOT ONLY",
        "config": config.to_dict(),
        "traces": trace_manifest,
        "source_sha256": _source_hashes(),
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
            "kam_abstraction": "Each simulated agent carries one aggregate activity bundle. F=1+number of external organizational units contacted; exertion=work*F*kq*kc*ks.",
            "objective": "KAM capacity-share mismatch + capacity-weighted overload + normalized network coordination cost + explicit structural management overhead + reorganization cost.",
            "d_kam": "Holonic local candidate generation, multi-level lowest-sufficient authority, delayed observations, local-neighbourhood KAM delta evaluation, and explicit control-message accounting.",
            "statistics": "Independent seeds are the inferential units; time steps are not treated as independent replicates. Pairwise tests use Holm correction and Friedman/Kendall-W omnibus results are also produced.",
            "publication_warning": "Do not treat results as confirmatory until the C1 configuration, final Leiden backend, external HPC comparator parameters, code hashes, and confirmatory seeds are frozen.",
            "dynamic_leiden": "Confirmatory Dynamic Leiden requires an established backend (python-igraph by default); no Louvain/spectral fallback is silently substituted.",
            "external_hpc": "The ParMETIS baseline is invoked out-of-process through the KAM_BENCH_PARTITION_V1 protocol; the simulator does not reimplement or modify ParMETIS decisions.",
        },
    }
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return paths
