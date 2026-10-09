from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from kam_bench.cli import _load_config
from kam_bench.experiments import _optional_backend_info, _source_hashes, _trace_sha256
from kam_bench.scenarios import SCENARIO_SPECS, generate_trace
from kam_bench.simulation import run_one

PROTOCOL_ID = "D1-S1-A1-followup-v1"
SCENARIOS = ["warehouse", "edge", "enterprise", "hpc"]

# Frozen follow-up design. These values were specified before inspecting any
# D1/S1/A1 results. C-KAM/D-KAM scientific parameters come unchanged from C1.1.
DESIGN = {
    "D1": {
        "purpose": "D-KAM robustness to stale information",
        "method": "d_kam",
        "new_delays": [0, 2, 4, 8],
        "reference_delay_from_C1_1": 1,
        "seeds": list(range(3001, 3031)),
        "steps": 200,
        "n_agents": 64,
        "n_units": 8,
        "parameter_rule": "Only dkam_observation_delay varies; all other C1.1 D-KAM parameters, including dkam_error_margin, remain frozen.",
    },
    "S1": {
        "purpose": "C-KAM vs D-KAM computational/coordination scalability",
        "methods": ["c_kam", "d_kam"],
        "n_agents": [32, 64, 128, 256, 512, 1024],
        "agents_per_initial_unit": 8,
        "seeds": list(range(5001, 5011)),
        "steps": 200,
        "parameter_rule": "All C1.1 algorithm parameters remain frozen; only system size and corresponding initial unit count vary.",
    },
    "A1": {
        "purpose": "C-KAM operator ablation",
        "method": "c_kam",
        "new_variants": {
            "move_only": {"allow_move": True, "allow_merge": False, "allow_split": False},
            "move_merge": {"allow_move": True, "allow_merge": True, "allow_split": False},
            "move_split": {"allow_move": True, "allow_merge": False, "allow_split": True},
        },
        "reference_variant_from_C1_1": "full_move_merge_split",
        "seeds": list(range(3001, 3031)),
        "steps": 200,
        "n_agents": 64,
        "n_units": 8,
        "parameter_rule": "Only operator availability varies; objective, thresholds, costs, adaptation cadence, and candidate generation remain C1.1-frozen.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def human_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.2f}h"
    return f"{hours/24.0:.2f}d"


def dense_comm_gib(n_agents: int, steps: int) -> float:
    # One dense float64 N x N matrix per snapshot; useful as a rough lower-bound
    # warning for the existing benchmark representation.
    return float(n_agents) * float(n_agents) * 8.0 * float(steps) / (1024.0**3)


def condition_label(item: Dict[str, object]) -> str:
    if item["block"] == "D1":
        return f"delay={item['delay']}"
    if item["block"] == "S1":
        return f"N={item['n_agents']}, U0={item['n_units']}"
    return str(item["variant"])


def item_id(item: Dict[str, object]) -> str:
    if item["block"] == "D1":
        cond = f"delay_{item['delay']}"
    elif item["block"] == "S1":
        cond = f"N_{item['n_agents']}"
    else:
        cond = str(item["variant"])
    return f"{item['block']}|{item['scenario']}|{cond}|seed_{item['seed']}|{item['method']}"


def checkpoint_dir(root: Path, item: Dict[str, object]) -> Path:
    if item["block"] == "D1":
        cond = f"delay_{item['delay']}"
    elif item["block"] == "S1":
        cond = f"N_{item['n_agents']}"
    else:
        cond = str(item["variant"])
    return root / "checkpoints" / str(item["block"]) / str(item["scenario"]) / cond / str(item["seed"]) / str(item["method"])


def algorithm_overrides(item: Dict[str, object]) -> Dict[str, object]:
    if item["block"] == "D1":
        return {"dkam_observation_delay": int(item["delay"])}
    if item["block"] == "A1":
        return dict(DESIGN["A1"]["new_variants"][str(item["variant"])])
    return {}


def algorithm_hash(base_algorithm, item: Dict[str, object]) -> str:
    cfg = replace(base_algorithm, **algorithm_overrides(item))
    return canonical_hash(asdict(cfg))


def trace_key(item: Dict[str, object]) -> Tuple[object, ...]:
    return (
        item["scenario"],
        int(item["seed"]),
        int(item["steps"]),
        int(item["n_agents"]),
        int(item["n_units"]),
    )


def build_plan(smoke: bool = False, blocks: Optional[Iterable[str]] = None) -> List[Dict[str, object]]:
    selected = set(blocks or ["D1", "S1", "A1"])
    plan: List[Dict[str, object]] = []
    if smoke:
        scenarios = ["warehouse"]
        if "D1" in selected:
            for scenario in scenarios:
                for seed in [9901]:
                    for delay in [0, 2]:
                        plan.append({"block": "D1", "scenario": scenario, "seed": seed, "method": "d_kam", "delay": delay, "steps": 30, "n_agents": 16, "n_units": 2})
        if "S1" in selected:
            for scenario in scenarios:
                for n in [16, 32]:
                    for seed in [9902]:
                        for method in ["c_kam", "d_kam"]:
                            plan.append({"block": "S1", "scenario": scenario, "seed": seed, "method": method, "steps": 30, "n_agents": n, "n_units": max(2, n // 8)})
        if "A1" in selected:
            for scenario in scenarios:
                for seed in [9903]:
                    for variant in DESIGN["A1"]["new_variants"]:
                        plan.append({"block": "A1", "scenario": scenario, "seed": seed, "method": "c_kam", "variant": variant, "steps": 30, "n_agents": 16, "n_units": 2})
        return plan

    if "D1" in selected:
        d = DESIGN["D1"]
        # Order by scenario/seed/delay so all four delays reuse exactly the same in-memory trace.
        for scenario in SCENARIOS:
            for seed in d["seeds"]:
                for delay in d["new_delays"]:
                    plan.append({"block": "D1", "scenario": scenario, "seed": seed, "method": "d_kam", "delay": delay, "steps": d["steps"], "n_agents": d["n_agents"], "n_units": d["n_units"]})

    if "S1" in selected:
        d = DESIGN["S1"]
        # Ascending N intentionally secures all smaller-scale checkpoints before the expensive end.
        for scenario in SCENARIOS:
            for n in d["n_agents"]:
                n_units = max(2, int(n) // int(d["agents_per_initial_unit"]))
                for seed in d["seeds"]:
                    for method in d["methods"]:
                        plan.append({"block": "S1", "scenario": scenario, "seed": seed, "method": method, "steps": d["steps"], "n_agents": int(n), "n_units": int(n_units)})

    if "A1" in selected:
        d = DESIGN["A1"]
        # Order by scenario/seed/variant so the three ablations reuse one trace.
        for scenario in SCENARIOS:
            for seed in d["seeds"]:
                for variant in d["new_variants"]:
                    plan.append({"block": "A1", "scenario": scenario, "seed": seed, "method": "c_kam", "variant": variant, "steps": d["steps"], "n_agents": d["n_agents"], "n_units": d["n_units"]})
    return plan


def checkpoint_complete(cp: Path, *, context_hash: str, item_hash: str, trace_hash: str, algo_hash: str) -> bool:
    marker = cp / "complete.json"
    if not marker.exists() or not (cp / "steps.csv").exists() or not (cp / "summary.json").exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("protocol_id") == PROTOCOL_ID
        and data.get("context_hash") == context_hash
        and data.get("item_hash") == item_hash
        and data.get("trace_sha256") == trace_hash
        and data.get("algorithm_sha256") == algo_hash
        and data.get("status") == "complete"
    )


def checkpoint_prelim_complete(cp: Path, *, context_hash: str, item_hash: str, algo_hash: str) -> Optional[Dict[str, object]]:
    """Fast resume check that does not regenerate an already-completed trace.

    A full trace hash check is still performed whenever any item in the trace group
    must be rerun. If every item sharing a trace is complete, the matching stored
    trace hash is sufficient because the frozen generator/source/context have not
    changed.
    """
    marker = cp / "complete.json"
    if not marker.exists() or not (cp / "steps.csv").exists() or not (cp / "summary.json").exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    ok = (
        data.get("protocol_id") == PROTOCOL_ID
        and data.get("context_hash") == context_hash
        and data.get("item_hash") == item_hash
        and data.get("algorithm_sha256") == algo_hash
        and data.get("status") == "complete"
        and bool(data.get("trace_sha256"))
    )
    return data if ok else None


def annotate_frame(df: pd.DataFrame, item: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    meta = {
        "experiment_block": item["block"],
        "condition": condition_label(item),
        "analysis_method": (
            f"d_kam_delay_{item['delay']}" if item["block"] == "D1"
            else f"{item['method']}_N_{item['n_agents']}" if item["block"] == "S1"
            else f"c_kam_{item['variant']}"
        ),
        "experimental_n_agents": int(item["n_agents"]),
        "experimental_initial_n_units": int(item["n_units"]),
    }
    if item["block"] == "D1":
        meta["dkam_observation_delay"] = int(item["delay"])
    elif item["block"] == "S1":
        meta["scale_n_agents"] = int(item["n_agents"])
    elif item["block"] == "A1":
        meta["ablation_variant"] = str(item["variant"])
    for key, value in reversed(list(meta.items())):
        out.insert(0, key, value)
    return out


def diagnostics_from_frames(steps: pd.DataFrame, events: pd.DataFrame) -> Dict[str, object]:
    diag: Dict[str, object] = {}
    if not steps.empty:
        diag["total_proposals"] = float(steps.get("step_proposals", pd.Series(dtype=float)).sum())
        diag["total_rejected_proposals"] = float(steps.get("step_rejected_proposals", pd.Series(dtype=float)).sum())
        diag["accepted_operation_count"] = float(steps.get("step_operations", pd.Series(dtype=float)).sum())
        props = float(diag["total_proposals"])
        diag["proposal_acceptance_fraction"] = float(diag["accepted_operation_count"] / props) if props > 0 else np.nan
        adapt = steps[steps.get("step_candidate_evaluations", 0) > 0] if "step_candidate_evaluations" in steps.columns else steps.iloc[0:0]
        diag["mean_observation_lag_on_search_steps"] = float(adapt["observation_lag"].mean()) if not adapt.empty and "observation_lag" in adapt else np.nan
    else:
        diag.update({"total_proposals": 0.0, "total_rejected_proposals": 0.0, "accepted_operation_count": 0.0, "proposal_acceptance_fraction": np.nan, "mean_observation_lag_on_search_steps": np.nan})

    if events.empty or "actual_delta_current_state" not in events.columns:
        diag.update({
            "accepted_event_count": 0.0,
            "actual_worsening_operation_count": 0.0,
            "actual_worsening_fraction": np.nan,
            "mean_delta_estimation_error": np.nan,
            "mean_abs_delta_estimation_error": np.nan,
            "mean_positive_underestimate_error": np.nan,
            "p95_positive_underestimate_error": np.nan,
        })
        return diag

    actual = pd.to_numeric(events["actual_delta_current_state"], errors="coerce")
    estimated = pd.to_numeric(events.get("estimated_delta", np.nan), errors="coerce")
    valid_actual = actual.notna()
    accepted_n = int(valid_actual.sum())
    worsening = int((actual[valid_actual] > 0).sum())
    valid_pair = actual.notna() & estimated.notna()
    error = actual[valid_pair] - estimated[valid_pair]
    positive = np.maximum(error.to_numpy(float), 0.0) if len(error) else np.array([], dtype=float)
    diag.update({
        "accepted_event_count": float(accepted_n),
        "actual_worsening_operation_count": float(worsening),
        "actual_worsening_fraction": float(worsening / accepted_n) if accepted_n else np.nan,
        "mean_delta_estimation_error": float(error.mean()) if len(error) else np.nan,
        "mean_abs_delta_estimation_error": float(np.abs(error).mean()) if len(error) else np.nan,
        "mean_positive_underestimate_error": float(positive.mean()) if len(positive) else np.nan,
        "p95_positive_underestimate_error": float(np.quantile(positive, 0.95)) if len(positive) else np.nan,
    })
    return diag


def write_checkpoint(cp: Path, *, steps: pd.DataFrame, events: pd.DataFrame, summary: Dict[str, object], marker: Dict[str, object]) -> None:
    cp.mkdir(parents=True, exist_ok=True)
    steps_tmp = cp / "steps.csv.tmp"
    steps.to_csv(steps_tmp, index=False)
    os.replace(steps_tmp, cp / "steps.csv")
    events_path = cp / "events.csv"
    if events.empty:
        if events_path.exists():
            events_path.unlink()
    else:
        events_tmp = cp / "events.csv.tmp"
        events.to_csv(events_tmp, index=False)
        os.replace(events_tmp, events_path)
    atomic_write_text(cp / "summary.json", json.dumps(summary, indent=2, sort_keys=True, default=json_default))
    atomic_write_text(cp / "complete.json", json.dumps(marker, indent=2, sort_keys=True, default=json_default))
    failure = cp / "failure.json"
    if failure.exists():
        failure.unlink()


def load_checkpoint(cp: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    steps = pd.read_csv(cp / "steps.csv")
    events_path = cp / "events.csv"
    events = pd.read_csv(events_path, low_memory=False) if events_path.exists() and events_path.stat().st_size else pd.DataFrame()
    summary = json.loads((cp / "summary.json").read_text(encoding="utf-8"))
    return steps, events, summary


def heartbeat(stop: threading.Event, label: str, start: float, interval: float = 60.0) -> None:
    while not stop.wait(interval):
        print(f"    ... still running {label} | elapsed {human_duration(time.monotonic()-start)}", flush=True)


def append_progress(progress_path: Path, row: Dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    header = not progress_path.exists()
    pd.DataFrame([row]).to_csv(progress_path, mode="a", header=header, index=False)


def read_existing_elapsed(plan: List[Dict[str, object]], out: Path, context_hash: str) -> List[float]:
    vals: List[float] = []
    for item in plan:
        marker = checkpoint_dir(out, item) / "complete.json"
        if not marker.exists():
            continue
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("context_hash") == context_hash and data.get("status") == "complete":
            v = data.get("wall_elapsed_seconds")
            if isinstance(v, (int, float)) and v > 0:
                vals.append(float(v))
    return vals


def aggregate_long(df: pd.DataFrame, group_cols: List[str], metrics: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if metrics is None:
        ignore = set(group_cols) | {"seed", "steps"}
        metrics = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            if not len(vals):
                continue
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se = sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            tcrit = float(stats.t.ppf(0.975, len(vals)-1)) if len(vals) > 1 else 0.0
            rows.append({**base, "metric": metric, "n": len(vals), "mean": mean, "sd": sd, "ci95_low": mean-tcrit*se, "ci95_high": mean+tcrit*se, "median": float(np.median(vals))})
    return pd.DataFrame(rows)


def add_c1_references(out: Path, c1_results: Optional[Path]) -> Dict[str, object]:
    info: Dict[str, object] = {"available": False}
    if c1_results is None:
        return info
    c1_results = Path(c1_results)
    rs_path = c1_results / "run_summary.csv"
    ev_path = c1_results / "raw" / "events.csv"
    st_path = c1_results / "raw" / "step_metrics.csv"
    if not rs_path.exists():
        info["reason"] = f"missing {rs_path}"
        return info
    rs = pd.read_csv(rs_path)
    seeds = set(range(3001, 3031))
    c1_subset = rs[rs["seed"].isin(seeds) & rs["scenario"].isin(SCENARIOS)].copy()

    refdir = out / "references_from_C1_1"
    refdir.mkdir(parents=True, exist_ok=True)
    d1_ref = c1_subset[c1_subset["method"] == "d_kam"].copy()
    d1_ref["experiment_block"] = "D1"
    d1_ref["condition"] = "delay=1"
    d1_ref["dkam_observation_delay"] = 1
    d1_ref["analysis_method"] = "d_kam_delay_1"
    d1_ref["reference_source"] = "C1.1"
    d1_ref.to_csv(refdir / "D1_delay1_run_summary_reference.csv", index=False)

    a1_ref = c1_subset[c1_subset["method"] == "c_kam"].copy()
    a1_ref["experiment_block"] = "A1"
    a1_ref["condition"] = "full_move_merge_split"
    a1_ref["ablation_variant"] = "full_move_merge_split"
    a1_ref["analysis_method"] = "c_kam_full_move_merge_split"
    a1_ref["reference_source"] = "C1.1"
    a1_ref.to_csv(refdir / "A1_full_ckam_run_summary_reference.csv", index=False)

    # Delay=1 diagnostics from the already-completed C1.1 raw files, when present.
    if ev_path.exists():
        ev = pd.read_csv(ev_path, low_memory=False)
        ev = ev[(ev["method"] == "d_kam") & ev["seed"].isin(seeds) & ev["scenario"].isin(SCENARIOS)]
    else:
        ev = pd.DataFrame()
    if st_path.exists():
        # The file is modest enough relative to C1.1 and reading it once saves 120 reruns.
        st = pd.read_csv(st_path, low_memory=False)
        st = st[(st["method"] == "d_kam") & st["seed"].isin(seeds) & st["scenario"].isin(SCENARIOS)]
    else:
        st = pd.DataFrame()
    diag_rows = []
    if not st.empty:
        for (scenario, seed), sg in st.groupby(["scenario", "seed"]):
            eg = ev[(ev["scenario"] == scenario) & (ev["seed"] == seed)] if not ev.empty else pd.DataFrame()
            diag_rows.append({"scenario": scenario, "seed": int(seed), "dkam_observation_delay": 1, "reference_source": "C1.1", **diagnostics_from_frames(sg, eg)})
    pd.DataFrame(diag_rows).to_csv(refdir / "D1_delay1_diagnostics_reference.csv", index=False)

    info.update({
        "available": True,
        "path": str(c1_results.resolve()),
        "run_summary_sha256": sha256_file(rs_path),
        "events_sha256": sha256_file(ev_path) if ev_path.exists() else None,
        "step_metrics_sha256": sha256_file(st_path) if st_path.exists() else None,
        "d1_reference_rows": int(len(d1_ref)),
        "a1_reference_rows": int(len(a1_ref)),
    })
    return info


def finalize(plan: List[Dict[str, object]], out: Path, *, context: Dict[str, object], trace_manifest: Dict[str, Dict[str, object]], c1_results: Optional[Path]) -> None:
    print("All requested checkpoints are complete. Building consolidated outputs...", flush=True)
    all_summaries: List[Dict[str, object]] = []
    block_steps: Dict[str, List[pd.DataFrame]] = {"D1": [], "S1": [], "A1": []}
    block_events: Dict[str, List[pd.DataFrame]] = {"D1": [], "S1": [], "A1": []}

    for item in plan:
        cp = checkpoint_dir(out, item)
        if not (cp / "complete.json").exists():
            raise RuntimeError(f"missing completed checkpoint: {cp}")
        steps, events, summary = load_checkpoint(cp)
        block_steps[str(item["block"])].append(steps)
        if not events.empty:
            block_events[str(item["block"])].append(events)
        all_summaries.append(summary)

    all_summary_df = pd.DataFrame(all_summaries)
    all_summary_df.to_csv(out / "followup_run_summary.csv", index=False)

    for block in ["D1", "S1", "A1"]:
        bdir = out / block
        rawdir = bdir / "raw"
        rawdir.mkdir(parents=True, exist_ok=True)
        bsum = all_summary_df[all_summary_df["experiment_block"] == block].copy()
        bsteps = pd.concat(block_steps[block], ignore_index=True) if block_steps[block] else pd.DataFrame()
        bevents = pd.concat(block_events[block], ignore_index=True) if block_events[block] else pd.DataFrame()
        bsum.to_csv(bdir / "run_summary_new.csv", index=False)
        bsteps.to_csv(rawdir / "step_metrics_new.csv", index=False)
        bevents.to_csv(rawdir / "events_new.csv", index=False)
        if block == "D1":
            group_cols = ["scenario", "dkam_observation_delay"]
        elif block == "S1":
            group_cols = ["scenario", "experimental_n_agents", "method"]
        else:
            group_cols = ["scenario", "ablation_variant"]
        aggregate_long(bsum, group_cols).to_csv(bdir / "aggregate_summary_long_new.csv", index=False)

    # D1 diagnostics and optional C1.1 delay=1 reference.
    d1 = all_summary_df[all_summary_df["experiment_block"] == "D1"].copy()
    d1_diag_cols = [
        "scenario", "seed", "dkam_observation_delay", "accepted_event_count",
        "actual_worsening_operation_count", "actual_worsening_fraction",
        "mean_delta_estimation_error", "mean_abs_delta_estimation_error",
        "mean_positive_underestimate_error", "p95_positive_underestimate_error",
        "total_proposals", "total_rejected_proposals", "accepted_operation_count",
        "proposal_acceptance_fraction", "mean_observation_lag_on_search_steps",
    ]
    d1[[c for c in d1_diag_cols if c in d1.columns]].to_csv(out / "D1" / "delay_diagnostics_new.csv", index=False)

    ref_info = add_c1_references(out, c1_results)
    if ref_info.get("available"):
        refdir = out / "references_from_C1_1"
        d1_ref = pd.read_csv(refdir / "D1_delay1_run_summary_reference.csv")
        # Normalize columns before concat; production new rows already contain these identifiers.
        d1_combined = pd.concat([d1, d1_ref], ignore_index=True, sort=False)
        d1_combined.to_csv(out / "D1" / "run_summary_with_C1_1_delay1.csv", index=False)
        aggregate_long(d1_combined, ["scenario", "dkam_observation_delay"]).to_csv(out / "D1" / "aggregate_with_C1_1_delay1_long.csv", index=False)
        diag_ref_path = refdir / "D1_delay1_diagnostics_reference.csv"
        if diag_ref_path.exists():
            dref = pd.read_csv(diag_ref_path)
            dnew = d1[[c for c in d1_diag_cols if c in d1.columns]].copy()
            dcomb = pd.concat([dnew, dref], ignore_index=True, sort=False)
            dcomb.to_csv(out / "D1" / "delay_diagnostics_with_C1_1_delay1.csv", index=False)
            aggregate_long(dcomb, ["scenario", "dkam_observation_delay"], metrics=[c for c in d1_diag_cols if c not in {"scenario", "seed", "dkam_observation_delay"} and c in dcomb.columns]).to_csv(out / "D1" / "delay_diagnostics_aggregate_long.csv", index=False)

        a1 = all_summary_df[all_summary_df["experiment_block"] == "A1"].copy()
        a1_ref = pd.read_csv(refdir / "A1_full_ckam_run_summary_reference.csv")
        a1_combined = pd.concat([a1, a1_ref], ignore_index=True, sort=False)
        a1_combined.to_csv(out / "A1" / "run_summary_with_C1_1_full.csv", index=False)
        aggregate_long(a1_combined, ["scenario", "ablation_variant"]).to_csv(out / "A1" / "aggregate_with_C1_1_full_long.csv", index=False)

    # S1 scaling exponents and D-KAM/C-KAM ratios.
    s1 = all_summary_df[all_summary_df["experiment_block"] == "S1"].copy()
    s1_means = s1.groupby(["scenario", "experimental_n_agents", "method"], as_index=False).mean(numeric_only=True)
    s1_means.to_csv(out / "S1" / "mean_by_size_method.csv", index=False)
    slope_metrics = ["total_decision_time_ms", "total_candidate_evaluations", "total_control_message_hops", "total_reorganization_cost", "disturbance_backlog_auc_mean"]
    slope_rows = []
    for scenario in SCENARIOS:
        for method in ["c_kam", "d_kam"]:
            g = s1_means[(s1_means["scenario"] == scenario) & (s1_means["method"] == method)].sort_values("experimental_n_agents")
            for metric in slope_metrics:
                x = pd.to_numeric(g["experimental_n_agents"], errors="coerce").to_numpy(float)
                y = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if mask.sum() >= 3:
                    lr = stats.linregress(np.log2(x[mask]), np.log2(y[mask]))
                    slope_rows.append({"scenario": scenario, "method": method, "metric": metric, "scaling_exponent": float(lr.slope), "r_squared": float(lr.rvalue**2), "p_value": float(lr.pvalue), "n_sizes": int(mask.sum())})
    pd.DataFrame(slope_rows).to_csv(out / "S1" / "scaling_exponents.csv", index=False)

    ratio_rows = []
    ratio_metrics = ["total_decision_time_ms", "total_candidate_evaluations", "total_control_message_hops", "total_reorganization_cost", "disturbance_backlog_auc_mean", "mean_raw_load_imbalance", "mean_cross_communication_ratio"]
    for scenario in SCENARIOS:
        for n in DESIGN["S1"]["n_agents"]:
            g = s1_means[(s1_means["scenario"] == scenario) & (s1_means["experimental_n_agents"] == n)].set_index("method")
            if not {"c_kam", "d_kam"}.issubset(set(g.index)):
                continue
            row = {"scenario": scenario, "n_agents": n}
            for metric in ratio_metrics:
                c = float(g.loc["c_kam", metric])
                d = float(g.loc["d_kam", metric])
                row[f"d_over_c_{metric}"] = d / c if c != 0 else np.nan
            ratio_rows.append(row)
    pd.DataFrame(ratio_rows).to_csv(out / "S1" / "dkam_vs_ckam_ratios_by_size.csv", index=False)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": utc_now(),
        "design": DESIGN,
        "context": context,
        "plan_run_count": len(plan),
        "expected_production_run_count": 1320,
        "trace_manifest": list(trace_manifest.values()),
        "c1_1_reference": ref_info,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "optional_backends": _optional_backend_info(),
        },
        "notes": {
            "checkpoint_unit": "one complete block/scenario/condition/seed/method run",
            "resume_rule": "A checkpoint is reused only if protocol, runner/context, trace, item, and effective AlgorithmConfig hashes match.",
            "paired_design": "Within each D1/A1 trace group and each S1 scenario-size-seed group, methods/conditions receive the identical exogenous trace.",
            "C1_1_reuse": "D1 delay=1 and A1 full C-KAM are not rerun; when the local C1_1_results directory is available, only their relevant reference rows/diagnostics are copied into the follow-up output.",
            "change_control": "No C1.1 objective weight, adaptation threshold, D-KAM safety margin, operator cost, or algorithm parameter is retuned. D1 varies delay; S1 varies system size; A1 varies only operator availability.",
        },
    }
    atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=json_default))
    atomic_write_text(out / "ALL_BLOCKS_COMPLETE.json", json.dumps({"protocol_id": PROTOCOL_ID, "completed_utc": utc_now(), "run_count": len(plan), "status": "complete"}, indent=2))
    print("Consolidated outputs complete.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkpointed D1 + S1 + A1 follow-up experiment runner")
    ap.add_argument("--config", type=Path, required=True, help="Frozen C1.1 JSON configuration")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c1-results", type=Path, default=None, help="Optional local C1_1_results directory for delay=1/full-C-KAM reference rows")
    ap.add_argument("--blocks", nargs="+", choices=["D1", "S1", "A1"], default=["D1", "S1", "A1"])
    ap.add_argument("--smoke", action="store_true", help="Software smoke test only; not scientific output")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--finalize-only", action="store_true")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    for scenario in SCENARIOS:
        if scenario not in SCENARIO_SPECS:
            raise KeyError(f"required scenario not available: {scenario!r}")

    plan = build_plan(smoke=args.smoke, blocks=args.blocks)
    expected = 9 if args.smoke and set(args.blocks) == {"D1", "S1", "A1"} else (1320 if set(args.blocks) == {"D1", "S1", "A1"} else len(plan))
    print(f"Protocol: {PROTOCOL_ID}{' [SMOKE]' if args.smoke else ''}", flush=True)
    print(f"Blocks: {', '.join(args.blocks)}", flush=True)
    print(f"Planned new runs: {len(plan)}" + (f" (full production plan: {expected})" if not args.smoke else ""), flush=True)
    if not args.smoke and set(args.blocks) == {"D1", "S1", "A1"} and len(plan) != 1320:
        raise RuntimeError(f"production plan count changed unexpectedly: {len(plan)} != 1320")
    if args.plan_only:
        counts = pd.DataFrame(plan).groupby("block").size()
        print(counts.to_string())
        return

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    design_for_context = {k: DESIGN[k] for k in args.blocks}
    base_config_hash = canonical_hash(cfg.to_dict())
    source_hashes = _source_hashes()
    runner_hash = sha256_file(Path(__file__).resolve())
    context = {
        "protocol_id": PROTOCOL_ID,
        "smoke": bool(args.smoke),
        "blocks": list(args.blocks),
        "base_config_sha256": base_config_hash,
        "scientific_source_sha256": source_hashes,
        "design_sha256": canonical_hash(design_for_context),
        "runner_sha256": runner_hash,
    }
    context["context_sha256"] = canonical_hash(context)
    context_hash = str(context["context_sha256"])
    context_path = out / "checkpoint_context.json"
    if context_path.exists():
        old = json.loads(context_path.read_text(encoding="utf-8"))
        if old != context:
            raise RuntimeError("Existing follow-up checkpoint context differs from current code/design. Keep that directory intact and use a new output directory rather than mixing protocols.")
    else:
        atomic_write_text(context_path, json.dumps(context, indent=2, sort_keys=True))

    if args.finalize_only:
        # Regenerate all trace hashes deterministically so finalization still verifies the intended plan.
        trace_manifest: Dict[str, Dict[str, object]] = {}
        current_key = None
        trace = None
        for item in plan:
            key = trace_key(item)
            if key != current_key:
                trace = generate_trace(str(item["scenario"]), int(item["seed"]), int(item["steps"]), int(item["n_agents"]), int(item["n_units"]), supervisor_fanout=cfg.algorithm.supervisor_fanout)
                current_key = key
            th = _trace_sha256(trace)
            trace_manifest[canonical_hash(key)] = {"scenario": item["scenario"], "seed": item["seed"], "steps": item["steps"], "n_agents": item["n_agents"], "initial_n_units": item["n_units"], "trace_sha256": th}
        finalize(plan, out, context=context, trace_manifest=trace_manifest, c1_results=args.c1_results)
        return

    # Operational progress state.
    progress_path = out / "progress_history.csv"
    existing_elapsed = read_existing_elapsed(plan, out, context_hash)
    rough_existing = 0
    for item in plan:
        marker = checkpoint_dir(out, item) / "complete.json"
        if marker.exists():
            try:
                m = json.loads(marker.read_text(encoding="utf-8"))
                if m.get("context_hash") == context_hash and m.get("status") == "complete":
                    rough_existing += 1
            except Exception:
                pass
    print(f"Existing matching completed checkpoints (pre-scan): {rough_existing}/{len(plan)}", flush=True)
    print("Resume behavior: rerun this same command at any time; completed checkpoints will be verified and skipped.", flush=True)
    if not args.smoke:
        print(f"S1 N=1024 dense communication matrices alone are roughly {dense_comm_gib(1024, DESIGN['S1']['steps']):.2f} GiB per generated trace before Python/object overhead.", flush=True)
        print("S1 runs in ascending N so all smaller-scale results are safely checkpointed before N=1024.", flush=True)

    total = len(plan)
    completed = 0
    run_durations = list(existing_elapsed)
    block_totals = {b: sum(1 for x in plan if x["block"] == b) for b in args.blocks}
    block_done = {b: 0 for b in args.blocks}
    trace_manifest: Dict[str, Dict[str, object]] = {}
    invocation_start = time.monotonic()

    # Group contiguous plan items that share one exogenous trace. This both
    # guarantees paired conditions and lets a resume skip an entirely completed
    # expensive trace group without regenerating (notably S1 N=1024).
    groups: List[List[Dict[str, object]]] = []
    for item in plan:
        if not groups or trace_key(groups[-1][0]) != trace_key(item):
            groups.append([item])
        else:
            groups[-1].append(item)

    for group in groups:
        first = group[0]
        key = trace_key(first)
        tk = canonical_hash(key)

        # Fast whole-group resume path: source/config/runner/design context, item
        # hashes and effective AlgorithmConfig hashes must match, and all stored
        # items must agree on one trace hash.
        prelim: List[Optional[Dict[str, object]]] = []
        for item in group:
            prelim.append(checkpoint_prelim_complete(
                checkpoint_dir(out, item),
                context_hash=context_hash,
                item_hash=canonical_hash(item),
                algo_hash=algorithm_hash(cfg.algorithm, item),
            ))
        if all(x is not None for x in prelim):
            stored_hashes = {str(x.get("trace_sha256")) for x in prelim if x is not None}
            if len(stored_hashes) == 1:
                th = next(iter(stored_hashes))
                meta = next((x.get("trace_manifest") for x in prelim if x and isinstance(x.get("trace_manifest"), dict)), None)
                trace_manifest[tk] = meta if isinstance(meta, dict) else {
                    "scenario": first["scenario"], "seed": int(first["seed"]), "steps": int(first["steps"]),
                    "n_agents": int(first["n_agents"]), "initial_n_units": int(first["n_units"]),
                    "trace_sha256": th, "resume_manifest_source": "checkpoint_marker",
                }
                for item in group:
                    completed += 1
                    block = str(item["block"])
                    block_done[block] += 1
                print(
                    f"[{completed}/{total}] resume-skip TRACE GROUP {first['block']} {first['scenario']} "
                    f"seed={first['seed']} N={first['n_agents']} ({len(group)} completed items; no trace regeneration)",
                    flush=True,
                )
                continue

        label_trace = f"{first['block']} {first['scenario']} seed={first['seed']} N={first['n_agents']} U0={first['n_units']}"
        print(f"\nTRACE {label_trace} | estimated dense communication payload {dense_comm_gib(int(first['n_agents']), int(first['steps'])):.3f} GiB", flush=True)
        t_trace = time.monotonic()
        trace = generate_trace(
            str(first["scenario"]), int(first["seed"]), int(first["steps"]), int(first["n_agents"]), int(first["n_units"]),
            supervisor_fanout=cfg.algorithm.supervisor_fanout,
        )
        trace_hash = _trace_sha256(trace)
        print(f"TRACE ready in {human_duration(time.monotonic()-t_trace)} | sha256={trace_hash[:12]}…", flush=True)
        trace_info = {
            "scenario": first["scenario"], "seed": int(first["seed"]), "steps": int(first["steps"]),
            "n_agents": int(first["n_agents"]), "initial_n_units": int(first["n_units"]),
            "description": trace.metadata.get("description", ""),
            "generator_status": trace.metadata.get("generator_status", "unknown"),
            "trace_sha256": trace_hash,
        }
        trace_manifest[tk] = trace_info

        for item in group:
            cp = checkpoint_dir(out, item)
            i_hash = canonical_hash(item)
            a_hash = algorithm_hash(cfg.algorithm, item)
            block = str(item["block"])
            if checkpoint_complete(cp, context_hash=context_hash, item_hash=i_hash, trace_hash=trace_hash, algo_hash=a_hash):
                completed += 1
                block_done[block] += 1
                print(f"[{completed}/{total} | {block} {block_done[block]}/{block_totals[block]}] resume-skip {item_id(item)}", flush=True)
                continue

            label = item_id(item)
            avg = float(np.mean(run_durations)) if run_durations else float("nan")
            remaining = total - completed
            eta = human_duration(avg * remaining) if np.isfinite(avg) else "unknown"
            print(f"[{completed+1}/{total} | {block} {block_done[block]+1}/{block_totals[block]}] RUN {label} | rough ETA {eta}", flush=True)
            start = time.monotonic()
            stop = threading.Event()
            hb = threading.Thread(target=heartbeat, args=(stop, label, start), daemon=True)
            hb.start()
            try:
                effective_algorithm = replace(cfg.algorithm, **algorithm_overrides(item))
                if args.smoke:
                    effective_algorithm = replace(effective_algorithm, adapt_start_step=4, split_min_size=4)
                steps, events, _assignments, summary = run_one(
                    trace,
                    str(item["method"]),
                    effective_algorithm,
                    cfg.objective,
                    save_assignments=False,
                )
                elapsed = time.monotonic() - start
                steps = annotate_frame(steps, item)
                events = annotate_frame(events, item) if not events.empty else events
                diag = diagnostics_from_frames(steps, events)
                summary = {
                    "experiment_block": block,
                    "condition": condition_label(item),
                    "analysis_method": (
                        f"d_kam_delay_{item['delay']}" if block == "D1"
                        else f"{item['method']}_N_{item['n_agents']}" if block == "S1"
                        else f"c_kam_{item['variant']}"
                    ),
                    "scenario": item["scenario"],
                    "seed": int(item["seed"]),
                    "method": item["method"],
                    "experimental_n_agents": int(item["n_agents"]),
                    "experimental_initial_n_units": int(item["n_units"]),
                    "wall_elapsed_seconds": float(elapsed),
                    **summary,
                    **diag,
                }
                if block == "D1":
                    summary["dkam_observation_delay"] = int(item["delay"])
                elif block == "S1":
                    summary["scale_n_agents"] = int(item["n_agents"])
                else:
                    summary["ablation_variant"] = str(item["variant"])
                marker = {
                    "protocol_id": PROTOCOL_ID,
                    "context_hash": context_hash,
                    "item_id": label,
                    "item_hash": i_hash,
                    "trace_sha256": trace_hash,
                    "trace_manifest": trace_info,
                    "algorithm_sha256": a_hash,
                    "wall_elapsed_seconds": float(elapsed),
                    "completed_utc": utc_now(),
                    "status": "complete",
                }
                write_checkpoint(cp, steps=steps, events=events, summary=summary, marker=marker)
            except BaseException as exc:
                elapsed = time.monotonic() - start
                cp.mkdir(parents=True, exist_ok=True)
                failure = {
                    "protocol_id": PROTOCOL_ID,
                    "context_hash": context_hash,
                    "item": item,
                    "item_id": label,
                    "trace_sha256": trace_hash,
                    "algorithm_sha256": a_hash,
                    "wall_elapsed_seconds": float(elapsed),
                    "failed_utc": utc_now(),
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_text(cp / "failure.json", json.dumps(failure, indent=2, sort_keys=True, default=json_default))
                append_progress(progress_path, {"timestamp_utc": utc_now(), "status": "FAILED", "overall_completed": completed, "overall_total": total, "block": block, "item_id": label, "elapsed_seconds": elapsed, "exception": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                stop.set()
                hb.join(timeout=2.0)

            completed += 1
            block_done[block] += 1
            run_durations.append(float(elapsed))
            avg = float(np.mean(run_durations))
            remaining = total - completed
            eta = human_duration(avg * remaining)
            total_elapsed = human_duration(time.monotonic() - invocation_start)
            print(f"    DONE {label} in {human_duration(elapsed)} | overall {completed}/{total} | invocation {total_elapsed} | rough ETA {eta}", flush=True)
            append_progress(progress_path, {"timestamp_utc": utc_now(), "status": "COMPLETE", "overall_completed": completed, "overall_total": total, "block": block, "block_completed": block_done[block], "block_total": block_totals[block], "item_id": label, "elapsed_seconds": elapsed, "rough_eta_seconds": avg * remaining})
            atomic_write_text(out / "progress_state.json", json.dumps({"protocol_id": PROTOCOL_ID, "updated_utc": utc_now(), "overall_completed": completed, "overall_total": total, "block_completed": block_done, "block_totals": block_totals, "last_completed_item": label, "rough_eta_seconds": avg * remaining}, indent=2, sort_keys=True, default=json_default))

        del trace
        gc.collect()

    finalize(plan, out, context=context, trace_manifest=trace_manifest, c1_results=args.c1_results)
    print(f"\n{PROTOCOL_ID} complete: {completed}/{total} runs.", flush=True)


if __name__ == "__main__":
    main()
