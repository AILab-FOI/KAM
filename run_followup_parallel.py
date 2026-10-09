from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Reuse the frozen scientific protocol/checkpoint implementation. This file is
# an operational scheduler only; it does not alter KAM-Bench scientific code.
import run_followup_experiments as serial
from kam_bench.cli import _load_config
from kam_bench.experiments import _source_hashes, _trace_sha256
from kam_bench.scenarios import generate_trace
from kam_bench.simulation import run_one

PARALLEL_RUNNER_ID = "D1-S1-A1-parallel-resume-v1"
DEFAULT_STRESS_1024_SEEDS = [5001, 5002, 5003]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def mem_available_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                kib = float(line.split()[1])
                return kib / (1024.0**2)
    except Exception:
        pass
    return float("nan")


def swap_used_gib() -> float:
    try:
        vals = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("SwapTotal:", "SwapFree:")):
                vals[line.split(":")[0]] = float(line.split()[1]) / (1024.0**2)
        return max(vals.get("SwapTotal", 0.0) - vals.get("SwapFree", 0.0), 0.0)
    except Exception:
        return float("nan")


def auto_workers(kind: str, requested: Optional[int]) -> int:
    if requested is not None and requested > 0:
        return int(requested)
    cpu = os.cpu_count() or 4
    avail = mem_available_gib()
    # Conservative estimates include the dense trace plus Python/algorithm state.
    if kind == "small":
        cpu_cap, per_worker, reserve = min(12, max(2, cpu // 2)), 0.75, 4.0
    elif kind == "512":
        cpu_cap, per_worker, reserve = min(8, max(2, cpu // 3)), 1.25, 5.0
    else:
        cpu_cap, per_worker, reserve = min(6, max(1, cpu // 4)), 2.40, 6.0
    if not np.isfinite(avail):
        return max(1, cpu_cap)
    by_mem = max(1, int(math.floor(max(avail - reserve, per_worker) / per_worker)))
    return max(1, min(cpu_cap, by_mem))


def parse_seed_spec(spec: str) -> List[int]:
    vals: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(part))
    vals = sorted(set(vals))
    bad = [x for x in vals if x not in serial.DESIGN["S1"]["seeds"]]
    if bad:
        raise ValueError(f"1024 seeds must come from frozen S1 seeds 5001..5010; invalid: {bad}")
    if not vals:
        raise ValueError("no 1024 seeds selected")
    return vals


def amended_plan(stress_1024_seeds: List[int]) -> List[Dict[str, object]]:
    full = serial.build_plan(smoke=False, blocks=["D1", "S1", "A1"])
    keep = []
    for item in full:
        if item["block"] == "S1" and int(item["n_agents"]) == 1024 and int(item["seed"]) not in stress_1024_seeds:
            continue
        keep.append(item)
    return keep


def load_or_create_context(out: Path, config_path: Path) -> Dict[str, object]:
    """Load a compatible scientific checkpoint context, or create it for a fresh clone.

    The context deliberately uses the frozen serial protocol runner hash. Parallel execution
    changes scheduling only, so fresh and resumed runs share the same scientific checkpoint
    identity as the original D1/S1/A1 protocol.
    """
    cp = out / "checkpoint_context.json"
    cfg = _load_config(config_path)
    expected = {
        "protocol_id": serial.PROTOCOL_ID,
        "smoke": False,
        "blocks": ["D1", "S1", "A1"],
        "base_config_sha256": serial.canonical_hash(cfg.to_dict()),
        "scientific_source_sha256": _source_hashes(),
        "design_sha256": serial.canonical_hash(serial.DESIGN),
        "runner_sha256": serial.sha256_file(Path(serial.__file__).resolve()),
    }
    expected["context_sha256"] = serial.canonical_hash(expected)

    if not cp.exists():
        serial.atomic_write_text(cp, json.dumps(expected, indent=2, sort_keys=True))
        print(f"Created fresh checkpoint context: {cp}")
        return expected

    context = json.loads(cp.read_text(encoding="utf-8"))
    for key in ["protocol_id", "base_config_sha256", "scientific_source_sha256", "design_sha256", "runner_sha256"]:
        if context.get(key) != expected.get(key):
            raise RuntimeError(
                f"checkpoint context mismatch for {key}: {context.get(key)!r} != {expected.get(key)!r}. "
                "Use a new output directory rather than mixing scientific protocols."
            )
    return context


def prelim_done(out: Path, item: Dict[str, object], context_hash: str, cfg) -> bool:
    cp = serial.checkpoint_dir(out, item)
    marker = serial.checkpoint_prelim_complete(
        cp,
        context_hash=context_hash,
        item_hash=serial.canonical_hash(item),
        algo_hash=serial.algorithm_hash(cfg.algorithm, item),
    )
    return marker is not None


def worker_run(payload: Dict[str, object]) -> Dict[str, object]:
    """Run one independent scientific item in a spawned process."""
    item = dict(payload["item"])
    config_path = Path(str(payload["config_path"]))
    out = Path(str(payload["out"]))
    context_hash = str(payload["context_hash"])
    start = time.monotonic()
    label = serial.item_id(item)
    cp = serial.checkpoint_dir(out, item)
    try:
        cfg = _load_config(config_path)
        i_hash = serial.canonical_hash(item)
        a_hash = serial.algorithm_hash(cfg.algorithm, item)

        trace = generate_trace(
            str(item["scenario"]), int(item["seed"]), int(item["steps"]),
            int(item["n_agents"]), int(item["n_units"]),
            supervisor_fanout=cfg.algorithm.supervisor_fanout,
        )
        trace_hash = _trace_sha256(trace)
        trace_info = {
            "scenario": item["scenario"], "seed": int(item["seed"]), "steps": int(item["steps"]),
            "n_agents": int(item["n_agents"]), "initial_n_units": int(item["n_units"]),
            "description": trace.metadata.get("description", ""),
            "generator_status": trace.metadata.get("generator_status", "unknown"),
            "trace_sha256": trace_hash,
        }
        if serial.checkpoint_complete(cp, context_hash=context_hash, item_hash=i_hash, trace_hash=trace_hash, algo_hash=a_hash):
            return {"status": "SKIP", "item": item, "item_id": label, "elapsed": time.monotonic()-start, "trace_sha256": trace_hash}

        effective_algorithm = replace(cfg.algorithm, **serial.algorithm_overrides(item))
        steps, events, _assignments, summary = run_one(
            trace, str(item["method"]), effective_algorithm, cfg.objective, save_assignments=False
        )
        elapsed = time.monotonic() - start
        steps = serial.annotate_frame(steps, item)
        events = serial.annotate_frame(events, item) if not events.empty else events
        diag = serial.diagnostics_from_frames(steps, events)
        block = str(item["block"])
        summary = {
            "experiment_block": block,
            "condition": serial.condition_label(item),
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
            "protocol_id": serial.PROTOCOL_ID,
            "context_hash": context_hash,
            "item_id": label,
            "item_hash": i_hash,
            "trace_sha256": trace_hash,
            "trace_manifest": trace_info,
            "algorithm_sha256": a_hash,
            "wall_elapsed_seconds": float(elapsed),
            "completed_utc": utc_now(),
            "status": "complete",
            "operational_runner": PARALLEL_RUNNER_ID,
        }
        serial.write_checkpoint(cp, steps=steps, events=events, summary=summary, marker=marker)
        del trace
        gc.collect()
        return {"status": "COMPLETE", "item": item, "item_id": label, "elapsed": elapsed, "trace_sha256": trace_hash}
    except BaseException as exc:
        elapsed = time.monotonic() - start
        cp.mkdir(parents=True, exist_ok=True)
        failure = {
            "protocol_id": serial.PROTOCOL_ID,
            "operational_runner": PARALLEL_RUNNER_ID,
            "context_hash": context_hash,
            "item": item,
            "item_id": label,
            "wall_elapsed_seconds": float(elapsed),
            "failed_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        serial.atomic_write_text(cp / "failure.json", json.dumps(failure, indent=2, sort_keys=True, default=serial.json_default))
        return {"status": "FAILED", "item": item, "item_id": label, "elapsed": elapsed, "error": f"{type(exc).__name__}: {exc}"}


def terminate_executor(executor: cf.ProcessPoolExecutor) -> None:
    # Python 3.10 lacks terminate_workers(); terminate spawned children explicitly.
    try:
        for proc in list(getattr(executor, "_processes", {}).values()):
            if proc.is_alive():
                proc.terminate()
    except Exception:
        pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def run_phase(name: str, items: List[Dict[str, object]], workers: int, *, config_path: Path, out: Path,
              context_hash: str, total_plan: int, initial_done: int, progress_path: Path) -> Tuple[int, bool]:
    if not items:
        print(f"\n===== {name}: nothing pending =====", flush=True)
        return initial_done, True
    print(f"\n===== {name}: {len(items)} pending item(s), {workers} worker process(es) =====", flush=True)
    print(f"MemAvailable at phase start: {mem_available_gib():.1f} GiB; swap used: {swap_used_gib():.1f} GiB", flush=True)
    print("Each worker is a separate Python process; scientific runs remain deterministic and independently checkpointed.", flush=True)

    ctx = mp.get_context("spawn")
    executor = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    futures: Dict[cf.Future, Tuple[Dict[str, object], float]] = {}
    pending_iter = iter(items)
    completed = initial_done
    phase_start = time.monotonic()
    failures = []

    def submit_one() -> bool:
        try:
            item = next(pending_iter)
        except StopIteration:
            return False
        payload = {"item": item, "config_path": str(config_path), "out": str(out), "context_hash": context_hash}
        fut = executor.submit(worker_run, payload)
        futures[fut] = (item, time.monotonic())
        return True

    for _ in range(min(workers, len(items))):
        submit_one()

    try:
        while futures:
            done, _ = cf.wait(list(futures), timeout=60.0, return_when=cf.FIRST_COMPLETED)
            if not done:
                active = sorted(((time.monotonic()-st, serial.item_id(it)) for it, st in futures.values()), reverse=True)
                top = active[:min(5, len(active))]
                print(f"  ... {len(futures)} active; phase elapsed {human(time.monotonic()-phase_start)}", flush=True)
                for elapsed, label in top:
                    print(f"      {label} | {human(elapsed)}", flush=True)
                continue
            for fut in done:
                item, submitted = futures.pop(fut)
                result = fut.result()
                status = result["status"]
                if status in {"COMPLETE", "SKIP"}:
                    completed += 1
                    elapsed = float(result.get("elapsed", 0.0))
                    print(f"[{completed}/{total_plan}] {status:8s} {result['item_id']} | worker {human(elapsed)}", flush=True)
                    serial.append_progress(progress_path, {
                        "timestamp_utc": utc_now(), "status": f"PARALLEL_{status}",
                        "overall_completed_amended": completed, "overall_total_amended": total_plan,
                        "phase": name, "item_id": result["item_id"], "elapsed_seconds": elapsed,
                    })
                    submit_one()
                else:
                    failures.append(result)
                    print(f"FAILED {result['item_id']}: {result.get('error')}", flush=True)
                    break
            if failures:
                break
    except KeyboardInterrupt:
        print("\nInterrupt received. Terminating currently running worker processes; all previously completed checkpoints are safe.", flush=True)
        terminate_executor(executor)
        return completed, False

    if failures:
        terminate_executor(executor)
        raise RuntimeError(f"parallel phase {name} stopped after failure: {failures[0]['item_id']} -> {failures[0].get('error')}")
    executor.shutdown(wait=True)
    return completed, True


def trace_manifest_from_checkpoints(plan: List[Dict[str, object]], out: Path) -> Dict[str, Dict[str, object]]:
    manifest: Dict[str, Dict[str, object]] = {}
    for item in plan:
        marker_path = serial.checkpoint_dir(out, item) / "complete.json"
        if not marker_path.exists():
            raise RuntimeError(f"missing checkpoint marker during finalization: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        info = marker.get("trace_manifest")
        if not isinstance(info, dict):
            raise RuntimeError(f"checkpoint lacks trace_manifest: {marker_path}")
        manifest[serial.canonical_hash(serial.trace_key(item))] = info
    return manifest


def write_primary_scaling_without_stress(out: Path) -> None:
    path = out / "S1" / "run_summary_new.csv"
    if not path.exists():
        return
    s1 = pd.read_csv(path)
    primary = s1[pd.to_numeric(s1["experimental_n_agents"], errors="coerce") <= 512].copy()
    means = primary.groupby(["scenario", "experimental_n_agents", "method"], as_index=False).mean(numeric_only=True)
    slope_metrics = ["total_decision_time_ms", "total_candidate_evaluations", "total_control_message_hops", "total_reorganization_cost", "disturbance_backlog_auc_mean"]
    rows = []
    for scenario in serial.SCENARIOS:
        for method in ["c_kam", "d_kam"]:
            g = means[(means["scenario"] == scenario) & (means["method"] == method)].sort_values("experimental_n_agents")
            for metric in slope_metrics:
                x = pd.to_numeric(g["experimental_n_agents"], errors="coerce").to_numpy(float)
                y = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
                if mask.sum() >= 3:
                    lr = stats.linregress(np.log2(x[mask]), np.log2(y[mask]))
                    rows.append({"scenario":scenario,"method":method,"metric":metric,"scaling_exponent":float(lr.slope),"r_squared":float(lr.rvalue**2),"p_value":float(lr.pvalue),"n_sizes":int(mask.sum()),"max_n_in_primary_fit":512})
    pd.DataFrame(rows).to_csv(out / "S1" / "scaling_exponents_primary_N32_512.csv", index=False)


def finalize_amended(plan: List[Dict[str, object]], out: Path, context: Dict[str, object], c1_results: Optional[Path], stress_seeds: List[int]) -> None:
    trace_manifest = trace_manifest_from_checkpoints(plan, out)
    serial.finalize(plan, out, context=context, trace_manifest=trace_manifest, c1_results=c1_results)
    write_primary_scaling_without_stress(out)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full = (stress_seeds == list(serial.DESIGN["S1"]["seeds"]))
    amendment = {
        "id": "S1.1-computational-feasibility-amendment",
        "adopted_before_any_N1024_run_completed": True,
        "reason": "The first N=1024 C-KAM run exceeded six hours while using one CPU core; the amendment changes only replication count and execution scheduling, not the scientific algorithms, parameters, workloads, steps, or metrics.",
        "parallel_execution": True,
        "N_32_to_512_seeds": list(serial.DESIGN["S1"]["seeds"]),
        "N_1024_stress_seeds": stress_seeds,
        "N_1024_role": "descriptive stress point" if not full else "full planned replication",
        "primary_scaling_fit": "N=32..512 with 10 seeds per size" if not full else "N=32..1024 with 10 seeds per size",
    }
    manifest["operational_amendment"] = amendment
    manifest["plan_run_count"] = len(plan)
    manifest["expected_production_run_count"] = len(plan)
    manifest["parallel_runner_id"] = PARALLEL_RUNNER_ID
    serial.atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, default=serial.json_default))
    serial.atomic_write_text(out / "S1_1_OPERATIONAL_AMENDMENT.json", json.dumps(amendment, indent=2, sort_keys=True))
    serial.atomic_write_text(out / "ALL_BLOCKS_COMPLETE.json", json.dumps({"protocol_id":serial.PROTOCOL_ID,"operational_runner":PARALLEL_RUNNER_ID,"completed_utc":utc_now(),"run_count":len(plan),"status":"complete"}, indent=2, sort_keys=True))


def make_return_zip(out: Path, root: Path) -> Path:
    zip_path = root / "FOLLOWUP_D1_S1_A1_PARALLEL_RESULTS_RETURN_ME.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(out.parent)
            if "checkpoints" in rel.parts:
                continue
            z.write(path, rel.as_posix())
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel checkpoint-compatible start/resume runner for D1/S1/A1")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--c1-results", type=Path, default=None)
    ap.add_argument("--s1-1024-seeds", default=os.environ.get("FOLLOWUP_1024_SEEDS", "5001,5002,5003"))
    ap.add_argument("--workers-small", type=int, default=int(os.environ["FOLLOWUP_WORKERS_SMALL"]) if os.environ.get("FOLLOWUP_WORKERS_SMALL") else None)
    ap.add_argument("--workers-512", type=int, default=int(os.environ["FOLLOWUP_WORKERS_512"]) if os.environ.get("FOLLOWUP_WORKERS_512") else None)
    ap.add_argument("--workers-1024", type=int, default=int(os.environ["FOLLOWUP_WORKERS_1024"]) if os.environ.get("FOLLOWUP_WORKERS_1024") else None)
    ap.add_argument("--only-up-to-512", action="store_true", help="Run/finish everything except N=1024 and stop without finalization")
    args = ap.parse_args()

    stress_seeds = parse_seed_spec(args.s1_1024_seeds)
    plan = amended_plan(stress_seeds)
    out = args.out.resolve()
    config = args.config.resolve()
    out.mkdir(parents=True, exist_ok=True)
    context = load_or_create_context(out, config)
    context_hash = str(context["context_sha256"])
    cfg = _load_config(config)

    small_w = auto_workers("small", args.workers_small)
    w512 = auto_workers("512", args.workers_512)
    w1024 = auto_workers("1024", args.workers_1024)

    done_flags = [prelim_done(out, item, context_hash, cfg) for item in plan]
    initial_done = int(sum(done_flags))
    pending = [item for item, done in zip(plan, done_flags) if not done]
    print(f"Parallel operational runner: {PARALLEL_RUNNER_ID}")
    print(f"Scientific checkpoint protocol: {serial.PROTOCOL_ID}")
    print(f"Amended plan size: {len(plan)} runs; compatible checkpoints already complete: {initial_done}; pending: {len(pending)}")
    print(f"N=1024 seeds: {stress_seeds} ({'FULL 10-seed plan' if len(stress_seeds)==10 else 'descriptive stress-point amendment'})")
    print(f"CPU logical processors: {os.cpu_count()}; MemAvailable: {mem_available_gib():.1f} GiB; swap used: {swap_used_gib():.1f} GiB")
    print(f"Auto workers: small={small_w}, N512={w512}, N1024={w1024}")
    if swap_used_gib() > 0.25:
        print("WARNING: swap is already in use. Close memory-heavy applications before N=1024, or lower FOLLOWUP_WORKERS_1024.")

    # Prioritize quick completion of A1 and all <=512 evidence before the 1024 stress runs.
    phases = [
        ("D1/A1/S1 <=256", [x for x in pending if x["block"] in {"D1","A1"} or (x["block"]=="S1" and int(x["n_agents"]) <= 256)], small_w),
        ("S1 N=512", [x for x in pending if x["block"]=="S1" and int(x["n_agents"]) == 512], w512),
    ]
    if not args.only_up_to_512:
        phases.append(("S1 N=1024 stress", [x for x in pending if x["block"]=="S1" and int(x["n_agents"]) == 1024], w1024))

    progress_path = out / "parallel_progress_history.csv"
    completed = initial_done
    for name, items, workers in phases:
        completed, ok = run_phase(name, items, workers, config_path=config, out=out, context_hash=context_hash,
                                  total_plan=len(plan), initial_done=completed, progress_path=progress_path)
        if not ok:
            print("Parallel run interrupted. Re-run the same command to resume.")
            return

    if args.only_up_to_512:
        print("\nEverything requested up to N=512 is checkpointed. N=1024 was intentionally not launched.")
        print("Run again without --only-up-to-512 when ready for the 1024 stress point.")
        return

    # Recount from disk before finalization.
    missing = [serial.item_id(x) for x in plan if not prelim_done(out, x, context_hash, cfg)]
    if missing:
        raise RuntimeError(f"cannot finalize: {len(missing)} amended-plan checkpoints are missing; first: {missing[:5]}")

    print("\n===== Finalizing consolidated D1/S1/A1 outputs =====")
    finalize_amended(plan, out, context, args.c1_results, stress_seeds)
    zip_path = make_return_zip(out, Path.cwd())
    print("\nAll amended-plan experiments complete.")
    print(f"Upload this file back to ChatGPT: {zip_path}")


if __name__ == "__main__":
    main()
