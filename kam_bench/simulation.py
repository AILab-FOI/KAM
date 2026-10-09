from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .algorithms import make_algorithm
from .config import AlgorithmConfig, ObjectiveWeights
from .metrics import objective_value, state_metrics
from .model import EPS, Organization, ScenarioTrace, Snapshot


def _method_seed(trace_seed: int, method_name: str) -> int:
    digest = hashlib.sha256(f"{trace_seed}:{method_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _service_step(
    offered: np.ndarray,
    org: Organization,
    total_capacity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Proportionally share each unit's capacity among its current members."""
    offered = np.asarray(offered, dtype=float)
    processed = np.zeros_like(offered)
    for u in org.units:
        members = org.members(u)
        demand = float(offered[members].sum())
        cap = float(org.capacity_shares[u] * total_capacity)
        if demand <= EPS:
            continue
        frac = min(1.0, cap / demand)
        processed[members] = offered[members] * frac
    backlog = np.maximum(offered - processed, 0.0)
    return processed, backlog



def _disturbance_auc(steps: pd.DataFrame) -> Tuple[float, float]:
    """Mean post-disturbance backlog and service-loss area until next event."""
    if steps.empty:
        return 0.0, 0.0
    event_idx = [int(i) for i, x in enumerate(steps["disturbance"].astype(str)) if x]
    if not event_idx:
        return 0.0, 0.0
    backlog_aucs = []
    service_aucs = []
    for pos, start in enumerate(event_idx):
        stop = event_idx[pos + 1] if pos + 1 < len(event_idx) else len(steps)
        window = steps.iloc[start:stop]
        backlog_aucs.append(float(window["backlog_to_capacity"].sum()))
        service_aucs.append(float(np.maximum(1.0 - window["service_ratio"].to_numpy(float), 0.0).sum()))
    return float(np.mean(backlog_aucs)), float(np.mean(service_aucs))

def run_one(
    trace: ScenarioTrace,
    method_name: str,
    algorithm_cfg: AlgorithmConfig,
    objective_weights: ObjectiveWeights,
    save_assignments: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Run one method on one pre-generated exogenous scenario trace.

    Arriving work and communication are paired across methods. Unserved work can
    be carried as endogenous backlog, so organizational decisions affect future
    offered load without changing the exogenous arrivals themselves.
    """
    org = Organization.from_trace(trace)
    algorithm = make_algorithm(
        method_name,
        algorithm_cfg,
        objective_weights,
        seed=_method_seed(trace.seed, method_name),
    )
    history: List[Snapshot] = []
    step_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    assignment_rows: List[Dict[str, object]] = []

    backlog = np.zeros(trace.n_agents, dtype=float)
    cumulative_arrivals = 0.0
    cumulative_processed = 0.0
    total_reorg_cost = 0.0
    total_control = 0
    total_control_hops = 0
    total_candidate_evaluations = 0
    total_moved = 0
    total_decision_ms = 0.0
    op_counts = {"move": 0, "merge": 0, "split": 0, "repartition": 0}

    for exogenous in trace.snapshots:
        arrivals = np.asarray(exogenous.work, dtype=float)
        offered = arrivals + backlog if algorithm_cfg.carry_backlog else arrivals.copy()
        runtime = Snapshot(
            step=exogenous.step,
            work=offered,
            communication=exogenous.communication,
            total_capacity=exogenous.total_capacity,
            phase=exogenous.phase,
            disturbance=exogenous.disturbance,
        )
        history.append(runtime)

        before_metrics = state_metrics(
            trace, runtime, org, algorithm_cfg.communication_threshold
        )
        before_obj, _ = objective_value(
            before_metrics, objective_weights, 0.0, float(offered.sum())
        )

        result = algorithm.adapt(trace, runtime, history, org)
        result.organization.validate()
        org = result.organization
        after_metrics = state_metrics(
            trace, runtime, org, algorithm_cfg.communication_threshold
        )
        after_obj, after_parts = objective_value(
            after_metrics,
            objective_weights,
            result.reorganization_cost,
            float(offered.sum()),
        )

        capacity_loss = (
            float(result.reorganization_cost)
            if algorithm_cfg.reorganization_consumes_capacity
            else 0.0
        )
        effective_capacity = max(float(runtime.total_capacity) - capacity_loss, 0.0)
        processed_vec, next_backlog = _service_step(offered, org, effective_capacity)
        if not algorithm_cfg.carry_backlog:
            next_backlog[:] = 0.0

        step_arrivals = float(arrivals.sum())
        step_offered = float(offered.sum())
        step_processed = float(processed_vec.sum())
        step_backlog = float(next_backlog.sum())
        cumulative_arrivals += step_arrivals
        cumulative_processed += step_processed
        cumulative_completion = cumulative_processed / max(cumulative_arrivals, EPS)
        service_ratio = step_processed / max(step_offered, EPS)

        total_reorg_cost += float(result.reorganization_cost)
        total_control += int(result.control_messages)
        total_control_hops += int(result.control_message_hops)
        total_candidate_evaluations += int(result.candidate_evaluations)
        total_moved += int(result.moved_agents)
        total_decision_ms += float(result.decision_time_ms)

        for idx, op in enumerate(result.operations):
            op_counts[op.operator] = op_counts.get(op.operator, 0) + 1
            payload = dict(op.payload)
            if "assignment" in payload:
                payload["assignment"] = f"<{len(payload['assignment'])} entries>"
            if "left" in payload:
                payload["left"] = ",".join(map(str, payload["left"]))
            if "right" in payload:
                payload["right"] = ",".join(map(str, payload["right"]))
            event_rows.append(
                {
                    "scenario": trace.name,
                    "seed": trace.seed,
                    "method": method_name,
                    "step": runtime.step,
                    "event_index": idx,
                    "operator": op.operator,
                    "authority": op.authority,
                    "observed_step": op.observed_step,
                    "observation_lag": runtime.step - op.observed_step,
                    "estimated_delta": op.estimated_delta,
                    "actual_delta_current_state": op.actual_delta,
                    "scope_agents": op.scope_agents,
                    "scope_units": op.scope_units,
                    **payload,
                }
            )

        row = {
            "scenario": trace.name,
            "seed": trace.seed,
            "method": method_name,
            "step": runtime.step,
            "phase": runtime.phase,
            "disturbance": runtime.disturbance,
            "observed_step": result.observed_step,
            "observation_lag": runtime.step - result.observed_step,
            "n_units": len(org.units),
            "arrival_work": step_arrivals,
            "offered_work": step_offered,
            "processed_work": step_processed,
            "backlog_end": step_backlog,
            "backlog_to_capacity": step_backlog / max(runtime.total_capacity, EPS),
            "service_ratio": service_ratio,
            "cumulative_completion_ratio": cumulative_completion,
            "raw_load_imbalance": after_metrics.raw_load_imbalance,
            "kam_capacity_mismatch": after_metrics.kam_capacity_mismatch,
            "overload_penalty": after_metrics.overload_penalty,
            "overload_fraction": after_metrics.overload_fraction,
            "makespan_proxy": after_metrics.makespan_proxy,
            "cross_communication": after_metrics.cross_communication,
            "cross_communication_ratio": after_metrics.cross_communication_ratio,
            "coordination_cost": after_metrics.coordination_cost,
            "coordination_cost_normalized": after_metrics.coordination_cost_normalized,
            "management_overhead_normalized": after_metrics.management_overhead_normalized,
            "kam_factor_mean": after_metrics.kam_factor_mean,
            "kam_factor_std": after_metrics.kam_factor_std,
            "objective": after_obj,
            "objective_kam_imbalance": after_parts["kam_imbalance"],
            "objective_overload": after_parts["overload"],
            "objective_coordination": after_parts["coordination"],
            "objective_management": after_parts["management"],
            "objective_reorganization": after_parts["reorganization"],
            "objective_before_adaptation": before_obj,
            "adaptation_delta_current_step": after_obj - before_obj,
            "step_reorganization_cost": result.reorganization_cost,
            "effective_capacity_after_reorganization": effective_capacity,
            "step_moved_agents": result.moved_agents,
            "step_control_messages": result.control_messages,
            "step_control_message_hops": result.control_message_hops,
            "step_candidate_evaluations": result.candidate_evaluations,
            "step_proposals": result.proposal_count,
            "step_rejected_proposals": result.rejected_proposals,
            "decision_time_ms": result.decision_time_ms,
            "step_operations": len(result.operations),
        }
        step_rows.append(row)

        if save_assignments:
            for agent, unit in enumerate(org.assignment.tolist()):
                assignment_rows.append(
                    {
                        "scenario": trace.name,
                        "seed": trace.seed,
                        "method": method_name,
                        "step": runtime.step,
                        "agent": agent,
                        "unit": int(unit),
                    }
                )
        backlog = next_backlog

    steps = pd.DataFrame(step_rows)
    events = pd.DataFrame(event_rows)
    assignments = pd.DataFrame(assignment_rows)

    disturbance_backlog_auc, disturbance_service_loss_auc = _disturbance_auc(steps)

    summary: Dict[str, float] = {
        "steps": float(len(steps)),
        "disturbance_backlog_auc_mean": disturbance_backlog_auc,
        "disturbance_service_loss_auc_mean": disturbance_service_loss_auc,
        "completion_ratio": float(cumulative_processed / max(cumulative_arrivals, EPS)),
        # backward-compatible name, now defined as cumulative completion ratio
        "mean_throughput_ratio": float(cumulative_processed / max(cumulative_arrivals, EPS)),
        "mean_service_ratio": float(steps["service_ratio"].mean()),
        "mean_backlog_to_capacity": float(steps["backlog_to_capacity"].mean()),
        "final_backlog": float(steps["backlog_end"].iloc[-1]),
        "max_backlog": float(steps["backlog_end"].max()),
        "mean_raw_load_imbalance": float(steps["raw_load_imbalance"].mean()),
        "mean_kam_capacity_mismatch": float(steps["kam_capacity_mismatch"].mean()),
        # backward-compatible aliases
        "mean_raw_load_cv": float(steps["raw_load_imbalance"].mean()),
        "mean_kam_load_cv": float(steps["kam_capacity_mismatch"].mean()),
        "mean_overload_penalty": float(steps["overload_penalty"].mean()),
        "mean_makespan_proxy": float(steps["makespan_proxy"].mean()),
        "mean_cross_communication_ratio": float(steps["cross_communication_ratio"].mean()),
        "mean_coordination_cost_normalized": float(
            steps["coordination_cost_normalized"].mean()
        ),
        "mean_management_overhead_normalized": float(
            steps["management_overhead_normalized"].mean()
        ),
        "mean_objective": float(steps["objective"].mean()),
        "final_objective": float(steps["objective"].iloc[-1]),
        "total_reorganization_cost": float(total_reorg_cost),
        "total_moved_agents": float(total_moved),
        "total_control_messages": float(total_control),
        "total_control_message_hops": float(total_control_hops),
        "total_candidate_evaluations": float(total_candidate_evaluations),
        "total_decision_time_ms": float(total_decision_ms),
        "mean_decision_time_ms": float(steps["decision_time_ms"].mean()),
        "move_count": float(op_counts.get("move", 0)),
        "merge_count": float(op_counts.get("merge", 0)),
        "split_count": float(op_counts.get("split", 0)),
        "repartition_count": float(op_counts.get("repartition", 0)),
        "final_n_units": float(steps["n_units"].iloc[-1]),
    }
    return steps, events, assignments, summary
