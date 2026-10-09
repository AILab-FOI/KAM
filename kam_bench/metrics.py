from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import ObjectiveWeights
from .model import EPS, Organization, ScenarioTrace, Snapshot


@dataclass(frozen=True)
class StateMetrics:
    raw_work_by_unit: Dict[int, float]
    kam_work_by_unit: Dict[int, float]
    capacity_by_unit: Dict[int, float]
    capacity_share_by_unit: Dict[int, float]
    utilization_by_unit: Dict[int, float]
    raw_load_imbalance: float
    kam_capacity_mismatch: float
    overload_penalty: float
    overload_fraction: float
    processed_work_capacity_only: float
    service_ratio_capacity_only: float
    makespan_proxy: float
    cross_communication: float
    cross_communication_ratio: float
    coordination_cost: float
    coordination_cost_normalized: float
    management_overhead_normalized: float
    total_communication: float
    total_work: float
    total_kam_exertion: float
    kam_factor_mean: float
    kam_factor_std: float

    # Compatibility aliases used by older result readers.
    @property
    def raw_load_cv(self) -> float:
        return self.raw_load_imbalance

    @property
    def kam_load_cv(self) -> float:
        return self.kam_capacity_mismatch

    @property
    def processed_work(self) -> float:
        return self.processed_work_capacity_only

    @property
    def throughput_ratio(self) -> float:
        return self.service_ratio_capacity_only


def _share_mismatch(values: Dict[int, float], shares: Dict[int, float]) -> float:
    """Pearson/chi-square distance between work/exertion and capacity shares.

    B = sqrt(sum_u (p_u - s_u)^2 / s_u), where p_u is the share of
    work/exertion and s_u is the capacity share. It is zero exactly when the
    two distributions match and, unlike an unweighted CV, cannot be improved
    merely by splitting a unit proportionally.
    """
    total = float(sum(values.values()))
    if total <= EPS:
        return 0.0
    out = 0.0
    for u, s in shares.items():
        s = max(float(s), EPS)
        p = float(values.get(u, 0.0)) / total
        out += (p - s) ** 2 / s
    return float(np.sqrt(max(out, 0.0)))


def symmetric_communication(snapshot: Snapshot) -> np.ndarray:
    c = np.asarray(snapshot.communication, dtype=float)
    return 0.5 * (c + c.T)


def kam_agent_factors(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    communication_threshold: float = 0.0,
) -> np.ndarray:
    """Compute the generalized KAM factor F*kq*kc*ks for each agent bundle.

    Each simulated agent carries one aggregate activity bundle. The activity's
    own organizational unit is counted once, and F increases by one for each
    *external* organizational unit with which the agent exchanges information
    during the observation window:

        F_i = 1 + |{u != P(i): communication(i,u) > threshold}|.

    This is a precise simulation interpretation of the source paper's frequency
    table and avoids an activity receiving F=0 merely because it had no explicit
    within-unit message in a short observation window.
    """
    n = trace.n_agents
    comm = np.asarray(snapshot.communication, dtype=float)
    if comm.shape != (n, n):
        raise ValueError("communication shape mismatch")
    F = np.ones(n, dtype=float)
    thr = float(communication_threshold)
    for i in range(n):
        touched_external = set()
        partners = np.flatnonzero((comm[i] + comm[:, i]) > thr)
        own = int(org.assignment[i])
        for j in partners:
            u = int(org.assignment[int(j)])
            if u != own:
                touched_external.add(u)
        F[i] += float(len(touched_external))
    kq = np.where(trace.q_impact, float(trace.metadata.get("kq", 1.65)), 1.0)
    kc = np.where(trace.c_impact, float(trace.metadata.get("kc", 1.08)), 1.0)
    return F * kq * kc * trace.ks


def kam_agent_exertion(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    communication_threshold: float = 0.0,
) -> np.ndarray:
    return np.asarray(snapshot.work, dtype=float) * kam_agent_factors(
        trace, snapshot, org, communication_threshold
    )


def _management_overhead(org: Organization, n_agents: int) -> float:
    """Normalized span/coordination complexity of the current leaf partition.

    H = sum_u n_u log2(n_u) / (N log2(N)).

    H=1 for one monolithic unit and approaches zero as the organization is split
    into small units. It is an explicit continuation-study modelling term, not
    a quantity claimed to be present in the 2012 KAM paper. Its purpose is to
    make the cost of very large coordination domains transparent rather than
    hiding it inside communication cost.
    """
    if n_agents <= 1:
        return 0.0
    denom = float(n_agents * np.log2(n_agents))
    num = 0.0
    for u in org.units:
        n = org.size(u)
        if n > 1:
            num += float(n * np.log2(n))
    return float(num / max(denom, EPS))


def state_metrics(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    communication_threshold: float = 0.0,
) -> StateMetrics:
    org.validate()
    units = org.units
    total_capacity = float(snapshot.total_capacity)
    if total_capacity <= 0 or not np.isfinite(total_capacity):
        raise ValueError("total_capacity must be finite and positive")
    shares = {u: float(org.capacity_shares[u]) for u in units}
    capacity = {u: shares[u] * total_capacity for u in units}

    work = np.asarray(snapshot.work, dtype=float)
    if np.any(~np.isfinite(work)) or np.any(work < 0):
        raise ValueError("work must be finite and nonnegative")
    raw = {u: float(work[org.assignment == u].sum()) for u in units}

    factors = kam_agent_factors(trace, snapshot, org, communication_threshold)
    exertion = work * factors
    kam = {u: float(exertion[org.assignment == u].sum()) for u in units}

    util = {u: raw[u] / max(capacity[u], EPS) for u in units}
    util_arr = np.array([util[u] for u in units], dtype=float)
    share_arr = np.array([shares[u] for u in units], dtype=float)

    overload = np.maximum(util_arr - 1.0, 0.0)
    overload_penalty = float(np.sum(share_arr * overload**2))
    overload_fraction = float(np.sum(share_arr * (util_arr > 1.0)))
    processed = float(sum(min(raw[u], capacity[u]) for u in units))
    total_work = float(work.sum())
    service_ratio = processed / max(total_work, EPS)
    makespan = float(np.max(util_arr)) if len(util_arr) else 0.0

    comm_sym = symmetric_communication(snapshot)
    if np.any(~np.isfinite(comm_sym)) or np.any(comm_sym < 0):
        raise ValueError("communication must be finite and nonnegative")
    tri_i, tri_j = np.triu_indices(trace.n_agents, k=1)
    edge_weight = comm_sym[tri_i, tri_j]
    total_comm = float(edge_weight.sum())
    cross = 0.0
    for i, j, w in zip(tri_i.tolist(), tri_j.tolist(), edge_weight.tolist()):
        if w > 0 and int(org.assignment[i]) != int(org.assignment[j]):
            cross += float(w)
    internal = max(total_comm - cross, 0.0)
    coordination = (
        internal * float(trace.internal_comm_base)
        + cross * float(trace.remote_comm_cost)
    )
    if total_comm <= EPS:
        coord_norm = 0.0
        cross_ratio = 0.0
    else:
        cross_ratio = cross / total_comm
        coord_norm = coordination / max(total_comm * float(trace.remote_comm_cost), EPS)

    management = _management_overhead(org, trace.n_agents)

    return StateMetrics(
        raw_work_by_unit=raw,
        kam_work_by_unit=kam,
        capacity_by_unit=capacity,
        capacity_share_by_unit=shares,
        utilization_by_unit=util,
        raw_load_imbalance=_share_mismatch(raw, shares),
        kam_capacity_mismatch=_share_mismatch(kam, shares),
        overload_penalty=overload_penalty,
        overload_fraction=overload_fraction,
        processed_work_capacity_only=processed,
        service_ratio_capacity_only=float(service_ratio),
        makespan_proxy=makespan,
        cross_communication=cross,
        cross_communication_ratio=float(cross_ratio),
        coordination_cost=float(coordination),
        coordination_cost_normalized=float(coord_norm),
        management_overhead_normalized=float(management),
        total_communication=total_comm,
        total_work=total_work,
        total_kam_exertion=float(exertion.sum()),
        kam_factor_mean=float(np.mean(factors)),
        kam_factor_std=float(np.std(factors)),
    )


def objective_value(
    metrics: StateMetrics,
    weights: ObjectiveWeights,
    reorganization_cost: float = 0.0,
    total_work: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    reorg_norm = float(reorganization_cost / max(total_work, EPS))
    parts = {
        "kam_imbalance": float(metrics.kam_capacity_mismatch),
        "overload": float(metrics.overload_penalty),
        "coordination": float(metrics.coordination_cost_normalized),
        "management": float(metrics.management_overhead_normalized),
        "reorganization": reorg_norm,
    }
    value = (
        weights.kam_imbalance * parts["kam_imbalance"]
        + weights.overload * parts["overload"]
        + weights.coordination * parts["coordination"]
        + weights.management * parts["management"]
        + weights.reorganization * parts["reorganization"]
    )
    return float(value), parts
