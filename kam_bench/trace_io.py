from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .model import ScenarioTrace, Snapshot
from .scenarios import build_holarchy


def trace_from_arrays(
    *,
    name: str,
    seed: int,
    work: np.ndarray,
    communication: np.ndarray,
    total_capacity: Sequence[float],
    initial_assignment: np.ndarray,
    initial_capacity_shares: Optional[Dict[int, float]] = None,
    q_impact: Optional[np.ndarray] = None,
    c_impact: Optional[np.ndarray] = None,
    ks: Optional[np.ndarray] = None,
    migration_cost: Optional[np.ndarray] = None,
    phase: Optional[Sequence[int]] = None,
    disturbance: Optional[Sequence[str]] = None,
    remote_comm_cost: float = 2.5,
    internal_comm_base: float = 0.6,
    management_scale: float = 0.1,
    supervisor_fanout: int = 3,
    initial_unit_supervisor: Optional[Dict[int, int]] = None,
    initial_supervisor_parent: Optional[Dict[int, int]] = None,
    root_supervisor: int = -1,
    metadata: Optional[Dict[str, object]] = None,
) -> ScenarioTrace:
    """Create a validated ScenarioTrace from measured/external arrays.

    Shapes:
      work:           (T, N), nonnegative arrivals/work demand
      communication:  (T, N, N), nonnegative weighted interactions
      total_capacity: (T,), strictly positive
      assignment:     (N,)
    """
    work = np.asarray(work, dtype=float)
    communication = np.asarray(communication, dtype=float)
    total_capacity = np.asarray(total_capacity, dtype=float)
    initial_assignment = np.asarray(initial_assignment, dtype=int)
    if work.ndim != 2:
        raise ValueError("work must have shape (T,N)")
    T, N = work.shape
    if T <= 0 or N <= 0:
        raise ValueError("work must be non-empty")
    if communication.shape != (T, N, N):
        raise ValueError("communication must have shape (T,N,N)")
    if total_capacity.shape != (T,):
        raise ValueError("total_capacity must have shape (T,)")
    if initial_assignment.shape != (N,):
        raise ValueError("initial_assignment must have shape (N,)")
    if np.any(~np.isfinite(work)) or np.any(work < 0):
        raise ValueError("work must be finite and nonnegative")
    if np.any(~np.isfinite(communication)) or np.any(communication < 0):
        raise ValueError("communication must be finite and nonnegative")
    if np.any(~np.isfinite(total_capacity)) or np.any(total_capacity <= 0):
        raise ValueError("total_capacity must be finite and positive")

    units = sorted(int(u) for u in np.unique(initial_assignment))
    if len(units) < 2:
        raise ValueError("at least two initial units are required")

    if initial_capacity_shares is None:
        initial_capacity_shares = {u: 1.0 / len(units) for u in units}
    else:
        initial_capacity_shares = {
            int(k): float(v) for k, v in initial_capacity_shares.items()
        }
        if set(initial_capacity_shares) != set(units):
            raise ValueError("capacity-share keys must match assignment units")
        if any(v <= 0 or not np.isfinite(v) for v in initial_capacity_shares.values()):
            raise ValueError("capacity shares must be finite and positive")
        total = sum(initial_capacity_shares.values())
        initial_capacity_shares = {u: v / total for u, v in initial_capacity_shares.items()}

    q_impact = np.ones(N, dtype=bool) if q_impact is None else np.asarray(q_impact, dtype=bool)
    c_impact = np.ones(N, dtype=bool) if c_impact is None else np.asarray(c_impact, dtype=bool)
    ks = np.ones(N, dtype=float) if ks is None else np.asarray(ks, dtype=float)
    migration_cost = (
        np.ones(N, dtype=float)
        if migration_cost is None
        else np.asarray(migration_cost, dtype=float)
    )
    for name_, arr in [
        ("q_impact", q_impact),
        ("c_impact", c_impact),
        ("ks", ks),
        ("migration_cost", migration_cost),
    ]:
        if arr.shape != (N,):
            raise ValueError(f"{name_} must have shape (N,)")
    if np.any(~np.isfinite(ks)) or np.any(ks <= 0):
        raise ValueError("ks must be finite and positive")
    if np.any(~np.isfinite(migration_cost)) or np.any(migration_cost < 0):
        raise ValueError("migration_cost must be finite and nonnegative")

    phase = [0] * T if phase is None else list(map(int, phase))
    disturbance = [""] * T if disturbance is None else list(map(str, disturbance))
    if len(phase) != T or len(disturbance) != T:
        raise ValueError("phase/disturbance length must equal T")

    if initial_unit_supervisor is None:
        unit_sup, sup_parent, root = build_holarchy(units, supervisor_fanout)
    else:
        unit_sup = {int(k): int(v) for k, v in initial_unit_supervisor.items()}
        if set(unit_sup) != set(units):
            raise ValueError("unit supervisor keys must match assignment units")
        sup_parent = {
            int(k): int(v) for k, v in (initial_supervisor_parent or {}).items()
        }
        root = int(root_supervisor)

    snapshots = [
        Snapshot(
            step=t,
            work=work[t].copy(),
            communication=communication[t].copy(),
            total_capacity=float(total_capacity[t]),
            phase=int(phase[t]),
            disturbance=str(disturbance[t]),
        )
        for t in range(T)
    ]
    md = {"kq": 1.65, "kc": 1.08, "source": "external"}
    if metadata:
        md.update(metadata)
    return ScenarioTrace(
        name=name,
        seed=int(seed),
        q_impact=q_impact,
        c_impact=c_impact,
        ks=ks,
        migration_cost=migration_cost,
        initial_assignment=initial_assignment,
        initial_capacity_shares=initial_capacity_shares,
        initial_unit_supervisor=unit_sup,
        initial_supervisor_parent=sup_parent,
        root_supervisor=root,
        snapshots=snapshots,
        remote_comm_cost=float(remote_comm_cost),
        internal_comm_base=float(internal_comm_base),
        management_scale=float(management_scale),
        metadata=md,
    )


def save_trace_npz(trace: ScenarioTrace, path: Path) -> None:
    path = Path(path)
    work = np.stack([s.work for s in trace.snapshots])
    communication = np.stack([s.communication for s in trace.snapshots])
    total_capacity = np.array([s.total_capacity for s in trace.snapshots], dtype=float)
    phase = np.array([s.phase for s in trace.snapshots], dtype=int)
    disturbance = np.array([s.disturbance for s in trace.snapshots], dtype=object)
    units = np.array(sorted(trace.initial_capacity_shares), dtype=int)
    shares = np.array([trace.initial_capacity_shares[int(u)] for u in units], dtype=float)
    unit_sup = np.array([trace.initial_unit_supervisor[int(u)] for u in units], dtype=int)
    sup_nodes = np.array(sorted(trace.initial_supervisor_parent), dtype=int)
    sup_parents = np.array(
        [trace.initial_supervisor_parent[int(s)] for s in sup_nodes], dtype=int
    )
    np.savez_compressed(
        path,
        name=np.array(trace.name),
        seed=np.array(trace.seed),
        work=work,
        communication=communication,
        total_capacity=total_capacity,
        phase=phase,
        disturbance=disturbance,
        initial_assignment=trace.initial_assignment,
        capacity_units=units,
        capacity_shares=shares,
        unit_supervisor=unit_sup,
        supervisor_nodes=sup_nodes,
        supervisor_parents=sup_parents,
        root_supervisor=np.array(trace.root_supervisor),
        q_impact=trace.q_impact,
        c_impact=trace.c_impact,
        ks=trace.ks,
        migration_cost=trace.migration_cost,
        remote_comm_cost=np.array(trace.remote_comm_cost),
        internal_comm_base=np.array(trace.internal_comm_base),
        management_scale=np.array(trace.management_scale),
        metadata_json=np.array(json.dumps(trace.metadata, default=str)),
    )


def load_trace_npz(path: Path, supervisor_fanout: int = 3) -> ScenarioTrace:
    path = Path(path)
    with np.load(path, allow_pickle=True) as d:
        shares = {
            int(u): float(v) for u, v in zip(d["capacity_units"], d["capacity_shares"])
        }
        unit_sup = None
        sup_parent = None
        root = -1
        if "unit_supervisor" in d:
            unit_sup = {
                int(u): int(s) for u, s in zip(d["capacity_units"], d["unit_supervisor"])
            }
            sup_parent = {
                int(s): int(p)
                for s, p in zip(d["supervisor_nodes"], d["supervisor_parents"])
            }
            root = int(d["root_supervisor"].item())
        metadata = {}
        if "metadata_json" in d:
            try:
                metadata = json.loads(str(d["metadata_json"].item()))
            except Exception:
                metadata = {}
        return trace_from_arrays(
            name=str(d["name"].item()),
            seed=int(d["seed"].item()),
            work=d["work"],
            communication=d["communication"],
            total_capacity=d["total_capacity"],
            initial_assignment=d["initial_assignment"],
            initial_capacity_shares=shares,
            q_impact=d["q_impact"],
            c_impact=d["c_impact"],
            ks=d["ks"],
            migration_cost=d["migration_cost"],
            phase=d["phase"],
            disturbance=d["disturbance"].tolist(),
            remote_comm_cost=float(d["remote_comm_cost"].item()),
            internal_comm_base=float(d["internal_comm_base"].item()),
            management_scale=float(d["management_scale"].item()),
            supervisor_fanout=supervisor_fanout,
            initial_unit_supervisor=unit_sup,
            initial_supervisor_parent=sup_parent,
            root_supervisor=root,
            metadata=metadata,
        )
