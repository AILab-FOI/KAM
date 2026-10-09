from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .model import ScenarioTrace, Snapshot


@dataclass(frozen=True)
class ScenarioSpec:
    """Common runtime/cost parameters shared by domain-specific generators."""

    name: str
    description: str
    capacity_headroom: float
    capacity_variation: float
    remote_comm_cost: float
    internal_comm_base: float
    management_scale: float
    disturbance_every: int


SCENARIO_SPECS: Dict[str, ScenarioSpec] = {
    "warehouse": ScenarioSpec(
        "warehouse",
        "Spatial warehouse benchmark with order hotspots, robot proximity and handoff coordination.",
        1.10, 0.035, 2.2, 0.62, 0.11, 20,
    ),
    "edge": ScenarioSpec(
        "edge",
        "Edge/service benchmark with regional demand, service-call graphs and network-latency-weighted traffic.",
        1.10, 0.055, 3.4, 0.52, 0.10, 18,
    ),
    "enterprise": ScenarioSpec(
        "enterprise",
        "Project/enterprise benchmark with skill profiles, project memberships and task-dependency collaboration.",
        1.16, 0.025, 1.9, 0.70, 0.16, 24,
    ),
    "hpc": ScenarioSpec(
        "hpc",
        "HPC benchmark with migratable grid work units, stencil-like communication, hotspots and stragglers.",
        1.06, 0.045, 4.5, 0.46, 0.08, 16,
    ),
}


def build_holarchy(units: Sequence[int], fanout: int = 3) -> Tuple[Dict[int, int], Dict[int, int], int]:
    """Build a deterministic multi-level supervisor tree over leaf units."""
    leaves = list(map(int, units))
    if not leaves:
        raise ValueError("at least one unit is required")
    fanout = max(int(fanout), 2)
    root = -1
    next_manager = -2
    unit_supervisor: Dict[int, int] = {}
    supervisor_parent: Dict[int, int] = {}
    current: List[Tuple[str, int]] = [("unit", u) for u in leaves]
    while len(current) > fanout:
        parents: List[Tuple[str, int]] = []
        for start in range(0, len(current), fanout):
            group = current[start : start + fanout]
            manager = next_manager
            next_manager -= 1
            parents.append(("manager", manager))
            for kind, node in group:
                if kind == "unit":
                    unit_supervisor[node] = manager
                else:
                    supervisor_parent[node] = manager
        current = parents
    for kind, node in current:
        if kind == "unit":
            unit_supervisor[node] = root
        else:
            supervisor_parent[node] = root
    return unit_supervisor, supervisor_parent, root


def _balanced_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.arange(n, dtype=int) % k
    rng.shuffle(labels)
    return labels


def _repair_nonempty(assignment: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(assignment, dtype=int).copy()
    for u in range(k):
        if np.any(a == u):
            continue
        counts = np.bincount(a, minlength=k)
        donor = int(np.argmax(counts))
        candidates = np.flatnonzero(a == donor)
        if len(candidates) <= 1:
            continue
        a[int(rng.choice(candidates))] = u
    return a


def _scramble_assignment(latent: np.ndarray, k: int, rng: np.random.Generator, fraction: float) -> np.ndarray:
    a = np.asarray(latent, dtype=int).copy()
    count = int(round(float(fraction) * len(a)))
    if count > 0:
        for i in rng.choice(len(a), count, replace=False):
            choices = [u for u in range(k) if u != int(a[i])]
            a[i] = int(rng.choice(choices))
    return _repair_nonempty(a, k, rng)


def _grid_centers(k: int) -> np.ndarray:
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = np.array([(x, y) for y in ys for x in xs], dtype=float)
    return pts[:k]


def _nearest_labels(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


def _sparsify_topk(mat: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Keep only locally strongest communication links, symmetrically.

    Real organizational and technical communication graphs are usually sparse;
    retaining a bounded strong-neighbour set also prevents numerical dust from
    being interpreted as a KAM interaction with every organizational unit.
    """
    A = np.asarray(mat, dtype=float).copy()
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    q = max(1, min(int(max_neighbors), max(n - 1, 1)))
    keep = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = A[i]
        positive = np.flatnonzero(row > 0)
        if len(positive) <= q:
            keep[i, positive] = True
        else:
            idx = positive[np.argpartition(row[positive], -q)[-q:]]
            keep[i, idx] = True
    keep = keep | keep.T
    A[~keep] = 0.0
    return A


def _common_trace_fields(
    *,
    name: str,
    seed: int,
    initial: np.ndarray,
    cap_shares: Dict[int, float],
    q_impact: np.ndarray,
    c_impact: np.ndarray,
    ks: np.ndarray,
    migration_cost: np.ndarray,
    snapshots: List[Snapshot],
    supervisor_fanout: int,
    metadata: Dict[str, object],
) -> ScenarioTrace:
    spec = SCENARIO_SPECS[name]
    unit_supervisor, supervisor_parent, root = build_holarchy(
        sorted(cap_shares), supervisor_fanout
    )
    md: Dict[str, object] = {
        "description": spec.description,
        "generator_status": "publication_pilot_domain_specific",
        "domain_model_version": "P1.0",
        "kq": 1.65,
        "kc": 1.08,
    }
    md.update(metadata)
    return ScenarioTrace(
        name=name,
        seed=int(seed),
        q_impact=np.asarray(q_impact, dtype=bool),
        c_impact=np.asarray(c_impact, dtype=bool),
        ks=np.asarray(ks, dtype=float),
        migration_cost=np.asarray(migration_cost, dtype=float),
        initial_assignment=np.asarray(initial, dtype=int),
        initial_capacity_shares={int(k): float(v) for k, v in cap_shares.items()},
        initial_unit_supervisor=unit_supervisor,
        initial_supervisor_parent=supervisor_parent,
        root_supervisor=root,
        snapshots=snapshots,
        remote_comm_cost=spec.remote_comm_cost,
        internal_comm_base=spec.internal_comm_base,
        management_scale=spec.management_scale,
        metadata=md,
    )


def _warehouse_trace(seed: int, steps: int, n: int, k: int, fanout: int) -> ScenarioTrace:
    rng = np.random.default_rng(seed)
    spec = SCENARIO_SPECS["warehouse"]
    centers = _grid_centers(k)
    # Persistent robot operating positions; deliberately noisy around zone centers.
    home = centers[np.arange(n) % k] + rng.normal(0.0, 0.10, size=(n, 2))
    home = np.clip(home, 0.0, 1.0)
    latent = _nearest_labels(home, centers)
    initial = _scramble_assignment(latent, k, rng, 0.34)

    speed = rng.lognormal(0.0, 0.12, n)
    handling = rng.lognormal(0.0, 0.18, n)
    q_impact = speed > np.quantile(speed, 0.25)
    c_impact = np.ones(n, dtype=bool)
    ks = 1.0 + 0.15 * (handling - handling.min()) / max(np.ptp(handling), 1e-12)
    migration_cost = 0.18 + 0.22 * rng.lognormal(-0.1, 0.35, n)

    cap_raw = speed.copy()
    # Organizational teams start with capacity proportional to their robots' speed.
    cap_units = np.array([cap_raw[initial == u].sum() for u in range(k)], dtype=float)
    cap_units /= cap_units.sum()
    cap_shares = {u: float(cap_units[u]) for u in range(k)}

    hotspot_count = max(2, min(4, k // 2))
    hotspots = rng.uniform(0.12, 0.88, size=(hotspot_count, 2))
    hotspot_strength = rng.uniform(0.8, 1.4, hotspot_count)
    snapshots: List[Snapshot] = []
    phase_len = max(12, steps // 5)

    for t in range(steps):
        phase = t // phase_len
        disturbance = ""
        if t > 0 and t % phase_len == 0:
            # Moving demand pattern is a genuine spatial change, not label rewiring.
            hotspots = np.clip(
                hotspots + rng.normal(0.0, 0.20, size=hotspots.shape), 0.08, 0.92
            )
            hotspot_strength = rng.uniform(0.8, 1.5, hotspot_count)
            disturbance = "demand_map_shift"

        d = np.linalg.norm(home[:, None, :] - hotspots[None, :, :], axis=2)
        attraction = np.exp(-(d**2) / (2 * 0.16**2))
        work = 0.24 + (attraction * hotspot_strength[None, :]).sum(axis=1)
        work *= handling
        work *= rng.lognormal(0.0, 0.10, n)

        if t % spec.disturbance_every == spec.disturbance_every // 2:
            h = int(rng.integers(0, hotspot_count))
            work *= 1.0 + 1.5 * attraction[:, h]
            disturbance = (disturbance + "+" if disturbance else "") + "urgent_order_burst"

        # Communication comes from proximity/collision awareness plus overlapping orders.
        pdist = np.linalg.norm(home[:, None, :] - home[None, :, :], axis=2)
        proximity = np.exp(-pdist / 0.12)
        overlap = attraction @ attraction.T
        comm = 0.34 * proximity + 0.78 * overlap
        comm *= np.sqrt(np.outer(work / max(work.mean(), 1e-12), work / max(work.mean(), 1e-12)))
        comm *= rng.lognormal(0.0, 0.09, size=(n, n))
        comm = 0.5 * (comm + comm.T)
        comm[pdist > 0.48] *= 0.12
        np.fill_diagonal(comm, 0.0)
        comm = _sparsify_topk(comm, max(8, min(14, n // 4)))

        capacity_noise = max(0.78, 1.0 + rng.normal(0.0, spec.capacity_variation))
        # Aisle disruption is represented as temporary effective throughput loss.
        if t % (2 * spec.disturbance_every) == spec.disturbance_every:
            capacity_noise *= 0.90
            disturbance = (disturbance + "+" if disturbance else "") + "aisle_disruption"
        total_capacity = float(work.sum() * spec.capacity_headroom * capacity_noise)
        snapshots.append(Snapshot(t, work.astype(float), comm.astype(float), total_capacity, phase, disturbance))

    return _common_trace_fields(
        name="warehouse", seed=seed, initial=initial, cap_shares=cap_shares,
        q_impact=q_impact, c_impact=c_impact, ks=ks,
        migration_cost=migration_cost, snapshots=snapshots,
        supervisor_fanout=fanout,
        metadata={
            "agent_positions": home.tolist(),
            "zone_centers": centers.tolist(),
            "mechanism": "2D demand hotspots + proximity/collision + shared-order coordination",
        },
    )


def _edge_trace(seed: int, steps: int, n: int, k: int, fanout: int) -> ScenarioTrace:
    rng = np.random.default_rng(seed)
    spec = SCENARIO_SPECS["edge"]
    n_regions = max(3, min(k, 6))
    n_services = max(4, min(10, int(round(np.sqrt(n))) + 2))
    region_xy = _grid_centers(n_regions)
    agent_region = np.arange(n) % n_regions
    rng.shuffle(agent_region)
    service_type = np.arange(n) % n_services
    rng.shuffle(service_type)

    # Acyclic-ish service-call graph with a few skip-level edges.
    call = np.zeros((n_services, n_services), dtype=float)
    for s in range(n_services - 1):
        call[s, s + 1] = rng.uniform(0.55, 1.15)
        if s + 2 < n_services and rng.random() < 0.55:
            call[s, s + 2] = rng.uniform(0.12, 0.45)
    for _ in range(max(1, n_services // 3)):
        a = int(rng.integers(0, n_services - 1))
        b = int(rng.integers(a + 1, n_services))
        call[a, b] += rng.uniform(0.08, 0.28)

    latent = (agent_region * k // n_regions).astype(int)
    latent = np.clip(latent, 0, k - 1)
    initial = _scramble_assignment(latent, k, rng, 0.30)
    compute_factor = rng.lognormal(0.0, 0.28, n_services)
    state_size = rng.lognormal(-0.2, 0.48, n)
    q_impact = np.ones(n, dtype=bool)
    c_impact = compute_factor[service_type] > np.quantile(compute_factor, 0.25)
    ks = 1.0 + 0.15 * (compute_factor[service_type] - compute_factor.min()) / max(np.ptp(compute_factor), 1e-12)
    migration_cost = 0.30 + 0.55 * state_size

    edge_capacity = rng.lognormal(0.0, 0.20, k)
    edge_capacity /= edge_capacity.sum()
    cap_shares = {u: float(edge_capacity[u]) for u in range(k)}

    regional = rng.lognormal(0.0, 0.22, n_regions)
    snapshots: List[Snapshot] = []
    phase_len = max(15, steps // 5)
    for t in range(steps):
        phase = t // phase_len
        disturbance = ""
        innovation = rng.lognormal(0.0, 0.30, n_regions)
        regional = 0.78 * regional + 0.22 * innovation
        if t % spec.disturbance_every == spec.disturbance_every // 2:
            r = int(rng.integers(0, n_regions))
            regional = regional.copy()
            regional[r] *= 2.8
            disturbance = f"regional_burst_r{r}"
        if t > 0 and t % phase_len == 0:
            # Change one service dependency to emulate application evolution.
            a = int(rng.integers(0, n_services - 1))
            b = int(rng.integers(a + 1, n_services))
            call[a, b] += rng.uniform(0.15, 0.55)
            disturbance = (disturbance + "+" if disturbance else "") + "service_graph_change"

        replicas = np.maximum(np.bincount(service_type, minlength=n_services), 1)
        service_pop = 0.65 + 0.70 * (call.sum(axis=0) + call.sum(axis=1))
        work = regional[agent_region] * service_pop[service_type] * compute_factor[service_type]
        work /= np.sqrt(replicas[service_type])
        work *= rng.lognormal(0.0, 0.11, n)

        region_d = np.linalg.norm(region_xy[:, None, :] - region_xy[None, :, :], axis=2)
        latency = 1.0 + 4.0 * region_d
        comm = np.zeros((n, n), dtype=float)
        for i in range(n):
            si, ri = int(service_type[i]), int(agent_region[i])
            for j in range(i + 1, n):
                sj, rj = int(service_type[j]), int(agent_region[j])
                volume = call[si, sj] + call[sj, si]
                if volume <= 0:
                    continue
                demand = np.sqrt(regional[ri] * regional[rj])
                # Store latency-weighted traffic so network locality matters directly.
                w = volume * demand * latency[ri, rj] * rng.lognormal(0.0, 0.08)
                comm[i, j] = comm[j, i] = float(w)
        comm = _sparsify_topk(comm, max(8, min(14, n // 4)))

        capacity_noise = max(0.72, 1.0 + rng.normal(0.0, spec.capacity_variation))
        if t % (2 * spec.disturbance_every) == spec.disturbance_every:
            capacity_noise *= 0.86
            disturbance = (disturbance + "+" if disturbance else "") + "edge_capacity_degradation"
        total_capacity = float(work.sum() * spec.capacity_headroom * capacity_noise)
        snapshots.append(Snapshot(t, work.astype(float), comm, total_capacity, phase, disturbance))

    return _common_trace_fields(
        name="edge", seed=seed, initial=initial, cap_shares=cap_shares,
        q_impact=q_impact, c_impact=c_impact, ks=ks,
        migration_cost=migration_cost, snapshots=snapshots,
        supervisor_fanout=fanout,
        metadata={
            "agent_region": agent_region.tolist(),
            "service_type": service_type.tolist(),
            "region_positions": region_xy.tolist(),
            "mechanism": "regional demand + service-call DAG + latency-weighted traffic",
        },
    )


def _enterprise_trace(seed: int, steps: int, n: int, k: int, fanout: int) -> ScenarioTrace:
    rng = np.random.default_rng(seed)
    spec = SCENARIO_SPECS["enterprise"]
    n_skills = 6
    n_projects = max(k + 2, min(2 * k, 16))
    # Employees have mixed but usually dominant skill profiles.
    skill = rng.dirichlet(np.full(n_skills, 0.75), size=n)
    dominant = np.argmax(skill, axis=1)
    latent = (dominant * k // n_skills).astype(int)
    latent = np.clip(latent, 0, k - 1)
    latent = _repair_nonempty(latent, k, rng)
    initial = _scramble_assignment(latent, k, rng, 0.28)

    requirements = rng.dirichlet(np.full(n_skills, 0.8), size=n_projects)
    complexity = rng.uniform(0.75, 1.45, n_projects)
    # Directed project dependencies, later symmetrized when inducing collaboration.
    dependency = np.zeros((n_projects, n_projects), dtype=float)
    for p in range(n_projects - 1):
        if rng.random() < 0.72:
            dependency[p, p + 1] = rng.uniform(0.25, 0.85)
        if p + 2 < n_projects and rng.random() < 0.35:
            dependency[p, p + 2] = rng.uniform(0.10, 0.40)

    q_impact = np.max(skill, axis=1) > np.quantile(np.max(skill, axis=1), 0.25)
    c_impact = np.ones(n, dtype=bool)
    skill_entropy = -(skill * np.log(np.maximum(skill, 1e-12))).sum(axis=1) / np.log(n_skills)
    ks = 1.0 + 0.15 * skill_entropy
    migration_cost = 0.32 + 0.40 * (0.5 + skill_entropy) * rng.lognormal(0.0, 0.20, n)

    headcount_capacity = np.array([len(np.flatnonzero(initial == u)) for u in range(k)], dtype=float)
    headcount_capacity *= rng.lognormal(0.0, 0.08, k)
    headcount_capacity /= headcount_capacity.sum()
    cap_shares = {u: float(headcount_capacity[u]) for u in range(k)}

    project_demand = rng.lognormal(-0.05, 0.22, n_projects)
    active = np.zeros(n_projects, dtype=float)
    active[: max(3, n_projects // 2)] = 1.0
    rng.shuffle(active)
    snapshots: List[Snapshot] = []
    phase_len = max(18, steps // 4)

    for t in range(steps):
        phase = t // phase_len
        disturbance = ""
        project_demand = 0.86 * project_demand + 0.14 * rng.lognormal(-0.05, 0.25, n_projects)
        if t > 0 and t % phase_len == 0:
            active = np.zeros(n_projects, dtype=float)
            chosen = rng.choice(n_projects, size=max(3, n_projects // 2), replace=False)
            active[chosen] = 1.0
            disturbance = "project_phase_transition"
        if t % spec.disturbance_every == spec.disturbance_every // 2:
            active_ids = np.flatnonzero(active > 0)
            p = int(rng.choice(active_ids))
            project_demand[p] *= 2.4
            disturbance = (disturbance + "+" if disturbance else "") + f"deadline_p{p}"

        match = skill @ requirements.T
        # Only sufficiently matching employees join a project, yielding overlapping teams.
        threshold = np.quantile(match, 0.62, axis=0)
        participation = np.maximum(match - threshold[None, :], 0.0)
        participation *= active[None, :]
        denom = np.maximum(participation.sum(axis=0), 1e-12)
        work = (participation * (project_demand * complexity / denom)[None, :]).sum(axis=1)
        work += 0.16 * rng.lognormal(-0.2, 0.20, n)
        work *= rng.lognormal(0.0, 0.07, n)

        # Collaboration from shared project memberships and dependency handoffs.
        weighted_membership = participation * np.sqrt(project_demand[None, :])
        shared = weighted_membership @ weighted_membership.T
        dep_flow = weighted_membership @ (dependency + dependency.T) @ weighted_membership.T
        comm = 0.82 * shared + 0.38 * dep_flow
        comm *= rng.lognormal(0.0, 0.07, size=(n, n))
        comm = 0.5 * (comm + comm.T)
        np.fill_diagonal(comm, 0.0)
        comm = _sparsify_topk(comm, max(8, min(14, n // 4)))

        capacity_noise = max(0.86, 1.0 + rng.normal(0.0, spec.capacity_variation))
        if t % (3 * spec.disturbance_every) == 2 * spec.disturbance_every:
            capacity_noise *= 0.92
            disturbance = (disturbance + "+" if disturbance else "") + "temporary_absence"
        total_capacity = float(work.sum() * spec.capacity_headroom * capacity_noise)
        snapshots.append(Snapshot(t, work.astype(float), comm.astype(float), total_capacity, phase, disturbance))

    return _common_trace_fields(
        name="enterprise", seed=seed, initial=initial, cap_shares=cap_shares,
        q_impact=q_impact, c_impact=c_impact, ks=ks,
        migration_cost=migration_cost, snapshots=snapshots,
        supervisor_fanout=fanout,
        metadata={
            "skills": skill.tolist(),
            "dominant_skill": dominant.tolist(),
            "mechanism": "skill-task matching + overlapping projects + project-dependency handoffs",
        },
    )


def _hpc_trace(seed: int, steps: int, n: int, k: int, fanout: int) -> ScenarioTrace:
    rng = np.random.default_rng(seed)
    spec = SCENARIO_SPECS["hpc"]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    coords = np.array([(i % cols, i // cols) for i in range(n)], dtype=float)
    coords[:, 0] /= max(cols - 1, 1)
    coords[:, 1] /= max(rows - 1, 1)
    # Spatial block decomposition gives a plausible but imperfect initial partition.
    unit_centers = _grid_centers(k)
    latent = _nearest_labels(coords, unit_centers)
    latent = _repair_nonempty(latent, k, rng)
    initial = _scramble_assignment(latent, k, rng, 0.27)

    base_compute = rng.lognormal(0.0, 0.25, n)
    state_size = rng.lognormal(-0.1, 0.42, n)
    q_impact = np.ones(n, dtype=bool)
    c_impact = np.ones(n, dtype=bool)
    # ks tracks local degree / algorithmic complexity, not random labels.
    ks = 1.0 + 0.15 * (base_compute - base_compute.min()) / max(np.ptp(base_compute), 1e-12)
    migration_cost = 0.38 + 0.70 * state_size

    processor_speed = rng.lognormal(0.0, 0.16, k)
    processor_speed /= processor_speed.sum()
    cap_shares = {u: float(processor_speed[u]) for u in range(k)}

    # Static stencil edges over the work-unit grid.
    base_comm = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = i % cols, i // cols
        for j in range(i + 1, n):
            xj, yj = j % cols, j // cols
            manhattan = abs(xi - xj) + abs(yi - yj)
            if manhattan == 1:
                base_comm[i, j] = base_comm[j, i] = rng.uniform(0.8, 1.25)
            elif abs(xi - xj) == 1 and abs(yi - yj) == 1 and rng.random() < 0.30:
                base_comm[i, j] = base_comm[j, i] = rng.uniform(0.12, 0.35)

    hotspot = np.array([0.25, 0.25], dtype=float)
    snapshots: List[Snapshot] = []
    phase_len = max(14, steps // 5)
    for t in range(steps):
        phase = t // phase_len
        disturbance = ""
        if t > 0 and t % phase_len == 0:
            hotspot = rng.uniform(0.10, 0.90, 2)
            disturbance = "adaptive_hotspot_move"
        distance = np.linalg.norm(coords - hotspot[None, :], axis=1)
        refinement = np.exp(-(distance**2) / (2 * 0.16**2))
        work = base_compute * (0.68 + 2.05 * refinement)
        work *= rng.lognormal(0.0, 0.09, n)
        if t % spec.disturbance_every == spec.disturbance_every // 2:
            count = max(1, n // 18)
            stragglers = rng.choice(n, size=count, replace=False)
            work[stragglers] *= 2.6
            disturbance = (disturbance + "+" if disturbance else "") + "stragglers"

        edge_hot = 1.0 + 1.4 * np.sqrt(np.outer(refinement, refinement))
        comm = base_comm * edge_hot
        comm *= rng.lognormal(0.0, 0.06, size=(n, n))
        comm = 0.5 * (comm + comm.T)
        np.fill_diagonal(comm, 0.0)
        comm = _sparsify_topk(comm, max(6, min(10, n // 5)))

        capacity_noise = max(0.72, 1.0 + rng.normal(0.0, spec.capacity_variation))
        if t % (3 * spec.disturbance_every) == 2 * spec.disturbance_every:
            capacity_noise *= 0.83
            disturbance = (disturbance + "+" if disturbance else "") + "resource_slowdown"
        total_capacity = float(work.sum() * spec.capacity_headroom * capacity_noise)
        snapshots.append(Snapshot(t, work.astype(float), comm.astype(float), total_capacity, phase, disturbance))

    return _common_trace_fields(
        name="hpc", seed=seed, initial=initial, cap_shares=cap_shares,
        q_impact=q_impact, c_impact=c_impact, ks=ks,
        migration_cost=migration_cost, snapshots=snapshots,
        supervisor_fanout=fanout,
        metadata={
            "task_coordinates": coords.tolist(),
            "mechanism": "2D overdecomposed stencil/AMR-like graph + moving compute hotspot + stragglers",
        },
    )


def generate_trace(
    name: str,
    seed: int,
    steps: int,
    n_agents: int,
    n_units: int,
    supervisor_fanout: int = 3,
) -> ScenarioTrace:
    """Generate one of the four domain-specific publication-pilot traces."""
    if name not in SCENARIO_SPECS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIO_SPECS)}")
    if n_agents < n_units or n_units < 2:
        raise ValueError("require n_agents >= n_units >= 2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    fn = {
        "warehouse": _warehouse_trace,
        "edge": _edge_trace,
        "enterprise": _enterprise_trace,
        "hpc": _hpc_trace,
    }[name]
    return fn(int(seed), int(steps), int(n_agents), int(n_units), int(supervisor_fanout))
