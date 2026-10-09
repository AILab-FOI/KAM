from __future__ import annotations

import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import networkx as nx
except Exception:  # optional at import time; required for louvain_partition
    nx = None

from .config import AlgorithmConfig, ObjectiveWeights
from .external_hpc import run_external_partitioner
from .metrics import (
    StateMetrics,
    objective_value,
    state_metrics,
    symmetric_communication,
)
from .model import AlgorithmResult, EPS, Operation, Organization, ScenarioTrace, Snapshot


def spectral_bisection(
    members: np.ndarray, communication: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    members = np.asarray(members, dtype=int)
    if len(members) < 4:
        return None
    A = communication[np.ix_(members, members)].astype(float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    if float(A.sum()) <= EPS:
        return None
    degree = A.sum(axis=1)
    inv = np.zeros_like(degree)
    mask = degree > EPS
    inv[mask] = 1.0 / np.sqrt(degree[mask])
    L = np.eye(len(members)) - inv[:, None] * A * inv[None, :]
    _, vecs = np.linalg.eigh(L)
    if vecs.shape[1] < 2:
        return None
    f = vecs[:, 1]
    cut = float(np.median(f))
    left_mask = f <= cut
    if left_mask.all() or (~left_mask).all():
        order = np.argsort(f)
        half = len(order) // 2
        left_mask = np.zeros(len(members), dtype=bool)
        left_mask[order[:half]] = True
    left, right = members[left_mask], members[~left_mask]
    if len(left) == 0 or len(right) == 0:
        return None
    return left, right


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 100) -> np.ndarray:
    n = len(X)
    if k <= 1:
        return np.zeros(n, dtype=int)
    if k > n:
        raise ValueError("k cannot exceed sample count")
    centers = [int(rng.integers(0, n))]
    while len(centers) < k:
        d2 = np.min(
            np.stack([np.sum((X - X[c]) ** 2, axis=1) for c in centers], axis=1), axis=1
        )
        d2[np.array(centers, dtype=int)] = 0.0
        if float(d2.sum()) <= EPS:
            candidates = [i for i in range(n) if i not in centers]
            centers.append(int(rng.choice(candidates)))
        else:
            centers.append(int(rng.choice(n, p=d2 / d2.sum())))
    centroids = X[np.array(centers)].copy()
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dist = np.stack([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(dist, axis=1)
        for cluster in range(k):
            if np.any(new_labels == cluster):
                continue
            counts = np.bincount(new_labels, minlength=k)
            donor = int(np.argmax(counts))
            donor_idx = np.flatnonzero(new_labels == donor)
            chosen = int(donor_idx[int(np.argmax(dist[donor_idx, donor]))])
            new_labels[chosen] = cluster
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster in range(k):
            centroids[cluster] = X[labels == cluster].mean(axis=0)
    return labels


def spectral_kway_partition(
    communication: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    A = np.asarray(communication, dtype=float).copy()
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    if float(A.sum()) <= EPS:
        labels = np.arange(n) % k
        rng.shuffle(labels)
        return labels.astype(int)
    degree = A.sum(axis=1)
    inv = np.zeros_like(degree)
    mask = degree > EPS
    inv[mask] = 1.0 / np.sqrt(degree[mask])
    L = np.eye(n) - inv[:, None] * A * inv[None, :]
    _, vecs = np.linalg.eigh(L)
    X = vecs[:, :k]
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(row_norm, EPS)
    return _kmeans(X, k, rng)




def _normalize_communities_to_k(
    communities: Sequence[Sequence[int]], communication: np.ndarray, k: int
) -> List[List[int]]:
    """Deterministically adjust a community cover to exactly k nonempty groups.

    This helper is used only by the non-KAM community baseline. If Louvain
    returns too few groups, the largest splittable group is bisected
    spectrally. If it returns too many, the smallest group is merged into the
    group with which it has the largest communication volume.
    """
    groups = [sorted(map(int, c)) for c in communities if len(c)]
    A = symmetric_communication(Snapshot(0, np.ones(len(communication)), communication, 1.0, 0))
    while len(groups) < k:
        candidates = sorted(range(len(groups)), key=lambda z: len(groups[z]), reverse=True)
        split_done = False
        for idx in candidates:
            members = np.asarray(groups[idx], dtype=int)
            if len(members) < 2:
                continue
            if len(members) >= 4:
                part = spectral_bisection(members, A)
            else:
                half = len(members) // 2
                part = (members[:half], members[half:])
            if part is None or len(part[0]) == 0 or len(part[1]) == 0:
                continue
            groups.pop(idx)
            groups.extend([sorted(part[0].tolist()), sorted(part[1].tolist())])
            split_done = True
            break
        if not split_done:
            break
    while len(groups) > k:
        idx = min(range(len(groups)), key=lambda z: (len(groups[z]), z))
        small = groups.pop(idx)
        best_j = None
        best_w = -1.0
        for j, g in enumerate(groups):
            w = float(A[np.ix_(np.asarray(small, int), np.asarray(g, int))].sum())
            if w > best_w:
                best_w, best_j = w, j
        if best_j is None:
            best_j = 0
        groups[best_j] = sorted(groups[best_j] + small)
    return groups


def louvain_kway_partition(
    communication: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Louvain community partition adjusted to the current number of units.

    This is an interim pilot structural baseline, not Leiden. The confirmatory
    protocol still requires a validated Leiden implementation or external
    assignment trace before the final paper runs.
    """
    A = np.asarray(communication, dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    if nx is None:
        raise RuntimeError("networkx is required for louvain_partition")
    if float(A.sum()) <= EPS:
        labels = np.arange(n) % k
        rng.shuffle(labels)
        return labels.astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.triu_indices(n, 1)
    for i, j in zip(rows.tolist(), cols.tolist()):
        w = float(A[i, j])
        if w > 0:
            G.add_edge(i, j, weight=w)
    seed = int(rng.integers(0, 2**31 - 1))
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    groups = _normalize_communities_to_k(communities, A, k)
    if len(groups) != k or any(len(g) == 0 for g in groups):
        return spectral_kway_partition(A, k, rng)
    labels = np.empty(n, dtype=int)
    for c, group in enumerate(groups):
        labels[np.asarray(group, dtype=int)] = c
    return labels



def _labels_to_communities(labels: np.ndarray) -> List[List[int]]:
    labels = np.asarray(labels, dtype=int)
    groups: List[List[int]] = []
    for lab in sorted(np.unique(labels).tolist()):
        groups.append(np.flatnonzero(labels == int(lab)).astype(int).tolist())
    return groups


def _contiguous_labels(assignment: np.ndarray) -> np.ndarray:
    values = sorted(int(x) for x in np.unique(assignment))
    mapping = {u: i for i, u in enumerate(values)}
    return np.array([mapping[int(x)] for x in assignment.tolist()], dtype=int)


def leiden_kway_partition(
    communication: np.ndarray,
    k: int,
    initial_assignment: np.ndarray,
    rng: np.random.Generator,
    *,
    backend: str = "igraph",
    resolution: float = 1.0,
    n_iterations: int = 4,
) -> Tuple[np.ndarray, str]:
    """Run an established Leiden implementation and normalize to exactly ``k`` groups.

    Confirmatory runs should use ``backend='igraph'``.  The function first tries
    python-igraph's ``Graph.community_leiden``.  ``backend='networkx'`` is
    supported for environments with a NetworkX backend that implements Leiden;
    NetworkX itself exposes only the dispatch API.  No Louvain or spectral
    substitute is silently used when a requested Leiden backend is absent.
    """
    A = np.asarray(communication, dtype=float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = len(A)
    if k <= 0 or k > n:
        raise ValueError("invalid number of Leiden communities")
    if float(A.sum()) <= EPS:
        # There is no community information to optimize. Preserve the current
        # organization rather than substituting a different partitioning method.
        labels = _contiguous_labels(np.asarray(initial_assignment, dtype=int))
        groups = _normalize_communities_to_k(_labels_to_communities(labels), A, k)
        out = np.empty(n, dtype=int)
        for c, g in enumerate(groups):
            out[np.asarray(g, dtype=int)] = c
        return out, "no_edges_preserve_current"

    requested = str(backend).lower().strip()
    seed = int(rng.integers(0, 2**31 - 1))
    initial = _contiguous_labels(np.asarray(initial_assignment, dtype=int)).tolist()
    errors: List[str] = []

    if requested in {"igraph", "auto"}:
        try:
            import igraph as ig  # optional dependency, established implementation

            rows, cols = np.triu_indices(n, 1)
            edges: List[Tuple[int, int]] = []
            weights: List[float] = []
            for i, j in zip(rows.tolist(), cols.tolist()):
                w = float(A[i, j])
                if w > 0:
                    edges.append((int(i), int(j)))
                    weights.append(w)
            graph = ig.Graph(n=n, edges=edges, directed=False)
            # igraph's Leiden uses its global RNG. A local Random instance gives
            # deterministic runs without altering Python's module-global RNG.
            if hasattr(ig, "set_random_number_generator"):
                ig.set_random_number_generator(random.Random(seed))
            clustering = graph.community_leiden(
                objective_function="modularity",
                weights=weights,
                resolution=float(resolution),
                initial_membership=initial,
                n_iterations=max(1, int(n_iterations)),
            )
            communities = [list(map(int, group)) for group in clustering]
            groups = _normalize_communities_to_k(communities, A, k)
            labels = np.empty(n, dtype=int)
            for c, group in enumerate(groups):
                labels[np.asarray(group, dtype=int)] = c
            return labels, f"igraph-{getattr(ig, '__version__', 'unknown')}"
        except Exception as exc:
            errors.append(f"igraph: {type(exc).__name__}: {exc}")
            if requested == "igraph":
                raise RuntimeError(
                    "Dynamic Leiden requires python-igraph for confirmatory runs. "
                    "Install the optional Leiden dependency before running this method. "
                    + errors[-1]
                ) from exc

    if requested in {"networkx", "auto"}:
        if nx is None:
            errors.append("networkx: unavailable")
        else:
            try:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                rows, cols = np.triu_indices(n, 1)
                for i, j in zip(rows.tolist(), cols.tolist()):
                    w = float(A[i, j])
                    if w > 0:
                        G.add_edge(int(i), int(j), weight=w)
                communities = nx.algorithms.community.leiden_communities(
                    G,
                    weight="weight",
                    resolution=float(resolution),
                    seed=seed,
                )
                groups = _normalize_communities_to_k(
                    [sorted(map(int, c)) for c in communities], A, k
                )
                labels = np.empty(n, dtype=int)
                for c, group in enumerate(groups):
                    labels[np.asarray(group, dtype=int)] = c
                return labels, f"networkx-backend-{nx.__version__}"
            except Exception as exc:
                errors.append(f"networkx: {type(exc).__name__}: {exc}")
                if requested == "networkx":
                    raise RuntimeError(
                        "NetworkX exposes Leiden through a backend dispatch; no Leiden-capable "
                        "backend is available in this environment. " + errors[-1]
                    ) from exc

    raise RuntimeError(
        "No established Leiden backend is available; refusing to substitute Louvain/spectral. "
        + " | ".join(errors)
    )

def _domain_penalty(trace: ScenarioTrace, snapshot: Snapshot, org: Organization) -> float:
    """Non-KAM domain knowledge term used by the domain-specific pilot baseline."""
    name = trace.name
    md = trace.metadata
    n = trace.n_agents
    if name == "warehouse":
        pos = np.asarray(md.get("agent_positions", []), dtype=float)
        if pos.shape != (n, 2):
            return 0.0
        total = 0.0
        for u in org.units:
            members = org.members(u)
            if len(members) <= 1:
                continue
            centroid = pos[members].mean(axis=0)
            total += float(np.linalg.norm(pos[members] - centroid, axis=1).sum())
        return total / max(n * np.sqrt(2.0), EPS)
    if name == "edge":
        region = np.asarray(md.get("agent_region", []), dtype=int)
        if region.shape != (n,):
            return 0.0
        # Gini impurity of region membership within each resource pool.
        total = 0.0
        for u in org.units:
            members = org.members(u)
            counts = np.bincount(region[members])
            p = counts[counts > 0] / max(len(members), 1)
            total += len(members) * float(1.0 - np.sum(p**2))
        return total / max(n, 1)
    if name == "enterprise":
        dominant = np.asarray(md.get("dominant_skill", []), dtype=int)
        if dominant.shape != (n,):
            return 0.0
        # Functional-specialization baseline: low within-unit skill impurity.
        total = 0.0
        for u in org.units:
            members = org.members(u)
            counts = np.bincount(dominant[members])
            p = counts[counts > 0] / max(len(members), 1)
            total += len(members) * float(1.0 - np.sum(p**2))
        return total / max(n, 1)
    if name == "hpc":
        coords = np.asarray(md.get("task_coordinates", []), dtype=float)
        if coords.shape != (n, 2):
            return 0.0
        total = 0.0
        for u in org.units:
            members = org.members(u)
            if len(members) <= 1:
                continue
            centroid = coords[members].mean(axis=0)
            total += float(np.linalg.norm(coords[members] - centroid, axis=1).sum())
        return total / max(n * np.sqrt(2.0), EPS)
    return 0.0

def map_clusters_to_units(
    cluster_labels: np.ndarray, old_assignment: np.ndarray, units: Sequence[int]
) -> np.ndarray:
    units = list(map(int, units))
    k = len(units)
    overlap = np.zeros((k, k), dtype=float)
    for c in range(k):
        for j, u in enumerate(units):
            overlap[c, j] = np.sum((cluster_labels == c) & (old_assignment == u))
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(r): units[int(c)] for r, c in zip(rows, cols)}
    return np.array([mapping[int(c)] for c in cluster_labels], dtype=int)


def split_capacity_fraction(
    cfg: AlgorithmConfig, snapshot: Snapshot, left: np.ndarray, right: np.ndarray
) -> float:
    if cfg.split_capacity_rule == "work":
        total = float(snapshot.work[np.r_[left, right]].sum())
        if total <= EPS:
            frac = float(len(right) / max(len(left) + len(right), 1))
        else:
            frac = float(snapshot.work[right].sum() / total)
    elif cfg.split_capacity_rule == "size":
        frac = float(len(right) / max(len(left) + len(right), 1))
    else:
        raise ValueError(f"unknown split_capacity_rule: {cfg.split_capacity_rule}")
    return float(np.clip(frac, 0.05, 0.95))


def operation_cost(trace: ScenarioTrace, operation: Operation, cfg: AlgorithmConfig) -> float:
    p = operation.payload
    if operation.operator == "move":
        return float(trace.migration_cost[int(p["agent"])])
    if operation.operator == "merge":
        return float(cfg.merge_structural_cost)
    if operation.operator == "split":
        return float(cfg.split_structural_cost)
    if operation.operator == "repartition":
        moved = np.asarray(p.get("moved_agents", []), dtype=int)
        return float(trace.migration_cost[moved].sum()) if len(moved) else 0.0
    return 0.0


def apply_operation(org: Organization, operation: Operation, cfg: AlgorithmConfig) -> int:
    p = operation.payload
    if operation.operator == "move":
        org.move(int(p["agent"]), int(p["target"]), min_unit_size=cfg.min_unit_size)
        return 1
    if operation.operator == "merge":
        org.merge(int(p["unit_a"]), int(p["unit_b"]))
        return 0
    if operation.operator == "split":
        org.split(
            int(p["unit"]),
            p["left"],
            p["right"],
            float(p["right_capacity_fraction"]),
        )
        return 0
    if operation.operator == "repartition":
        old = org.assignment.copy()
        org.replace_assignment(np.asarray(p["assignment"], dtype=int))
        return int(np.count_nonzero(old != org.assignment))
    raise ValueError(f"unknown operation {operation.operator}")


def candidate_moves(org: Organization, cfg: AlgorithmConfig) -> Iterable[Operation]:
    if not cfg.allow_move or len(org.units) <= 1:
        return []
    out: List[Operation] = []
    for agent in range(len(org.assignment)):
        source = int(org.assignment[agent])
        if org.size(source) <= cfg.min_unit_size:
            continue
        for target in org.units:
            if target == source:
                continue
            out.append(
                Operation(
                    "move",
                    {"agent": int(agent), "source": source, "target": int(target)},
                    authority=org.lca_supervisor(source, target),
                )
            )
    return out


def candidate_merges(org: Organization, cfg: AlgorithmConfig) -> Iterable[Operation]:
    if not cfg.allow_merge or len(org.units) <= 2:
        return []
    units = org.units
    return [
        Operation(
            "merge",
            {"unit_a": int(a), "unit_b": int(b)},
            authority=org.lca_supervisor(a, b),
        )
        for i, a in enumerate(units)
        for b in units[i + 1 :]
    ]


def candidate_splits(
    org: Organization, snapshot: Snapshot, cfg: AlgorithmConfig
) -> Iterable[Operation]:
    if not cfg.allow_split:
        return []
    comm = symmetric_communication(snapshot)
    out: List[Operation] = []
    for unit in org.units:
        members = org.members(unit)
        if len(members) < max(cfg.split_min_size, 2 * cfg.min_unit_size):
            continue
        part = spectral_bisection(members, comm)
        if part is None:
            continue
        left, right = part
        if len(left) < cfg.min_unit_size or len(right) < cfg.min_unit_size:
            continue
        out.append(
            Operation(
                "split",
                {
                    "unit": int(unit),
                    "left": left.tolist(),
                    "right": right.tolist(),
                    "right_capacity_fraction": split_capacity_fraction(
                        cfg, snapshot, left, right
                    ),
                },
                authority=org.supervisor(unit),
            )
        )
    return out


def _kam_factor_subset(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    indices: np.ndarray,
    threshold: float,
) -> np.ndarray:
    comm = np.asarray(snapshot.communication, dtype=float)
    kq_const = float(trace.metadata.get("kq", 1.65))
    kc_const = float(trace.metadata.get("kc", 1.08))
    out = np.empty(len(indices), dtype=float)
    for z, i_ in enumerate(indices.tolist()):
        i = int(i_)
        own = int(org.assignment[i])
        external = set()
        partners = np.flatnonzero((comm[i] + comm[:, i]) > threshold)
        for j in partners:
            u = int(org.assignment[int(j)])
            if u != own:
                external.add(u)
        F = 1.0 + float(len(external))
        kq = kq_const if bool(trace.q_impact[i]) else 1.0
        kc = kc_const if bool(trace.c_impact[i]) else 1.0
        out[z] = F * kq * kc * float(trace.ks[i])
    return out


def _changed_assignment_agents(old: Organization, new: Organization) -> np.ndarray:
    return np.flatnonzero(old.assignment != new.assignment).astype(int)


def _affected_agents(
    snapshot: Snapshot, old: Organization, new: Organization, threshold: float
) -> np.ndarray:
    changed = _changed_assignment_agents(old, new)
    if len(changed) == 0:
        return changed
    comm = np.asarray(snapshot.communication, dtype=float)
    affected = set(int(x) for x in changed.tolist())
    for i in changed.tolist():
        partners = np.flatnonzero((comm[i] + comm[:, i]) > threshold)
        affected.update(int(x) for x in partners.tolist())
    return np.array(sorted(affected), dtype=int)


def _share_mismatch_dict(values: Dict[int, float], shares: Dict[int, float]) -> float:
    total = float(sum(values.values()))
    if total <= EPS:
        return 0.0
    v = 0.0
    for u, s in shares.items():
        s = max(float(s), EPS)
        p = float(values.get(u, 0.0)) / total
        v += (p - s) ** 2 / s
    return float(np.sqrt(max(v, 0.0)))


def _management_from_sizes(sizes: Dict[int, int], n_agents: int) -> float:
    if n_agents <= 1:
        return 0.0
    num = sum(float(n * np.log2(n)) for n in sizes.values() if n > 1)
    return float(num / max(n_agents * np.log2(n_agents), EPS))


def evaluate_candidate_local(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    operation: Operation,
    cfg: AlgorithmConfig,
    weights: ObjectiveWeights,
    base_metrics: Optional[StateMetrics] = None,
) -> Tuple[float, float]:
    """Exact one-operation delta from local changes plus aggregate summaries.

    The candidate is *not* evaluated by recomputing every agent. KAM factors are
    recomputed only for agents whose organizational interaction frequency can
    change: reassigned agents and their communication neighbours. Other terms
    are updated from unit summaries and changed-edge classifications. This is
    the numerical counterpart of the D-KAM claim that a decision holon needs
    local boundary information plus aggregate summaries, not the full global
    agent state.
    """
    base_metrics = base_metrics or state_metrics(
        trace, snapshot, org, cfg.communication_threshold
    )
    base_obj, _ = objective_value(
        base_metrics, weights, 0.0, float(snapshot.work.sum())
    )

    candidate = org.clone()
    apply_operation(candidate, operation, cfg)
    threshold = float(cfg.communication_threshold)
    affected = _affected_agents(snapshot, org, candidate, threshold)
    changed = _changed_assignment_agents(org, candidate)

    # Unit raw-work summaries. Only structurally touched units need a fresh sum.
    raw = dict(base_metrics.raw_work_by_unit)
    touched_units = set(int(org.assignment[i]) for i in affected.tolist())
    touched_units.update(int(candidate.assignment[i]) for i in affected.tolist())
    touched_units.update(org.units)
    # For merge/split capacity keys change, so rebuilding raw from unit summaries
    # is simpler and still requires only each unit's aggregate workload.
    raw = {
        u: float(snapshot.work[candidate.assignment == u].sum())
        for u in candidate.units
    }

    # Update KAM exertion using only the affected communication neighbourhood.
    kam = dict(base_metrics.kam_work_by_unit)
    if len(affected):
        old_f = _kam_factor_subset(trace, snapshot, org, affected, threshold)
        new_f = _kam_factor_subset(trace, snapshot, candidate, affected, threshold)
        w = np.asarray(snapshot.work, dtype=float)[affected]
        for idx, agent in enumerate(affected.tolist()):
            old_u = int(org.assignment[agent])
            new_u = int(candidate.assignment[agent])
            kam[old_u] = float(kam.get(old_u, 0.0) - w[idx] * old_f[idx])
            kam[new_u] = float(kam.get(new_u, 0.0) + w[idx] * new_f[idx])
    kam = {u: max(float(kam.get(u, 0.0)), 0.0) for u in candidate.units}

    shares = {u: float(candidate.capacity_shares[u]) for u in candidate.units}
    kam_imbalance = _share_mismatch_dict(kam, shares)

    C = float(snapshot.total_capacity)
    overload = 0.0
    for u, s in shares.items():
        util = float(raw[u]) / max(s * C, EPS)
        overload += s * max(util - 1.0, 0.0) ** 2

    # Only edges incident to changed-assignment agents can change their
    # internal/cross classification.
    cross = float(base_metrics.cross_communication)
    if len(changed):
        comm = symmetric_communication(snapshot)
        changed_set = set(int(x) for x in changed.tolist())
        n = trace.n_agents
        delta_cross = 0.0
        seen_edges = set()
        for i in changed_set:
            partners = np.flatnonzero(comm[i] > 0)
            for j_ in partners.tolist():
                j = int(j_)
                if i == j:
                    continue
                edge = (i, j) if i < j else (j, i)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                w_ij = float(comm[edge[0], edge[1]])
                old_cross = int(org.assignment[edge[0]]) != int(org.assignment[edge[1]])
                new_cross = int(candidate.assignment[edge[0]]) != int(candidate.assignment[edge[1]])
                delta_cross += w_ij * (int(new_cross) - int(old_cross))
        cross += delta_cross
    total_comm = float(base_metrics.total_communication)
    internal = max(total_comm - cross, 0.0)
    if total_comm <= EPS:
        coord_norm = 0.0
    else:
        coord = (
            internal * float(trace.internal_comm_base)
            + cross * float(trace.remote_comm_cost)
        )
        coord_norm = coord / max(total_comm * float(trace.remote_comm_cost), EPS)

    sizes = {u: candidate.size(u) for u in candidate.units}
    management = _management_from_sizes(sizes, trace.n_agents)
    reorg_norm = operation_cost(trace, operation, cfg) / max(float(snapshot.work.sum()), EPS)

    cand_obj = (
        weights.kam_imbalance * kam_imbalance
        + weights.overload * overload
        + weights.coordination * coord_norm
        + weights.management * management
        + weights.reorganization * reorg_norm
    )
    operation.scope_agents = int(len(affected))
    operation.scope_units = int(len(set(org.assignment[affected].tolist()) | set(candidate.assignment[affected].tolist()))) if len(affected) else 0
    operation.observed_step = int(snapshot.step)
    return float(cand_obj - base_obj), float(cand_obj)


def evaluate_candidate(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    operation: Operation,
    cfg: AlgorithmConfig,
    weights: ObjectiveWeights,
) -> Tuple[float, float]:
    """Full reference evaluator, mainly used by tests."""
    base_m = state_metrics(trace, snapshot, org, cfg.communication_threshold)
    base, _ = objective_value(base_m, weights, 0.0, float(snapshot.work.sum()))
    candidate = org.clone()
    apply_operation(candidate, operation, cfg)
    cost = operation_cost(trace, operation, cfg)
    cand_m = state_metrics(trace, snapshot, candidate, cfg.communication_threshold)
    cand, _ = objective_value(cand_m, weights, cost, float(snapshot.work.sum()))
    return float(cand - base), float(cand)


def _record_actual_delta(
    trace: ScenarioTrace,
    current: Snapshot,
    org_before: Organization,
    org_after: Organization,
    op: Operation,
    cfg: AlgorithmConfig,
    weights: ObjectiveWeights,
) -> None:
    before_m = state_metrics(trace, current, org_before, cfg.communication_threshold)
    before, _ = objective_value(before_m, weights, 0.0, float(current.work.sum()))
    after_m = state_metrics(trace, current, org_after, cfg.communication_threshold)
    after, _ = objective_value(
        after_m,
        weights,
        operation_cost(trace, op, cfg),
        float(current.work.sum()),
    )
    op.actual_delta = float(after - before)


class BaseAlgorithm:
    name = "base"

    def __init__(self, cfg: AlgorithmConfig, weights: ObjectiveWeights, seed: int):
        self.cfg = cfg
        self.weights = weights
        self.rng = np.random.default_rng(seed)

    def should_adapt(self, snapshot: Snapshot) -> bool:
        return (
            snapshot.step >= max(int(self.cfg.adapt_start_step), 0)
            and snapshot.step % max(self.cfg.adapt_interval, 1) == 0
        )

    def adapt(self, trace, snapshot, snapshot_history, org) -> AlgorithmResult:
        return AlgorithmResult(org.clone(), observed_step=snapshot.step)


class StaticAlgorithm(BaseAlgorithm):
    name = "static"


class RandomFeasibleAlgorithm(BaseAlgorithm):
    name = "random_feasible"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        cost = 0.0
        moved = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.random_moves_per_adaptation):
                candidates = list(candidate_moves(new, self.cfg))
                if not candidates:
                    break
                op = candidates[int(self.rng.integers(0, len(candidates)))]
                before = new.clone()
                cost += operation_cost(trace, op, self.cfg)
                moved += apply_operation(new, op, self.cfg)
                op.observed_step = snapshot.step
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, cost, moved,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class GreedyLoadAlgorithm(BaseAlgorithm):
    name = "greedy_load"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        total_cost = 0.0
        moved = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                base = state_metrics(trace, snapshot, new, self.cfg.communication_threshold)
                base_score = base.raw_load_imbalance + 2.0 * base.overload_penalty
                best: Optional[Tuple[float, Operation]] = None
                for op in candidate_moves(new, self.cfg):
                    cand = new.clone()
                    apply_operation(cand, op, self.cfg)
                    m = state_metrics(trace, snapshot, cand, self.cfg.communication_threshold)
                    score = m.raw_load_imbalance + 2.0 * m.overload_penalty
                    delta = score - base_score
                    if best is None or delta < best[0]:
                        best = (delta, op)
                if best is None or best[0] >= -1e-9:
                    break
                op = best[1]
                before = new.clone()
                total_cost += operation_cost(trace, op, self.cfg)
                moved += apply_operation(new, op, self.cfg)
                op.estimated_delta = float(best[0])
                op.observed_step = snapshot.step
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, total_cost, moved,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class WorkStealingAlgorithm(BaseAlgorithm):
    name = "work_stealing"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        total_cost = 0.0
        moved = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                m = state_metrics(trace, snapshot, new, self.cfg.communication_threshold)
                source = max(new.units, key=lambda u: m.utilization_by_unit[u])
                target = min(new.units, key=lambda u: m.utilization_by_unit[u])
                if source == target or new.size(source) <= self.cfg.min_unit_size:
                    break
                source_agents = new.members(source)
                src_load = m.raw_work_by_unit[source]
                tgt_load = m.raw_work_by_unit[target]
                src_cap = m.capacity_by_unit[source]
                tgt_cap = m.capacity_by_unit[target]
                current_gap = abs(src_load / max(src_cap, EPS) - tgt_load / max(tgt_cap, EPS))
                best_agent = None
                best_gap = current_gap
                for a in source_agents:
                    w = float(snapshot.work[int(a)])
                    gap = abs(
                        (src_load - w) / max(src_cap, EPS)
                        - (tgt_load + w) / max(tgt_cap, EPS)
                    )
                    if gap < best_gap:
                        best_gap = gap
                        best_agent = int(a)
                if best_agent is None:
                    break
                op = Operation(
                    "move",
                    {"agent": best_agent, "source": int(source), "target": int(target)},
                    estimated_delta=float(best_gap - current_gap),
                    authority=new.lca_supervisor(source, target),
                    observed_step=snapshot.step,
                )
                before = new.clone()
                total_cost += operation_cost(trace, op, self.cfg)
                moved += apply_operation(new, op, self.cfg)
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, total_cost, moved,
            control_messages=len(new.units) * 2 if ops else 0,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class AffinityGreedyAlgorithm(BaseAlgorithm):
    name = "affinity_greedy"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        total_cost = 0.0
        moved = 0
        comm = symmetric_communication(snapshot)
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                best: Optional[Tuple[float, Operation]] = None
                m = state_metrics(trace, snapshot, new, self.cfg.communication_threshold)
                for a in range(trace.n_agents):
                    source = int(new.assignment[a])
                    if new.size(source) <= self.cfg.min_unit_size:
                        continue
                    own_aff = float(comm[a, new.members(source)].sum())
                    for target in new.units:
                        if target == source:
                            continue
                        target_aff = float(comm[a, new.members(target)].sum())
                        projected = (
                            m.raw_work_by_unit[target] + snapshot.work[a]
                        ) / max(m.capacity_by_unit[target], EPS)
                        score = target_aff - own_aff - max(projected - 1.25, 0.0) * (
                            target_aff + 1.0
                        )
                        if best is None or score > best[0]:
                            best = (
                                score,
                                Operation(
                                    "move",
                                    {"agent": int(a), "source": source, "target": int(target)},
                                    estimated_delta=-float(score),
                                    authority=new.lca_supervisor(source, target),
                                    observed_step=snapshot.step,
                                ),
                            )
                if best is None or best[0] <= 0.0:
                    break
                op = best[1]
                before = new.clone()
                total_cost += operation_cost(trace, op, self.cfg)
                moved += apply_operation(new, op, self.cfg)
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, total_cost, moved,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )




def _raw_nonkam_objective(
    trace: ScenarioTrace,
    snapshot: Snapshot,
    org: Organization,
    weights: ObjectiveWeights,
    cfg: AlgorithmConfig,
    reorg_cost: float = 0.0,
) -> float:
    """KAM-free counterpart of the proposed objective for strong baselines.

    It uses raw workload/capacity mismatch in place of KAM exertion mismatch,
    while retaining the same calibrated overload, communication, management,
    and reorganization weights. This makes comparison substantially less
    sensitive to arbitrary baseline coefficients.
    """
    m = state_metrics(trace, snapshot, org, cfg.communication_threshold)
    return float(
        weights.kam_imbalance * m.raw_load_imbalance
        + weights.overload * m.overload_penalty
        + weights.coordination * m.coordination_cost_normalized
        + weights.management * m.management_overhead_normalized
        + weights.reorganization * reorg_cost / max(float(snapshot.work.sum()), EPS)
    )

class BalancedAffinityAlgorithm(BaseAlgorithm):
    """Strong non-KAM multi-objective MOVE baseline.

    It uses raw load balance, overload, communication cost, and migration cost,
    but no KAM F*kq*kc*ks term and no MERGE/SPLIT. This is intended to prevent
    the proposed methods being compared only with single-objective strawmen.
    """

    name = "balanced_affinity"

    def _score(self, trace, snapshot, org, reorg_cost=0.0) -> float:
        return _raw_nonkam_objective(
            trace, snapshot, org, self.weights, self.cfg, reorg_cost
        )

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        cost = 0.0
        moved = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                base = self._score(trace, snapshot, new, 0.0)
                best: Optional[Tuple[float, Operation]] = None
                for op in candidate_moves(new, self.cfg):
                    cand = new.clone()
                    apply_operation(cand, op, self.cfg)
                    c = operation_cost(trace, op, self.cfg)
                    delta = self._score(trace, snapshot, cand, c) - base
                    if best is None or delta < best[0]:
                        best = (float(delta), op)
                if best is None or best[0] >= -1e-9:
                    break
                op = best[1]
                before = new.clone()
                c = operation_cost(trace, op, self.cfg)
                cost += c
                moved += apply_operation(new, op, self.cfg)
                op.estimated_delta = float(best[0])
                op.observed_step = snapshot.step
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, cost, moved,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class GraphPartitionAlgorithm(BaseAlgorithm):
    name = "graph_partition"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        if snapshot.step % max(self.cfg.graph_partition_interval, 1) != 0:
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )
        k = len(new.units)
        clusters = spectral_kway_partition(snapshot.communication, k, self.rng)
        assignment = map_clusters_to_units(clusters, new.assignment, new.units)
        if np.array_equal(assignment, new.assignment):
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )
        moved_agents = np.flatnonzero(assignment != new.assignment).astype(int)
        op = Operation(
            "repartition",
            {"assignment": assignment.tolist(), "moved_agents": moved_agents.tolist()},
            authority=new.root_supervisor,
            observed_step=snapshot.step,
        )
        cost = operation_cost(trace, op, self.cfg)
        candidate = new.clone()
        apply_operation(candidate, op, self.cfg)
        base_score = _raw_nonkam_objective(trace, snapshot, new, self.weights, self.cfg, 0.0)
        candidate_score = _raw_nonkam_objective(trace, snapshot, candidate, self.weights, self.cfg, cost)
        op.estimated_delta = float(candidate_score - base_score)
        if op.estimated_delta >= -self.cfg.min_improvement:
            return AlgorithmResult(
                new, candidate_evaluations=1, proposal_count=1, rejected_proposals=1,
                decision_time_ms=(time.perf_counter() - start) * 1000.0, observed_step=snapshot.step
            )
        before = new.clone()
        moved = apply_operation(new, op, self.cfg)
        _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
        return AlgorithmResult(
            new,
            [op],
            cost,
            moved,
            control_messages=trace.n_agents + len(new.units),
            candidate_evaluations=1, proposal_count=1,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )




class LouvainPartitionAlgorithm(BaseAlgorithm):
    """Interim pilot community baseline using NetworkX Louvain.

    It intentionally does not claim to be Leiden. The final confirmatory
    release should replace or augment this with a validated Leiden dependency
    or externally generated Leiden assignments.
    """

    name = "louvain_partition"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        if not self.should_adapt(snapshot) or snapshot.step % max(self.cfg.graph_partition_interval, 1) != 0:
            return AlgorithmResult(new, decision_time_ms=(time.perf_counter()-start)*1000.0, observed_step=snapshot.step)
        k = len(new.units)
        clusters = louvain_kway_partition(snapshot.communication, k, self.rng)
        assignment = map_clusters_to_units(clusters, new.assignment, new.units)
        if np.array_equal(assignment, new.assignment):
            return AlgorithmResult(new, decision_time_ms=(time.perf_counter()-start)*1000.0, observed_step=snapshot.step)
        moved_agents = np.flatnonzero(assignment != new.assignment).astype(int)
        op = Operation("repartition", {"assignment": assignment.tolist(), "moved_agents": moved_agents.tolist()}, authority=new.root_supervisor, observed_step=snapshot.step)
        cost = operation_cost(trace, op, self.cfg)
        candidate = new.clone()
        apply_operation(candidate, op, self.cfg)
        base_score = _raw_nonkam_objective(trace, snapshot, new, self.weights, self.cfg, 0.0)
        candidate_score = _raw_nonkam_objective(trace, snapshot, candidate, self.weights, self.cfg, cost)
        op.estimated_delta = float(candidate_score - base_score)
        if op.estimated_delta >= -self.cfg.min_improvement:
            return AlgorithmResult(
                new, candidate_evaluations=1, proposal_count=1, rejected_proposals=1,
                decision_time_ms=(time.perf_counter()-start)*1000.0, observed_step=snapshot.step
            )
        before = new.clone()
        moved = apply_operation(new, op, self.cfg)
        _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
        return AlgorithmResult(
            new, [op], cost, moved,
            control_messages=trace.n_agents + len(new.units),
            control_message_hops=trace.n_agents + len(new.units),
            candidate_evaluations=1, proposal_count=1,
            decision_time_ms=(time.perf_counter()-start)*1000.0,
            observed_step=snapshot.step,
        )


class DynamicLeidenAlgorithm(BaseAlgorithm):
    """Migration-aware dynamic Leiden refinement baseline.

    Leiden supplies the target communication communities. Communities are mapped
    to the current resource units with a maximum-overlap assignment to minimize
    unnecessary migration. A bounded number of target-consistent agent moves is
    then selected using the KAM-free objective, making this a strong dynamic
    communication/load baseline without using F*kq*kc*ks.
    """

    name = "dynamic_leiden"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        interval = max(int(self.cfg.leiden_interval), 1)
        if not self.should_adapt(snapshot) or snapshot.step % interval != 0:
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )

        k = len(new.units)
        labels, backend_used = leiden_kway_partition(
            snapshot.communication,
            k,
            new.assignment,
            self.rng,
            backend=self.cfg.leiden_backend,
            resolution=self.cfg.leiden_resolution,
            n_iterations=self.cfg.leiden_iterations,
        )
        target = map_clusters_to_units(labels, new.assignment, new.units)
        mismatch = np.flatnonzero(target != new.assignment).astype(int)
        if len(mismatch) == 0:
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )

        budget = max(
            int(self.cfg.leiden_min_migration_agents),
            int(np.ceil(float(self.cfg.leiden_migration_budget_fraction) * trace.n_agents)),
        )
        budget = min(budget, len(mismatch))
        work = new.clone()
        chosen_agents: List[int] = []
        evaluated = 0

        # Greedy refinement toward the Leiden target. The target itself is fixed
        # for this adaptation point; only KAM-free raw-load/communication terms
        # decide which of the target-consistent moves fit inside the budget.
        for _ in range(budget):
            base = _raw_nonkam_objective(trace, snapshot, work, self.weights, self.cfg, 0.0)
            best: Optional[Tuple[float, int]] = None
            for agent in mismatch.tolist():
                agent = int(agent)
                if agent in chosen_agents:
                    continue
                source = int(work.assignment[agent])
                dest = int(target[agent])
                if source == dest or work.size(source) <= self.cfg.min_unit_size:
                    continue
                op = Operation(
                    "move",
                    {"agent": agent, "source": source, "target": dest},
                    authority=work.lca_supervisor(source, dest),
                    observed_step=snapshot.step,
                )
                cand = work.clone()
                try:
                    apply_operation(cand, op, self.cfg)
                except ValueError:
                    continue
                c = operation_cost(trace, op, self.cfg)
                delta = _raw_nonkam_objective(
                    trace, snapshot, cand, self.weights, self.cfg, c
                ) - base
                evaluated += 1
                if best is None or delta < best[0]:
                    best = (float(delta), agent)
            if best is None or best[0] >= -self.cfg.min_improvement:
                break
            agent = int(best[1])
            source = int(work.assignment[agent])
            dest = int(target[agent])
            apply_operation(
                work,
                Operation("move", {"agent": agent, "source": source, "target": dest}),
                self.cfg,
            )
            chosen_agents.append(agent)

        if not chosen_agents:
            return AlgorithmResult(
                new,
                candidate_evaluations=evaluated,
                proposal_count=1,
                rejected_proposals=1,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )

        assignment = new.assignment.copy()
        assignment[np.asarray(chosen_agents, dtype=int)] = target[np.asarray(chosen_agents, dtype=int)]
        moved_agents = np.flatnonzero(assignment != new.assignment).astype(int)
        op = Operation(
            "repartition",
            {
                "assignment": assignment.tolist(),
                "moved_agents": moved_agents.tolist(),
                "leiden_backend": backend_used,
                "leiden_resolution": float(self.cfg.leiden_resolution),
                "leiden_target_mismatch": int(len(mismatch)),
                "migration_budget_agents": int(budget),
            },
            authority=new.root_supervisor,
            observed_step=snapshot.step,
        )
        cost = operation_cost(trace, op, self.cfg)
        base_score = _raw_nonkam_objective(trace, snapshot, new, self.weights, self.cfg, 0.0)
        candidate = new.clone()
        apply_operation(candidate, op, self.cfg)
        cand_score = _raw_nonkam_objective(trace, snapshot, candidate, self.weights, self.cfg, cost)
        op.estimated_delta = float(cand_score - base_score)
        if op.estimated_delta >= -self.cfg.min_improvement:
            return AlgorithmResult(
                new,
                candidate_evaluations=evaluated + 1,
                proposal_count=1,
                rejected_proposals=1,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )
        before = new.clone()
        moved = apply_operation(new, op, self.cfg)
        _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
        return AlgorithmResult(
            new,
            [op],
            cost,
            moved,
            control_messages=trace.n_agents + len(new.units),
            control_message_hops=trace.n_agents + len(new.units),
            candidate_evaluations=evaluated + 1,
            proposal_count=1,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class ParMETISAdaptiveAlgorithm(BaseAlgorithm):
    """External established HPC baseline using ParMETIS AdaptiveRepart.

    The Python simulator does not reimplement ParMETIS. It serializes the
    current weighted task graph and invokes an external driver. The bundled
    ``external/parmetis_adaptive_driver.cpp`` calls
    ``ParMETIS_V3_AdaptiveRepart`` when compiled with MPI/METIS/ParMETIS.
    """

    name = "parmetis_adaptive"

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        if trace.name != "hpc":
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )
        if not self.should_adapt(snapshot) or snapshot.step % max(int(self.cfg.hpc_external_interval), 1) != 0:
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )

        ext, meta = run_external_partitioner(
            trace, snapshot, new, self.cfg, seed=int(trace.seed * 1000003 + snapshot.step)
        )
        units = list(meta["units"])
        assignment = np.array([int(units[int(p)]) for p in ext.partition.tolist()], dtype=int)
        if np.array_equal(assignment, new.assignment):
            return AlgorithmResult(
                new,
                candidate_evaluations=1,
                proposal_count=1,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )
        moved_agents = np.flatnonzero(assignment != new.assignment).astype(int)
        present_parts = set(int(x) for x in ext.partition.tolist())
        empty_parts = sorted(set(range(len(units))) - present_parts)
        op = Operation(
            "repartition",
            {
                "assignment": assignment.tolist(),
                "moved_agents": moved_agents.tolist(),
                "external_backend": "ParMETIS_V3_AdaptiveRepart",
                "external_edgecut": int(ext.edgecut),
                "external_graph_edges": int(meta["edge_count"]),
                "external_empty_partition_count": int(len(empty_parts)),
                "external_empty_partitions": ",".join(map(str, empty_parts)),
            },
            authority=new.root_supervisor,
            observed_step=snapshot.step,
        )
        # ParMETIS owns the optimization criterion for this baseline. We apply
        # its adaptive repartitioning result as returned rather than filtering it
        # through the KAM or KAM-free objective.
        before = new.clone()
        # ParMETIS partitions model fixed physical resource pools. A pool with
        # no assigned task still exists and retains its capacity, so an empty
        # returned partition is valid for this baseline.
        new.allow_empty_units = True
        cost = operation_cost(trace, op, self.cfg)
        moved = apply_operation(new, op, self.cfg)
        _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
        return AlgorithmResult(
            new,
            [op],
            cost,
            moved,
            control_messages=0,
            control_message_hops=0,
            candidate_evaluations=1,
            proposal_count=1,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class DomainHeuristicAlgorithm(BaseAlgorithm):
    """Strong domain-aware non-KAM MOVE baseline for the publication pilot.

    The score combines raw capacity balance, overload, communication cost,
    migration cost and one explicit domain term (spatial compactness, regional
    impurity, functional skill impurity, or HPC spatial compactness). It never
    uses F*kq*kc*ks and never uses MERGE/SPLIT.
    """

    name = "domain_heuristic"

    def _score(self, trace, snapshot, org, reorg_cost=0.0) -> float:
        base = _raw_nonkam_objective(
            trace, snapshot, org, self.weights, self.cfg, reorg_cost
        )
        domain = _domain_penalty(trace, snapshot, org)
        return float(base + 0.50 * domain)

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        cost = 0.0
        moved = 0
        evaluated = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                base = self._score(trace, snapshot, new, 0.0)
                best: Optional[Tuple[float, Operation]] = None
                for op in candidate_moves(new, self.cfg):
                    evaluated += 1
                    cand = new.clone()
                    apply_operation(cand, op, self.cfg)
                    c = operation_cost(trace, op, self.cfg)
                    delta = self._score(trace, snapshot, cand, c) - base
                    if best is None or delta < best[0]:
                        best = (float(delta), op)
                if best is None or best[0] >= -1e-9:
                    break
                op = best[1]
                before = new.clone()
                c = operation_cost(trace, op, self.cfg)
                cost += c
                moved += apply_operation(new, op, self.cfg)
                op.estimated_delta = float(best[0])
                op.observed_step = snapshot.step
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        return AlgorithmResult(
            new, ops, cost, moved,
            candidate_evaluations=evaluated,
            decision_time_ms=(time.perf_counter()-start)*1000.0,
            observed_step=snapshot.step,
        )

class CKAMAlgorithm(BaseAlgorithm):
    name = "c_kam"

    def _best(self, trace, snapshot, org) -> Tuple[Optional[Operation], int]:
        candidates = list(candidate_moves(org, self.cfg))
        candidates += list(candidate_merges(org, self.cfg))
        candidates += list(candidate_splits(org, snapshot, self.cfg))
        base = state_metrics(trace, snapshot, org, self.cfg.communication_threshold)
        best: Optional[Operation] = None
        best_delta = np.inf
        for op in candidates:
            try:
                delta, _ = evaluate_candidate_local(
                    trace, snapshot, org, op, self.cfg, self.weights, base
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if delta < best_delta:
                best_delta, best = delta, op
        if best is not None:
            best.estimated_delta = float(best_delta)
        return best, len(candidates)

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        ops: List[Operation] = []
        total_cost = 0.0
        moved = 0
        proposals = 0
        candidate_evaluations = 0
        if self.should_adapt(snapshot):
            for _ in range(self.cfg.max_ops_per_adaptation):
                op, evaluated = self._best(trace, snapshot, new)
                candidate_evaluations += evaluated
                if op is None:
                    break
                proposals += 1
                if op.estimated_delta >= -self.cfg.min_improvement:
                    break
                before = new.clone()
                total_cost += operation_cost(trace, op, self.cfg)
                moved += apply_operation(new, op, self.cfg)
                _record_actual_delta(trace, snapshot, before, new, op, self.cfg, self.weights)
                ops.append(op)
        # Centralized collection forwards each agent-level record to the root;
        # hop-volume therefore grows with hierarchy depth.
        control = trace.n_agents + len(new.units) if self.should_adapt(snapshot) else 0
        control_hops = 0
        if self.should_adapt(snapshot):
            control_hops = sum(
                new.supervisor_distance(int(new.assignment[i]), new.root_supervisor)
                for i in range(trace.n_agents)
            ) + len(new.units)
        return AlgorithmResult(
            new,
            ops,
            total_cost,
            moved,
            control_messages=control,
            control_message_hops=control_hops,
            candidate_evaluations=candidate_evaluations,
            proposal_count=proposals,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=snapshot.step,
        )


class DKAMAlgorithm(BaseAlgorithm):
    name = "d_kam"

    def _candidate_targets(
        self, snapshot: Snapshot, org: Organization, source: int
    ) -> List[int]:
        """Targets visible through sibling summaries or local boundary traffic."""
        comm = symmetric_communication(snapshot)
        members = org.members(source)
        visible = set(org.sibling_units(source))
        boundary: List[Tuple[float, int]] = []
        for other in org.units:
            if other == source:
                continue
            vol = float(comm[np.ix_(members, org.members(other))].sum())
            if vol > 0:
                boundary.append((vol, int(other)))
        boundary.sort(reverse=True)
        limit = max(int(self.cfg.dkam_candidate_targets_per_unit), 1)
        visible.update(u for _, u in boundary[:limit])
        return sorted(visible)

    def _local_proposals(self, trace, observed, org) -> Tuple[List[Operation], int]:
        proposals: List[Operation] = []
        evaluated = 0
        base = state_metrics(trace, observed, org, self.cfg.communication_threshold)
        per_unit_limit = max(int(self.cfg.dkam_local_proposals_per_unit), 1)

        if self.cfg.allow_move:
            for source in org.units:
                if org.size(source) <= self.cfg.min_unit_size:
                    continue
                targets = self._candidate_targets(observed, org, source)
                scored: List[Tuple[float, Operation]] = []
                for agent in org.members(source):
                    for target in targets:
                        if target == source:
                            continue
                        op = Operation(
                            "move",
                            {"agent": int(agent), "source": int(source), "target": int(target)},
                            authority=org.lca_supervisor(source, target),
                            observed_step=observed.step,
                        )
                        try:
                            evaluated += 1
                            delta, _ = evaluate_candidate_local(
                                trace, observed, org, op, self.cfg, self.weights, base
                            )
                        except ValueError:
                            continue
                        op.estimated_delta = float(delta)
                        scored.append((delta, op))
                scored.sort(key=lambda x: x[0])
                proposals.extend(op for _, op in scored[:per_unit_limit])

        if self.cfg.allow_merge and len(org.units) > 2:
            seen = set()
            for unit in org.units:
                targets = self._candidate_targets(observed, org, unit)
                if not targets:
                    continue
                comm = symmetric_communication(observed)
                members = org.members(unit)
                other = max(
                    targets,
                    key=lambda v: float(comm[np.ix_(members, org.members(v))].sum()),
                )
                pair = tuple(sorted((int(unit), int(other))))
                if pair in seen:
                    continue
                seen.add(pair)
                op = Operation(
                    "merge",
                    {"unit_a": pair[0], "unit_b": pair[1]},
                    authority=org.lca_supervisor(pair[0], pair[1]),
                    observed_step=observed.step,
                )
                try:
                    evaluated += 1
                    delta, _ = evaluate_candidate_local(
                        trace, observed, org, op, self.cfg, self.weights, base
                    )
                except ValueError:
                    continue
                op.estimated_delta = float(delta)
                proposals.append(op)

        if self.cfg.allow_split:
            for op in candidate_splits(org, observed, self.cfg):
                op.observed_step = observed.step
                try:
                    evaluated += 1
                    delta, _ = evaluate_candidate_local(
                        trace, observed, org, op, self.cfg, self.weights, base
                    )
                except ValueError:
                    continue
                op.estimated_delta = float(delta)
                proposals.append(op)

        return proposals, evaluated

    @staticmethod
    def _aggregation_message_count(org: Organization, n_agents: int) -> int:
        # agent->leaf summary + leaf->manager + manager->parent aggregation
        return int(n_agents + len(org.units) + len(org.supervisor_parent))

    @staticmethod
    def _transaction_message_count(org: Organization, op: Operation) -> int:
        p = op.payload
        if op.operator == "move":
            units = [int(p["source"]), int(p["target"])]
        elif op.operator == "merge":
            units = [int(p["unit_a"]), int(p["unit_b"])]
        elif op.operator == "split":
            units = [int(p["unit"])]
        else:
            units = []
        # Proposal path plus PREPARE/ACK/COMMIT along affected branches.
        hops = sum(org.supervisor_distance(u, op.authority) for u in units)
        return int(max(1, hops) + 3 * max(1, hops))

    def adapt(self, trace, snapshot, snapshot_history, org):
        start = time.perf_counter()
        new = org.clone()
        if not self.should_adapt(snapshot):
            return AlgorithmResult(
                new,
                decision_time_ms=(time.perf_counter() - start) * 1000.0,
                observed_step=snapshot.step,
            )

        delay = max(0, int(self.cfg.dkam_observation_delay))
        observed_index = max(0, len(snapshot_history) - 1 - delay)
        observed = snapshot_history[observed_index]
        ops: List[Operation] = []
        total_cost = 0.0
        moved = 0
        total_proposals = 0
        rejected = 0
        control_messages = self._aggregation_message_count(new, trace.n_agents)
        # Aggregation messages are one-hop transmissions; unlike C-KAM, raw
        # agent records are not forwarded all the way to the root.
        control_hops = control_messages
        candidate_evaluations = 0

        for _ in range(self.cfg.max_ops_per_adaptation):
            proposals, evaluated = self._local_proposals(trace, observed, new)
            candidate_evaluations += evaluated
            total_proposals += len(proposals)
            if not proposals:
                break
            proposals.sort(key=lambda op: op.estimated_delta)
            chosen = proposals[0]
            threshold = -(self.cfg.min_improvement + self.cfg.dkam_error_margin)
            if chosen.estimated_delta >= threshold:
                rejected += len(proposals)
                break
            before = new.clone()
            tx = self._transaction_message_count(before, chosen)
            control_messages += tx
            control_hops += tx
            try:
                c = operation_cost(trace, chosen, self.cfg)
                moved_now = apply_operation(new, chosen, self.cfg)
            except ValueError:
                rejected += 1
                break
            _record_actual_delta(trace, snapshot, before, new, chosen, self.cfg, self.weights)
            total_cost += c
            moved += moved_now
            ops.append(chosen)

        return AlgorithmResult(
            new,
            ops,
            total_cost,
            moved,
            control_messages=control_messages,
            control_message_hops=control_hops,
            candidate_evaluations=candidate_evaluations,
            proposal_count=total_proposals,
            rejected_proposals=rejected,
            decision_time_ms=(time.perf_counter() - start) * 1000.0,
            observed_step=observed.step,
        )


ALGORITHM_CLASSES = {
    "static": StaticAlgorithm,
    "random_feasible": RandomFeasibleAlgorithm,
    "greedy_load": GreedyLoadAlgorithm,
    "affinity_greedy": AffinityGreedyAlgorithm,
    "balanced_affinity": BalancedAffinityAlgorithm,
    "graph_partition": GraphPartitionAlgorithm,
    "louvain_partition": LouvainPartitionAlgorithm,
    "dynamic_leiden": DynamicLeidenAlgorithm,
    "parmetis_adaptive": ParMETISAdaptiveAlgorithm,
    "domain_heuristic": DomainHeuristicAlgorithm,
    "work_stealing": WorkStealingAlgorithm,
    "c_kam": CKAMAlgorithm,
    "d_kam": DKAMAlgorithm,
}


def make_algorithm(
    name: str, cfg: AlgorithmConfig, weights: ObjectiveWeights, seed: int
) -> BaseAlgorithm:
    if name not in ALGORITHM_CLASSES:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(ALGORITHM_CLASSES)}")
    return ALGORITHM_CLASSES[name](cfg, weights, seed)
