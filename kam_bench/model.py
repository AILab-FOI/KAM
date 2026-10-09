from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Snapshot:
    """Exogenous or runtime state observed at one simulation step.

    In generated traces ``work`` is exogenous arriving work. During a run the
    simulator may create a runtime Snapshot whose work includes carried backlog.
    The communication matrix and capacity remain exogenous and paired across
    methods for the same trace/seed.
    """

    step: int
    work: np.ndarray
    communication: np.ndarray
    total_capacity: float
    phase: int
    disturbance: str = ""


@dataclass
class ScenarioTrace:
    name: str
    seed: int
    q_impact: np.ndarray
    c_impact: np.ndarray
    ks: np.ndarray
    migration_cost: np.ndarray
    initial_assignment: np.ndarray
    initial_capacity_shares: Dict[int, float]
    initial_unit_supervisor: Dict[int, int]
    initial_supervisor_parent: Dict[int, int]
    root_supervisor: int
    snapshots: List[Snapshot]
    remote_comm_cost: float
    internal_comm_base: float
    management_scale: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        return int(len(self.initial_assignment))

    @property
    def n_units(self) -> int:
        return int(len(self.initial_capacity_shares))


@dataclass
class Organization:
    """Current mutable leaf partition plus a supervisor holarchy.

    Leaf organizational units own capacity shares and contain agents. Each leaf
    has an immediate supervisor. Supervisors can themselves have supervisors,
    with ``root_supervisor`` as the top authority. Structural operations change
    leaf membership but do not create physical capacity.
    """

    assignment: np.ndarray
    capacity_shares: Dict[int, float]
    unit_supervisor: Dict[int, int]
    supervisor_parent: Dict[int, int]
    next_unit_id: int
    root_supervisor: int = -1
    # Some external partitioners can legitimately leave a physical resource
    # partition temporarily empty. This flag is False for KAM and all native
    # baselines, and is enabled only by the ParMETIS adapter.
    allow_empty_units: bool = False

    def clone(self) -> "Organization":
        return Organization(
            assignment=self.assignment.copy(),
            capacity_shares=dict(self.capacity_shares),
            unit_supervisor=dict(self.unit_supervisor),
            supervisor_parent=dict(self.supervisor_parent),
            next_unit_id=int(self.next_unit_id),
            root_supervisor=int(self.root_supervisor),
            allow_empty_units=bool(self.allow_empty_units),
        )

    @property
    def units(self) -> List[int]:
        return sorted(int(u) for u in self.capacity_shares)

    def members(self, unit: int) -> np.ndarray:
        return np.flatnonzero(self.assignment == unit)

    def size(self, unit: int) -> int:
        return int(np.count_nonzero(self.assignment == unit))

    def supervisor(self, unit: int) -> int:
        return int(self.unit_supervisor.get(int(unit), self.root_supervisor))

    def supervisor_ancestors(self, supervisor: int) -> List[int]:
        cur = int(supervisor)
        out = [cur]
        seen = {cur}
        while cur != self.root_supervisor:
            cur = int(self.supervisor_parent.get(cur, self.root_supervisor))
            if cur in seen:
                raise ValueError("cycle in supervisor holarchy")
            out.append(cur)
            seen.add(cur)
        return out

    def unit_ancestors(self, unit: int) -> List[int]:
        return self.supervisor_ancestors(self.supervisor(unit))

    def lca_supervisor(self, unit_a: int, unit_b: int) -> int:
        a_path = self.unit_ancestors(int(unit_a))
        b_set = set(self.unit_ancestors(int(unit_b)))
        for x in a_path:
            if x in b_set:
                return int(x)
        return int(self.root_supervisor)

    def supervisor_distance(self, unit: int, authority: int) -> int:
        """Number of upward supervisory hops from a unit to an authority."""
        authority = int(authority)
        path = self.unit_ancestors(int(unit))
        if authority not in path:
            return max(len(path) - 1, 0)
        return int(path.index(authority) + 1)  # leaf->immediate supervisor is one hop

    def sibling_units(self, unit: int) -> List[int]:
        sup = self.supervisor(int(unit))
        return [u for u in self.units if u != int(unit) and self.supervisor(u) == sup]

    def validate(self) -> None:
        units = set(self.units)
        if not units:
            raise ValueError("organization has no units")
        assigned = set(int(x) for x in np.unique(self.assignment))
        if not assigned.issubset(units):
            raise ValueError(f"assigned units {assigned} are not a subset of capacity units {units}")
        if not self.allow_empty_units and assigned != units:
            raise ValueError(f"assigned units {assigned} != capacity units {units}")
        if not self.allow_empty_units and any(self.size(u) <= 0 for u in units):
            raise ValueError("empty units are not allowed")
        shares = np.array(list(self.capacity_shares.values()), dtype=float)
        if np.any(~np.isfinite(shares)) or np.any(shares <= 0):
            raise ValueError("capacity shares must be finite and positive")
        if not np.isclose(shares.sum(), 1.0, atol=1e-9):
            raise ValueError(f"capacity shares must sum to 1, got {shares.sum()}")
        if set(self.unit_supervisor) != units:
            raise ValueError("unit_supervisor keys must match current units")
        # Validate every manager path terminates at the root.
        for u in units:
            self.unit_ancestors(u)

    @classmethod
    def from_trace(cls, trace: ScenarioTrace) -> "Organization":
        units = sorted(trace.initial_capacity_shares)
        org = cls(
            assignment=trace.initial_assignment.copy(),
            capacity_shares=dict(trace.initial_capacity_shares),
            unit_supervisor=dict(trace.initial_unit_supervisor),
            supervisor_parent=dict(trace.initial_supervisor_parent),
            next_unit_id=max(units) + 1 if units else 0,
            root_supervisor=int(trace.root_supervisor),
            allow_empty_units=False,
        )
        org.validate()
        return org

    def move(self, agent: int, target: int, min_unit_size: int = 1) -> Tuple[int, int]:
        agent = int(agent)
        target = int(target)
        source = int(self.assignment[agent])
        if target not in self.capacity_shares:
            raise ValueError(f"unknown target unit {target}")
        if source == target:
            raise ValueError("source and target are identical")
        if self.size(source) <= min_unit_size:
            raise ValueError("move would violate min_unit_size")
        self.assignment[agent] = target
        self.validate()
        return source, target

    def merge(self, unit_a: int, unit_b: int) -> int:
        a, b = int(unit_a), int(unit_b)
        if a == b or a not in self.capacity_shares or b not in self.capacity_shares:
            raise ValueError("invalid merge")
        keep, retire = (a, b) if a < b else (b, a)
        authority = self.lca_supervisor(keep, retire)
        same_parent = self.supervisor(keep) == self.supervisor(retire)
        self.assignment[self.assignment == retire] = keep
        self.capacity_shares[keep] += self.capacity_shares.pop(retire)
        self.unit_supervisor[keep] = self.supervisor(keep) if same_parent else authority
        self.unit_supervisor.pop(retire, None)
        self.validate()
        return keep

    def split(
        self,
        unit: int,
        left: Sequence[int],
        right: Sequence[int],
        right_capacity_fraction: float,
    ) -> int:
        unit = int(unit)
        left_arr = np.asarray(left, dtype=int)
        right_arr = np.asarray(right, dtype=int)
        members = set(self.members(unit).tolist())
        if set(left_arr.tolist()) | set(right_arr.tolist()) != members:
            raise ValueError("split parts must cover exactly the current unit members")
        if set(left_arr.tolist()) & set(right_arr.tolist()):
            raise ValueError("split parts overlap")
        if len(left_arr) == 0 or len(right_arr) == 0:
            raise ValueError("split parts must be non-empty")
        frac = float(right_capacity_fraction)
        if not (0.0 < frac < 1.0):
            raise ValueError("right_capacity_fraction must be in (0,1)")
        new_unit = int(self.next_unit_id)
        self.next_unit_id += 1
        old_share = self.capacity_shares[unit]
        self.capacity_shares[unit] = old_share * (1.0 - frac)
        self.capacity_shares[new_unit] = old_share * frac
        self.assignment[right_arr] = new_unit
        self.unit_supervisor[new_unit] = self.supervisor(unit)
        self.validate()
        return new_unit

    def replace_assignment(self, assignment: np.ndarray) -> None:
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape != self.assignment.shape:
            raise ValueError("assignment shape mismatch")
        allowed = set(self.units)
        assigned = set(int(x) for x in np.unique(assignment))
        if not assigned.issubset(allowed):
            raise ValueError("replacement assignment contains unknown units")
        if not self.allow_empty_units and assigned != allowed:
            raise ValueError("replacement assignment would create empty units")
        self.assignment = assignment.copy()
        self.validate()


@dataclass
class Operation:
    operator: str
    payload: Dict[str, object]
    estimated_delta: float = 0.0
    authority: int = -1
    observed_step: int = -1
    actual_delta: float = float("nan")
    scope_agents: int = 0
    scope_units: int = 0


@dataclass
class AlgorithmResult:
    organization: Organization
    operations: List[Operation] = field(default_factory=list)
    reorganization_cost: float = 0.0
    moved_agents: int = 0
    control_messages: int = 0
    control_message_hops: int = 0
    candidate_evaluations: int = 0
    proposal_count: int = 0
    rejected_proposals: int = 0
    decision_time_ms: float = 0.0
    observed_step: int = -1
