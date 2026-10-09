# Formal specification: C-KAM and holonic D-KAM (v1.0-rc1)

## 1. Dynamic multi-agent organization

At time `t`, let agents be `V={1,...,N}` and let the leaf organizational units form a partition `U_t`. Assignment is

`P_t : V -> U_t`.

Each unit `u` owns a positive capacity share `s_u`, with

`sum_u s_u = 1`.

The total available processing capacity in step `t` is `C(t)`, hence

`C_u(t) = s_u C(t)`.

The leaf units are embedded in a supervisor tree (holarchy). For two leaf units `u,v`, `LCA(u,v)` is their lowest common supervisory holon.

## 2. Dynamic work and backlog

Let `a_i(t)` be exogenous arriving work and `b_i(t)` backlog entering the step. Offered work is

`d_i(t) = a_i(t) + b_i(t)`.

After organizational adaptation, reorganization cost `R_t` may reduce effective total capacity:

`C_eff(t) = max(C(t) - R_t, 0)`.

Within each leaf unit, capacity is shared proportionally among offered agent workloads. If

`D_u(t)=sum_{i:P(i)=u} d_i(t)`, then the unit service fraction is

`phi_u(t)=min(1, s_u C_eff(t) / D_u(t))`

for nonzero `D_u`. Agent processing and backlog are

`y_i(t)=phi_{P(i)}(t) d_i(t)`

and

`b_i(t+1)=d_i(t)-y_i(t)`.

Thus work is conserved: cumulative arrivals equal cumulative processed work plus final backlog.

## 3. Generalized KAM activity weight and exertion

Each simulated agent represents an aggregate activity bundle. Let weighted communication volume between agents `i,j` be `m_ij(t)`.

The activity's own organizational unit is counted once. Its interaction frequency is

`F_i(t) = 1 + | { u != P_t(i) : exists j with P_t(j)=u and m_ij(t)>theta } |`.

The KAM-style weight is

`W_i(t)=F_i(t) kq_i kc_i ks_i`.

Dynamic organizational exertion is defined as

`E_i(t)=d_i(t) W_i(t)`.

This is an explicit continuation-study extension. `d_i(t)` represents the amount of the activity executed/demanded during the window. If execution count itself is identified with the original frequency, the construction approaches the repeated-frequency structure of the original NIOP calculation.

Unit exertion is

`E_u(t)=sum_{i:P_t(i)=u} E_i(t)`.

## 4. Capacity-share mismatch

For nonnegative unit quantities `x_u`, let

`p_u = x_u / sum_v x_v`.

The mismatch between quantity and capacity distribution is

`B(x,s) = sqrt( sum_u (p_u-s_u)^2 / s_u )`.

C-KAM and D-KAM use

`B_KAM(t)=B(E(t),s(t))`.

Raw work mismatch `B_raw(t)=B(D(t),s(t))` is retained as an outcome metric but is not the KAM objective component.

## 5. Overload

Raw utilization is

`rho_u(t)=D_u(t)/(s_u C(t))`.

The capacity-weighted overload penalty is

`O(t)=sum_u s_u max(rho_u(t)-1,0)^2`.

## 6. Communication/coordination cost

Let `M_cross(t)` and `M_internal(t)` be the total weighted traffic across and within leaf units, with total `M=M_cross+M_internal`.

With internal communication cost `c_in` and cross-unit cost `c_out > c_in`, normalized network cost is

`X(t) = [c_in M_internal + c_out M_cross] / [c_out M]`

for `M>0`, and zero otherwise.

This term captures communication **volume**. The KAM `F_i` term separately captures the **breadth** of organizational boundaries touched by an activity.

## 7. Structural management overhead

For unit sizes `n_u`, define

`H(t)= [sum_u n_u log2(n_u)] / [N log2(N)]`

for `N>1`.

This explicit modelling extension represents span/coordination complexity. It prevents a trivial one-unit solution: merging lowers cross-unit traffic but increases `H`; splitting does the reverse.

## 8. Reorganization cost

For an operation `o`,

- `MOVE(i,u,v)` costs the migration/handover cost `r_i`;
- `MERGE(u,v)` costs `kappa_merge`;
- `SPLIT(u,S1,S2)` costs `kappa_split`.

The objective uses normalized cost

`R_norm(o)=R(o)/sum_i d_i(t)`.

## 9. Frozen objective form

For nonnegative weights `alpha,beta,gamma,eta,lambda`,

`J = alpha B_KAM + beta O + gamma X + eta H + lambda R_norm`.

The **form and normalization** are frozen. Numerical weights are calibrated only on designated pilot seeds and then frozen for confirmatory experiments.

## 10. Structural operators

### MOVE
`MOVE(i,u,v)` changes only `P(i)` from source `u` to existing target `v`. A move is infeasible if it violates minimum unit size.

### MERGE
`MERGE(u,v)` replaces two units by their union and sums their capacity shares. If they have different immediate supervisors, the merged unit attaches to `LCA(u,v)`.

### SPLIT
`SPLIT(u,S1,S2)` partitions the members of `u`, preserving total capacity. In the default rule, capacity is divided in proportion to observed offered work, so a split cannot improve raw balance merely by creating a new unit. A spectral bisection of the local communication graph is the v1 split-candidate oracle; it is not claimed to be part of original KAM.

## 11. Local exactness of a single-operation delta

A structural operation changes assignment labels only for a subset `C` of agents. A KAM frequency `F_i` can change only for an agent in `C` or for an agent communicating with a member of `C`. Therefore define the affected neighbourhood

`A = C union N(C)`.

Only `A` needs its KAM factors recomputed.

Likewise, cross/internal status can change only for communication edges incident to `C`. Unit raw loads and capacities change only in structurally affected units. The remaining global information needed for `J` consists of aggregate unit summaries and fixed totals.

The simulator implements this localized delta and regression-tests it against full global recomputation for MOVE, MERGE and SPLIT.

## 12. C-KAM

At each adaptation epoch:

1. collect the current global observation;
2. generate all feasible MOVE candidates, all feasible MERGE pairs, and one spectral SPLIT candidate per eligible unit;
3. compute `Delta J(o)` for each candidate;
4. choose `o* = argmin Delta J(o)`;
5. commit iff `Delta J(o*) < -epsilon`;
6. repeat up to `max_ops_per_adaptation`.

With a stationary state, `J >= 0`, and every accepted operation reducing the true objective by at least `epsilon>0`, no more than `ceil(J_0/epsilon)` accepted operations can occur before termination. The terminal state is an epsilon-local optimum relative to the generated MOVE/MERGE/SPLIT candidate set, not a global optimum.

## 13. Holonic D-KAM

### Observation and aggregation
Agents report local load and boundary communication to their leaf manager. Managers form aggregate summaries and propagate summaries upward in the supervisor tree. Raw agent state is not forwarded globally.

### Candidate generation
For source unit `u`, D-KAM considers:

- sibling units visible through the immediate supervisor;
- a bounded number of units with the strongest observed boundary traffic;
- local spectral SPLIT candidates inside `u`.

Each leaf manager emits at most a configured number of MOVE proposals plus relevant MERGE/SPLIT proposals.

### Authority
An operation is authorized by the lowest supervisory holon spanning the affected units:

- local SPLIT: immediate supervisor of the unit;
- sibling MOVE/MERGE: their common manager;
- cross-branch MOVE/MERGE: `LCA(u,v)`.

### Commit
The simulator accounts for a lightweight proposal / PREPARE / ACK / COMMIT exchange. Disjoint subtrees may conceptually reorganize independently; the current reference implementation serializes accepted operations within one simulation process for deterministic comparison.

### Delay
D-KAM can evaluate proposals from an observation delayed by `d` steps. The event log stores both estimated improvement under the observed state and actual improvement under the current state.

### Conditional conservative-descent result
If the estimation error is bounded by `delta`, i.e.

`|DeltaJ_hat - DeltaJ| <= delta`,

and D-KAM accepts only when

`DeltaJ_hat < -(epsilon + delta)`,

then the true current objective decreases by at least `epsilon`. Under stationary conditions this yields the same finite-descent argument as C-KAM. In dynamic experiments `delta` is generally unknown; violations are measured empirically rather than assumed away.

## 14. Decentralization/scalability observables

The reference implementation records:

- control-message count;
- control-message hop transmissions;
- candidate evaluations;
- wall-clock decision time;
- proposal/rejection counts;
- observation lag;
- local information footprint (`scope_agents`, `scope_units`) per operation.

These are analyzed together with performance quality to quantify the price/benefit of decentralization.
