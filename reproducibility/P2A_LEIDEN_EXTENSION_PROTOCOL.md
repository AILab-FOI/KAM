# P2a Dynamic-Leiden extension

The original comparator-only P2 pilot successfully validated the established igraph Leiden and ParMETIS backends. ParMETIS can be frozen from that pilot, but Dynamic Leiden cannot yet be frozen under the predeclared rule because communication reduction continued to improve materially between migration budgets 0.10 and 0.20.

## Frozen before this extension

- Leiden iterations: **2**. Four iterations did not produce a material or consistent communication advantage over two iterations across the original grid, so the smaller value is retained.
- Leiden resolution: **1.0**.
- Leiden interval: **8**.
- P2 seeds: **1201, 1213, 1223**.
- P2 problem size: 48 agents, 6 units, 80 steps.

## Extension grid

Only migration budget is varied: **0.30, 0.40, 0.50**. The existing 0.20/2 result is the lower anchor.

## Selection rule fixed before extension results are seen

Let R(b) be the `ALL_DOMAINS_DESCRIPTIVE` median relative reduction in cross-unit communication versus Static at migration budget b. Select the smallest b in {0.20, 0.30, 0.40} for which the next larger tested budget improves R by **less than 0.02 absolute** (less than two percentage points). If no such plateau occurs, select **0.50** as the declared ceiling, because allowing more than half the agents to migrate at one adaptation event is treated as near-global repartitioning rather than bounded dynamic refinement.

No C-KAM or D-KAM method is executed by this extension.
