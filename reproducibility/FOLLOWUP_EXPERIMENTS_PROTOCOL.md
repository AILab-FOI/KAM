# D1 + S1 + A1 Follow-up Experiment Protocol

Protocol ID: `D1-S1-A1-followup-v1`

This package runs the three predeclared experiments needed after the frozen C1.1 confirmatory benchmark. It does **not** alter the C1.1 objective, thresholds, reorganization costs, D-KAM safety margin, adaptation interval, or any calibrated baseline parameter.

## D1 — D-KAM observation-delay robustness

New runs use D-KAM only at delays `0, 2, 4, 8`, all four domains, seeds `3001..3030`, 64 agents, 8 initial units, 200 steps. Delay `1` is reused from C1.1 when a local `C1_1_results` directory is present. The D-KAM error margin remains frozen at its C1.1 value. The output records estimation error and the frequency of accepted operations whose true current-state objective delta is worsening.

New-run count: `4 domains × 4 delays × 30 seeds = 480`.

## S1 — scalability

C-KAM and D-KAM are run in all four domains at `N = 32, 64, 128, 256, 512, 1024`, with one initial unit per 8 agents (`U0 = N/8`), 10 fresh seeds `5001..5010`, and 200 steps. All scientific parameters remain C1.1-frozen. Sizes are executed in ascending order so all smaller-scale checkpoints are secured before the expensive `N=1024` cases.

New-run count: `4 domains × 6 sizes × 2 methods × 10 seeds = 480`.

## A1 — C-KAM operator ablation

Three new C-KAM variants are run in all four domains using seeds `3001..3030`, 64 agents, 8 initial units, and 200 steps:

- `move_only`: MOVE enabled; MERGE/SPLIT disabled.
- `move_merge`: MOVE/MERGE enabled; SPLIT disabled.
- `move_split`: MOVE/SPLIT enabled; MERGE disabled.

The full MOVE+MERGE+SPLIT C-KAM condition is reused from C1.1 when available locally.

New-run count: `4 domains × 3 variants × 30 seeds = 360`.

## Total

The production plan contains exactly **1,320 new runs**. Every completed scenario/condition/seed/method is checkpointed independently. Re-running the same launcher verifies the frozen context and skips only checkpoints whose protocol, item, trace and effective algorithm hashes all match.

A heartbeat is printed every 60 seconds during a long individual run. The launcher also prints overall/block progress, elapsed run time, and an approximate ETA.

## Important resource note

The current simulator stores dense `N × N` communication matrices for each snapshot. For `N=1024` and 200 steps, the raw float64 communication matrices alone are roughly 1.56 GiB per generated trace, before Python/object overhead. The script runs traces sequentially and explicitly releases the previous trace, but a machine with ample RAM is recommended for the largest S1 points.
