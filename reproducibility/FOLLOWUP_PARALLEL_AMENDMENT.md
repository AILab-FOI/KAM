# D1/S1/A1 parallel-resume and S1.1 feasibility amendment

## Why this patch exists

The original runner is intentionally single-process. During the first S1 N=1024 C-KAM run, wall time exceeded six hours while one logical CPU was saturated. No N=1024 result had completed at the point this operational amendment was adopted. All D1 runs and the already-completed S1 runs remain valid and are reused from their existing per-run checkpoints.

## Scientific invariants

This patch does **not** change C-KAM, D-KAM, the objective, workload generators, 200-step horizon, adaptation parameters, metrics, or random seeds. Independent scenario/seed/method items are dispatched to separate processes. Each process regenerates the same deterministic exogenous trace and verifies its SHA-256 before reusing/writing a checkpoint.

## Replication amendment for N=1024

The recommended default is:

- N = 32, 64, 128, 256, 512: all 10 frozen S1 seeds (5001-5010), as originally planned.
- N = 1024: fixed seeds 5001, 5002, 5003, treated as a **descriptive computational stress point**.
- Primary scaling exponents are calculated from N=32..512, where every size retains 10 replications.

This is a computational-feasibility amendment, not a result-driven method change. The original full 10-seed N=1024 plan remains available by setting `FOLLOWUP_1024_SEEDS=5001-5010`.

## Parallelism and memory

The runner computes safe worker counts from logical CPUs and Linux `MemAvailable`, with conservative per-worker memory estimates. You may override:

```bash
FOLLOWUP_WORKERS_SMALL=12
FOLLOWUP_WORKERS_512=6
FOLLOWUP_WORKERS_1024=4
```

Do not use all 32 logical CPUs for N=1024: the dense communication trace is 1.562 GiB before Python/algorithm overhead. Close memory-heavy applications first. The runner warns if swap is already in use.

## Safe stop/resume

Every completed item is checkpointed in the existing `FOLLOWUP_D1_S1_A1_results/checkpoints/` tree. Ctrl-C may discard only the items currently active in worker processes; all completed items remain reusable. Re-run the same command to resume.
