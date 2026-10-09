#!/usr/bin/env python3
"""Test-only external partitioner for exercising the KAM-Bench process adapter.

This is NOT a scientific baseline. It greedily assigns heavy vertices to the
currently lightest capacity-normalized partition and writes protocol-compliant
output. The confirmatory configuration must never point to this helper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MAGIC = "KAM_BENCH_PARTITION_V1"
OUT = "KAM_BENCH_PARTITION_RESULT_V1"


def read_problem(path: Path):
    tok = path.read_text().split()
    if not tok or tok[0] != MAGIC:
        raise RuntimeError("bad input magic")
    pos = 1
    d = {}
    while pos < len(tok):
        key = tok[pos]; pos += 1
        if key in {"n", "nparts"}:
            d[key] = int(tok[pos]); pos += 1
        elif key in {"imbalance_tolerance", "ipc2redist"}:
            d[key] = float(tok[pos]); pos += 1
        elif key in {"capacity_shares"}:
            k = d["nparts"]
            d[key] = [float(x) for x in tok[pos:pos+k]]; pos += k
        elif key in {"current_part", "vertex_weight", "migration_size"}:
            n = d["n"]
            d[key] = [int(x) for x in tok[pos:pos+n]]; pos += n
        elif key == "edges":
            m = int(tok[pos]); pos += 1
            d["edges"] = []
            for _ in range(m):
                i, j, w = map(int, tok[pos:pos+3]); pos += 3
                d["edges"].append((i, j, w))
        else:
            raise RuntimeError(f"unknown key {key}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    p = read_problem(a.input)
    n, k = p["n"], p["nparts"]
    caps = p["capacity_shares"]
    weights = p["vertex_weight"]
    loads = [0.0] * k
    part = [-1] * n
    # Seed every partition so the adapter's non-empty-partition invariant holds.
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    for u, i in enumerate(order[:k]):
        part[i] = u
        loads[u] += weights[i]
    for i in order[k:]:
        u = min(range(k), key=lambda z: loads[z] / max(caps[z], 1e-12))
        part[i] = u
        loads[u] += weights[i]
    a.output.write_text(
        OUT + "\n" + f"n {n}\nedgecut -1\npart " + " ".join(map(str, part)) + "\n"
    )


if __name__ == "__main__":
    main()
