#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -n "${KAM_PYTHON:-}" ]]; then
  PY="$KAM_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python || true)"
fi
CONFIG="$ROOT/examples/paper_confirmatory_C1.json"
OUT="${FOLLOWUP_OUT:-$ROOT/FOLLOWUP_D1_S1_A1_results}"
LOG="$ROOT/FOLLOWUP_D1_S1_A1_parallel_run.log"
RUNNER="$ROOT/run_followup_parallel.py"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

trap 'code=$?; if [[ $code -ne 0 ]]; then echo; echo "Parallel resume stopped with exit code $code."; echo "Do NOT delete $OUT; rerun this same script to resume."; fi; exit $code' EXIT

echo
echo "===== Parallel D1/S1/A1 resume $(date -Iseconds) ====="
if [[ -z "$PY" || ! -x "$PY" ]]; then echo "ERROR: no usable Python interpreter found. Activate .venv or set KAM_PYTHON=/path/to/python."; exit 2; fi
if [[ ! -f "$CONFIG" ]]; then echo "ERROR: missing $CONFIG"; exit 2; fi
if [[ ! -f "$ROOT/run_followup_experiments.py" ]]; then echo "ERROR: original run_followup_experiments.py must remain in this directory"; exit 2; fi

# Scientific source remains C1.1-frozen. The new file changes scheduling only.
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
check_hash() {
  local file="$1" expected="$2" actual
  [[ -f "$file" ]] || { echo "ERROR: missing $file"; exit 2; }
  actual="$(sha256_of "$file")"
  [[ "$actual" == "$expected" ]] || { echo "ERROR: frozen scientific hash mismatch: $file"; echo "expected $expected"; echo "actual   $actual"; exit 2; }
}
check_hash "kam_bench/algorithms.py" "28bd000826ce1f4c086e3d87e129655f02cb800f8b35f57318e6f9a2aee0c098"
check_hash "kam_bench/config.py" "97570e63ea580c5e10bcd9b5f9fe442433d60fb408d93d8ac1c2d1c56a8502a0"
check_hash "kam_bench/metrics.py" "9cc1abbb4eafddb478aee0910151c54a81f84c944ff652acb2884bbf8be6fc88"
check_hash "kam_bench/model.py" "663b6bad00acb8351358c9e9c78abcb808603b574d38992488f0633eda2bbe52"
check_hash "kam_bench/scenarios.py" "861ad59a2b8058dcb65205f6c624be27de8b29cc4369cb34135db192639a010f"
check_hash "kam_bench/simulation.py" "15ec839ce56fa7eb5d0586cd3f8f05062ed77501043a5a9097fd1f57e6c3c1fb"
check_hash "examples/paper_confirmatory_C1.json" "40c83322c1893d4d21cc14f711333974d1335873287a861f6dd6e2b438ff6328"

echo "Frozen scientific source hashes: OK"

echo "===== Regression tests ====="
"$PY" -m pytest -q

C1_ARGS=()
if [[ -f "$ROOT/C1_1_results/run_summary.csv" ]]; then
  C1_ARGS=(--c1-results "$ROOT/C1_1_results")
  echo "Full C1.1 references found and will be reused."
elif [[ -f "$ROOT/paper_results/C1_1/run_summary.csv" ]]; then
  C1_ARGS=(--c1-results "$ROOT/paper_results/C1_1")
  echo "Compact C1.1 run-summary references found and will be reused (raw delay=1 diagnostics remain available in paper_results/followup/D1)."
fi

echo "===== Resource-aware parallel start/resume ====="
echo "Default amendment: N=32..512 retain all 10 seeds; N=1024 uses fixed seeds 5001,5002,5003 as a descriptive stress point."
echo "To insist on the original full N=1024 replication, run:"
echo "  FOLLOWUP_1024_SEEDS=5001-5010 ./run_followup_parallel.sh"
echo "Worker overrides are available as FOLLOWUP_WORKERS_SMALL, FOLLOWUP_WORKERS_512, FOLLOWUP_WORKERS_1024."

"$PY" "$RUNNER" --config "$CONFIG" --out "$OUT" "${C1_ARGS[@]}"

# Copy operational log into finalized result directory if finalization occurred.
cp "$LOG" "$OUT/FOLLOWUP_D1_S1_A1_parallel_run.log" || true

trap - EXIT
