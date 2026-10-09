#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="${C1_1_OUT:-$ROOT/C1_1_results}"
LOG="$ROOT/C1_1_run.log"
if [[ -n "${KAM_PYTHON:-}" ]]; then
  PY="$KAM_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python || true)"
fi
DRIVER="$ROOT/external/parmetis_adaptive_driver"
CONFIG="$ROOT/examples/paper_confirmatory_C1.json"

exec > >(tee -a "$LOG") 2>&1
printf '\n===== C1.1 attempt %s =====\n' "$(date -Iseconds)"
echo "===== C1.1 preflight ====="
if [[ -z "$PY" || ! -x "$PY" ]]; then echo "ERROR: no usable Python interpreter found. Activate .venv or set KAM_PYTHON=/path/to/python."; exit 2; fi
if [[ ! -x "$DRIVER" ]]; then echo "ERROR: missing executable $DRIVER."; exit 2; fi

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
check_hash() {
  local file="$1" expected="$2" actual
  if [[ ! -f "$file" ]]; then echo "ERROR: missing frozen file $file"; exit 2; fi
  actual="$(sha256_of "$file")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: C1.1 hash mismatch for $file"
    echo " expected: $expected"
    echo " actual:   $actual"
    exit 2
  fi
}

check_hash "kam_bench/__init__.py" "617daefcd9f9b1b03b0465fa546431fca717c60539374492186e45c171d72d05"
check_hash "kam_bench/algorithms.py" "28bd000826ce1f4c086e3d87e129655f02cb800f8b35f57318e6f9a2aee0c098"
check_hash "kam_bench/cli.py" "5b3647ca6a2c944c0ae7174d31717fe1bfb9daf6d6770f60f8781fae1f3f3ff6"
check_hash "kam_bench/config.py" "97570e63ea580c5e10bcd9b5f9fe442433d60fb408d93d8ac1c2d1c56a8502a0"
check_hash "kam_bench/experiments.py" "ac62d2214b2a71bb6a9f3fa12b7c7453520c10940b5918e260cbf47272e26d69"
check_hash "kam_bench/external_hpc.py" "76a9a79604a614d8c794b8ccd7a89f6719e46197933649d13f0ac922705e5a30"
check_hash "kam_bench/metrics.py" "9cc1abbb4eafddb478aee0910151c54a81f84c944ff652acb2884bbf8be6fc88"
check_hash "kam_bench/model.py" "663b6bad00acb8351358c9e9c78abcb808603b574d38992488f0633eda2bbe52"
check_hash "kam_bench/scenarios.py" "861ad59a2b8058dcb65205f6c624be27de8b29cc4369cb34135db192639a010f"
check_hash "kam_bench/simulation.py" "15ec839ce56fa7eb5d0586cd3f8f05062ed77501043a5a9097fd1f57e6c3c1fb"
check_hash "kam_bench/statistics.py" "1a61f459ae1776d45b68632383bfcfcc4551f4f34b41212d38097e7a1fb4317d"
check_hash "kam_bench/trace_io.py" "56157e2066aa57f162ebfaccecfe51dea9232b2a9f3c471aa4a6ab85b6e8dc8e"
check_hash "examples/paper_confirmatory_C1.json" "40c83322c1893d4d21cc14f711333974d1335873287a861f6dd6e2b438ff6328"
check_hash "reproducibility/C1_SEED_MANIFEST.json" "950eed28001cbe340f23f444501414e9182e33141e1cdb4dc8ad7e9b788a43e1"
check_hash "reproducibility/C1_FROZEN_COMPARATOR_PARAMETERS.json" "3b19b18e2bd3580fbe54abfff0a4c68b494a325d5a4739676399f5d3d303bd8b"
FROZEN_DRIVER_SHA="59fbdbad198313aa7636c92ccb2fc93a62c5905de39b4a3a87127ac007352fce"
actual_driver_sha="$(sha256_of "$DRIVER")"
if [[ "$actual_driver_sha" == "$FROZEN_DRIVER_SHA" ]]; then
  echo "ParMETIS driver binary hash: exact paper build ($actual_driver_sha)"
else
  echo "WARNING: ParMETIS driver binary differs from the original paper build."
  echo "         original: $FROZEN_DRIVER_SHA"
  echo "         current:  $actual_driver_sha"
  echo "         This is expected after recompilation on another machine; exact source provenance is recorded in reproducibility/C1_EXTERNAL_PROVENANCE.json."
fi
check_hash "run_C1_checkpointed.py" "5b7f0206749f484a82bce4606f497b63e1e449bc413d442acb2798705506d6fb"
check_hash "reproducibility/C1_1_FREEZE_RECORD.md" "1f45aa2497ade3c9622f9a5bd09fc0d4a2ed829676edc304beee1cc7198325e3"

echo "Frozen scientific config + C1.1 source hashes: OK"

"$PY" - <<'PYCHECK'
import sys
import numpy, pandas, scipy, networkx, igraph
expected={'python':(3,10),'numpy':'2.2.6','pandas':'2.3.3','scipy':'1.15.3','networkx':'3.4.2','igraph':'1.0.0'}
actual={'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'networkx':networkx.__version__,'igraph':igraph.__version__}
if sys.version_info[:2] != expected['python']:
    raise SystemExit(f"Python version mismatch: {sys.version_info[:3]} != 3.10.x")
for k,v in actual.items():
    if v != expected[k]: raise SystemExit(f"{k} version mismatch: {v} != {expected[k]}")
print('Python/backend versions: OK', actual)
PYCHECK

check_commit() {
  local dir="$1" expected="$2" label="$3"
  if [[ -d "$dir/.git" ]]; then
    local actual="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$actual" != "$expected" ]]; then echo "ERROR: $label commit mismatch: $actual != $expected"; exit 2; fi
    echo "$label commit: OK ($actual)"
  else
    echo "WARNING: $label git repository not present; driver binary hash remains authoritative."
  fi
}
check_commit "${KAM_PARMETIS_SOURCE_ROOT:-$ROOT/.p2_deps/src}/GKlib" "3b7d61b9f885063c89901f3901fb4426f9cfb58f" "GKlib"
check_commit "${KAM_PARMETIS_SOURCE_ROOT:-$ROOT/.p2_deps/src}/METIS" "272d4a91c5f66c92327493339a476c553ebf1f5d" "METIS"
check_commit "${KAM_PARMETIS_SOURCE_ROOT:-$ROOT/.p2_deps/src}/ParMETIS" "7436de671433c248e4203e1640a65b778845e6fc" "ParMETIS"

echo "===== C1.1 regression tests ====="
"$PY" -m pytest -q

echo "===== C1.1 checkpointed confirmatory run ====="
"$PY" run_C1_checkpointed.py --config "$CONFIG" --out "$OUT"

mkdir -p "$OUT/c1_1_freeze"
cp "$CONFIG" "$OUT/c1_1_freeze/"
cp reproducibility/C1_1_FREEZE_RECORD.md \
   reproducibility/C1_SEED_MANIFEST.json \
   reproducibility/C1_FROZEN_COMPARATOR_PARAMETERS.json \
   reproducibility/C1_1_RELEASE_HASHES.txt \
   reproducibility/C1_EXTERNAL_PROVENANCE.json \
   reproducibility/paper_confirmatory_C1.json \
   "$OUT/c1_1_freeze/"
cp "$LOG" "$OUT/C1_1_run.log"

ZIP="$ROOT/C1_1_RESULTS_RETURN_ME.zip"
rm -f "$ZIP"
# Checkpoints remain on the machine for resume/audit but are duplicate data, so
# omit them from the transfer archive to keep upload size reasonable.
( cd "$(dirname "$OUT")" && zip -qr "$ZIP" "$(basename "$OUT")" -x "$(basename "$OUT")/checkpoints/*" )

echo "===== C1.1 finished ====="
echo "Upload this file back to ChatGPT:"
echo "  $ZIP"
