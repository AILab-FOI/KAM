#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="${KAM_PARMETIS_DEPS:-$ROOT/.p2_deps}"
SRC="$DEPS/src"
PREFIX="$DEPS/local"
JOBS="${KAM_BUILD_JOBS:-$(nproc 2>/dev/null || echo 2)}"
[[ "$JOBS" -gt 8 ]] && JOBS=8

GKLIB_COMMIT="3b7d61b9f885063c89901f3901fb4426f9cfb58f"
METIS_COMMIT="272d4a91c5f66c92327493339a476c553ebf1f5d"
PARMETIS_COMMIT="7436de671433c248e4203e1640a65b778845e6fc"

for cmd in git cmake make gcc mpicc mpicxx; do
  command -v "$cmd" >/dev/null || { echo "ERROR: missing required command: $cmd"; exit 2; }
done

clone_at() {
  local url="$1" dir="$2" commit="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$url" "$dir"
  fi
  git -C "$dir" fetch --all --tags --prune
  git -C "$dir" checkout --detach "$commit"
}

mkdir -p "$SRC" "$PREFIX"
clone_at https://github.com/KarypisLab/GKlib.git "$SRC/GKlib" "$GKLIB_COMMIT"
clone_at https://github.com/KarypisLab/METIS.git "$SRC/METIS" "$METIS_COMMIT"
clone_at https://github.com/KarypisLab/ParMETIS.git "$SRC/ParMETIS" "$PARMETIS_COMMIT"

echo "===== Building GKlib ====="
(
  cd "$SRC/GKlib"
  make distclean >/dev/null 2>&1 || true
  make config prefix="$PREFIX" cc=gcc
  make -j"$JOBS"
  make install
)

echo "===== Building METIS ====="
(
  cd "$SRC/METIS"
  make distclean >/dev/null 2>&1 || true
  make config prefix="$PREFIX" gklib_path="$PREFIX" cc=gcc
  make -j"$JOBS"
  make install
)

echo "===== Building ParMETIS ====="
(
  cd "$SRC/ParMETIS"
  make distclean >/dev/null 2>&1 || true
  make config prefix="$PREFIX" gklib_path="$PREFIX" metis_path="$PREFIX" cc="$(command -v mpicc)"
  make -j"$JOBS"
  make install
)

echo "===== Building KAM-Bench AdaptiveRepart driver ====="
make -C "$ROOT/external" -f Makefile.parmetis clean
make -C "$ROOT/external" -f Makefile.parmetis \
  CXX="$(command -v mpicxx)" \
  PARMETIS_PREFIX="$PREFIX" \
  METIS_PREFIX="$PREFIX" \
  GKLIB_PREFIX="$PREFIX"

DRIVER="$ROOT/external/parmetis_adaptive_driver"
[[ -x "$DRIVER" ]] || { echo "ERROR: driver was not created: $DRIVER"; exit 2; }

echo
echo "Build complete."
echo "GKlib:   $(git -C "$SRC/GKlib" rev-parse HEAD)"
echo "METIS:   $(git -C "$SRC/METIS" rev-parse HEAD)"
echo "ParMETIS:$(git -C "$SRC/ParMETIS" rev-parse HEAD)"
echo "Driver:  $DRIVER"
echo "SHA-256: $(sha256sum "$DRIVER" | awk '{print $1}')"
echo
echo "The C1.1 configuration invokes the driver through mpirun from the repository root."
