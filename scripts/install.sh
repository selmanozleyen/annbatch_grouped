#!/usr/bin/env bash
# Install annbatch_grouped and (optionally) a specific annbatch branch.
#
# Usage:
#   ./scripts/install.sh                  # install from paths.conf settings
#   ./scripts/install.sh main             # override: install annbatch@main
#   ./scripts/install.sh feat/my-branch   # override: install annbatch@feat/my-branch
set -euo pipefail
cd "$(dirname "$0")/.."

# ---------- Parse paths.conf ----------
parse_conf() {
    local key="$1" default="$2"
    if [[ -f paths.conf ]]; then
        val=$(grep -E "^\s*${key}\s*=" paths.conf | head -1 | sed 's/^[^=]*=\s*//' | xargs)
        echo "${val:-$default}"
    else
        echo "$default"
    fi
}

ANNBATCH_REPO=$(parse_conf ANNBATCH_REPO "https://github.com/scverse/annbatch.git")
ANNBATCH_REF=$(parse_conf ANNBATCH_REF "")

# CLI arg overrides paths.conf
if [[ ${1:-} != "" ]]; then
    ANNBATCH_REF="$1"
fi

# ---------- Install this package ----------
echo "=== Installing annbatch_grouped (editable) ==="
pip install -e ".[viz]"

# ---------- Install zarrs for fast zarr I/O ----------
echo ""
echo "=== Ensuring zarrs is installed ==="
pip install zarrs

# ---------- Install annbatch from git if ref is set ----------
if [[ -n "$ANNBATCH_REF" ]]; then
    echo ""
    echo "=== Installing annbatch from ${ANNBATCH_REPO}@${ANNBATCH_REF} ==="
    pip install "annbatch @ git+${ANNBATCH_REPO}@${ANNBATCH_REF}" --force-reinstall --no-deps
    echo ""
    echo "Installed annbatch ref: ${ANNBATCH_REF}"
else
    echo ""
    echo "=== ANNBATCH_REF not set, keeping current annbatch install ==="
fi

echo ""
python -c "import annbatch; print(f'annbatch version: {annbatch.__version__}')" 2>/dev/null || true
python -c "import zarrs; print('zarrs: OK')" 2>/dev/null || echo "WARNING: zarrs not available, using default zarr codec pipeline"
python -c "import annbatch_grouped; print('annbatch_grouped: OK')"
echo "Done."
