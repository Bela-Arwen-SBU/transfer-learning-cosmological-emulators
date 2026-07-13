#!/bin/bash
# =============================================================================
# run_chain_plots.sh
# Generates parameter chains ONLY (--chain 1, no data vectors computed)
# for triangle plot comparison: original vs modified (maxcorr=0.15, T=150).
#
# Usage:
#   source start_cocoa.sh   (sets $ROOTDIR and activates conda env)
#   bash /home/grads/tmp/roman_cs_baseline/run_chain_plots.sh
#
# Output chain files go to:
#   $ROOTDIR/projects/roman_real/chains/chain_orig_params_cs_1.*
#   $ROOTDIR/projects/roman_real/chains/chain_mod_params_cs_150.*
# =============================================================================

set -e

# =============================================================================
# Confirmed paths
# =============================================================================
EXPECTED_ROOTDIR="/home/grads/data/bela/cocoa/Cocoa"
if [ "$ROOTDIR" != "$EXPECTED_ROOTDIR" ]; then
    echo "ERROR: ROOTDIR='$ROOTDIR', expected '$EXPECTED_ROOTDIR'"
    echo "Did you run 'source start_cocoa.sh'?"
    exit 1
fi

YAML_SRC="/home/grads/tmp/roman_cs_baseline/roman_emulator_cs_mead.yaml"
GENERATOR="$ROOTDIR/external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py"
FILEROOT_REL="emulators/roman_cs_mead"
FILEROOT_ABS="$ROOTDIR/projects/roman_real/$FILEROOT_REL"
CHAINS_DIR="$ROOTDIR/projects/roman_real/chains"

# =============================================================================
# Validate
# =============================================================================
echo "=== Validating paths ==="
[ -f "$YAML_SRC" ]   || { echo "ERROR: YAML not found: $YAML_SRC"; exit 1; }
[ -f "$GENERATOR" ]  || { echo "ERROR: Generator not found: $GENERATOR"; exit 1; }
[ -d "$CHAINS_DIR" ] || { echo "ERROR: chains dir not found: $CHAINS_DIR"; exit 1; }
[ -f "$ROOTDIR/projects/roman_real/data/w0wa_fisher_covmat_LCDMcenter.txt" ] \
    || { echo "ERROR: Fisher covmat not found"; exit 1; }
echo "  All paths OK ✓"
echo ""

# =============================================================================
# Step 1: Create fileroot dir and write patched YAML
# Three changes vs original:
#   1. rename parameter_covmat_file -> params_covmat_file
#   2. fix path to be relative to fileroot: ../../data/...
#   3. add probe: cs under train_args
# =============================================================================
echo "=== Creating fileroot and patching YAML ==="
mkdir -p "$FILEROOT_ABS"

YAML_DST="$FILEROOT_ABS/roman_emulator_cs_mead_chain.yaml"

sed 's|parameter_covmat_file:.*|params_covmat_file: '"'"'../../data/w0wa_fisher_covmat_LCDMcenter.txt'"'"'|' \
    "$YAML_SRC" \
    | sed '/^train_args:/a\  probe: cs' \
    > "$YAML_DST"

echo "  Patched YAML: $YAML_DST"
echo "  Verifying patches:"
grep -n "params_covmat_file\|^  probe:" "$YAML_DST"

# Verify relative path resolves correctly
[ -f "$FILEROOT_ABS/../../data/w0wa_fisher_covmat_LCDMcenter.txt" ] \
    || { echo "ERROR: Fisher covmat not reachable via relative path"; exit 1; }
echo "  Fisher covmat reachable from fileroot ✓"
echo ""

# =============================================================================
# Case 1: Original — no modification (T=1, maxcorr=1.0)
# =============================================================================
echo "=== CASE 1: Original (T=1, maxcorr=1.0) ==="
mpirun -n 1 --oversubscribe \
    python "$GENERATOR" \
        --root     "projects/roman_real" \
        --fileroot "$FILEROOT_REL" \
        --yaml     "roman_emulator_cs_mead_chain.yaml" \
        --nparams  10000 \
        --datavsfile "chain_orig_dvs" \
        --paramfile  "chain_orig_params" \
        --failfile   "chain_orig_failed" \
        --chain   1 \
        --unif    0 \
        --temp    1 \
        --maxcorr 1.0 \
        --freqchk 5000 \
        --loadchk 0 \
        --append  0

echo "Case 1 output:"
ls "$CHAINS_DIR"/chain_orig_params_cs_1.* 2>/dev/null || echo "WARNING: no output files found"
echo ""

# =============================================================================
# Case 2: Modified — maxcorr=0.15, T=150
# =============================================================================
echo "=== CASE 2: Modified (T=150, maxcorr=0.15) ==="
mpirun -n 1 --oversubscribe \
    python "$GENERATOR" \
        --root     "projects/roman_real" \
        --fileroot "$FILEROOT_REL" \
        --yaml     "roman_emulator_cs_mead_chain.yaml" \
        --nparams  10000 \
        --datavsfile "chain_mod_dvs" \
        --paramfile  "chain_mod_params" \
        --failfile   "chain_mod_failed" \
        --chain   1 \
        --unif    0 \
        --temp    150 \
        --maxcorr 0.15 \
        --freqchk 5000 \
        --loadchk 0 \
        --append  0

echo "Case 2 output:"
ls "$CHAINS_DIR"/chain_mod_params_cs_150.* 2>/dev/null || echo "WARNING: no output files found"
echo ""

echo "=== Done! ==="
echo "GetDist chain roots:"
echo "  Original : $CHAINS_DIR/chain_orig_params_cs_1"
echo "  Modified : $CHAINS_DIR/chain_mod_params_cs_150"
echo "Now open triangle_plot_comparison.ipynb"
