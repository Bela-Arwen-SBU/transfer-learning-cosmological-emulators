#!/bin/bash
# =============================================================================
# run_chain_plots.sh
# Generates parameter chains ONLY (--chain 1, no data vectors computed)
# for triangle plot comparison: original vs modified (maxcorr=0.15, T=150).
#
# Usage:
#   source start_cocoa.sh   (sets $ROOTDIR and activates conda env)
#   bash /home/grads/tmp/roman_cs_baseline/run_chain_plots.sh
# =============================================================================

set -e

EXPECTED_ROOTDIR="/home/grads/data/bela/cocoa/Cocoa"
if [ "$ROOTDIR" != "$EXPECTED_ROOTDIR" ]; then
    echo "ERROR: ROOTDIR='$ROOTDIR', expected '$EXPECTED_ROOTDIR'"
    exit 1
fi

YAML_SRC="/home/grads/tmp/roman_cs_baseline/roman_emulator_cs_mead.yaml"
GENERATOR="$ROOTDIR/external_modules/code/emulators/emultrf/emultraining/dataset_generator_lensing.py"
PYTHON="$ROOTDIR/.local/bin/python"
FILEROOT_REL="emulators/roman_cs_mead"
FILEROOT_ABS="$ROOTDIR/projects/roman_real/$FILEROOT_REL"
CHAINS_DIR="$ROOTDIR/projects/roman_real/chains"
PYTHONPATH_RUN="$ROOTDIR/cobaya:$PYTHONPATH"

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
# Create fileroot dir and write patched YAML
# =============================================================================
echo "=== Creating fileroot and patching YAML ==="
mkdir -p "$FILEROOT_ABS"
YAML_DST="$FILEROOT_ABS/roman_emulator_cs_mead_chain.yaml"

PYTHONPATH="$PYTHONPATH_RUN" $PYTHON - "$YAML_SRC" "$YAML_DST" << 'PYEOF'
import sys, yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    info = yaml.safe_load(f)

ta = info['train_args']

# 1. rename parameter_covmat_file -> params_covmat_file with correct relative path
if 'parameter_covmat_file' in ta:
    del ta['parameter_covmat_file']
ta['params_covmat_file'] = '../../data/w0wa_fisher_covmat_LCDMcenter.txt'

# 2. add probe
ta['probe'] = 'cs'

# 3. add top-level ord (copied from cosmic_shear.extra_args.ord)
ta['ord'] = ta['cosmic_shear']['extra_args']['ord']

# 4. swap ones.dataset -> example1.dataset (ones.dataset causes non-pos-def cov)
info['likelihood']['roman_real.combo_3x2pt']['data_file'] = 'example1.dataset'

# 5. fix w0pwa and w as fixed LCDM values so they are not in sampled params
for p, val in [('w0pwa', -1.0), ('w', -1.0)]:
    if p in info['params'] and isinstance(info['params'][p], dict) and 'prior' in info['params'][p]:
        info['params'][p] = {'value': val}

with open(dst, 'w') as f:
    yaml.dump(info, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"  Written: {dst}")
PYEOF

echo "  Patched YAML: $YAML_DST"
echo "  Verifying patches:"
grep -n "params_covmat_file\|probe:\|data_file:" "$YAML_DST"
[ -f "$FILEROOT_ABS/../../data/w0wa_fisher_covmat_LCDMcenter.txt" ] \
    || { echo "ERROR: Fisher covmat not reachable via relative path"; exit 1; }
echo "  Fisher covmat reachable from fileroot ✓"
echo ""

cd "$ROOTDIR"
echo "Working directory: $(pwd)"
echo ""

# =============================================================================
# Case 1: Original (T=1, maxcorr=1.0)
# =============================================================================
echo "=== CASE 1: Original (T=1, maxcorr=1.0) ==="
PYTHONPATH="$PYTHONPATH_RUN" mpirun -n 1 --oversubscribe \
    $PYTHON "$GENERATOR" \
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
# Case 2: Modified (T=150, maxcorr=0.15)
# =============================================================================
echo "=== CASE 2: Modified (T=150, maxcorr=0.15) ==="
PYTHONPATH="$PYTHONPATH_RUN" mpirun -n 1 --oversubscribe \
    $PYTHON "$GENERATOR" \
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
