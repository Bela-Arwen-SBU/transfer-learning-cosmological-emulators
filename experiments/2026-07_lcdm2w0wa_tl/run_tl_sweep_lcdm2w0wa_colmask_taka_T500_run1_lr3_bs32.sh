#!/bin/bash
# =============================================================================
# TL N_train sweep: LCDM -> w0wa, Takahashi, T500 run1 (COLUMN-MASKED INPUT)
# Locks the 15 pretrained input columns via MASK_INPUT_COLS; only the new
# (w, w0pwa) columns plus the strategy's unfrozen layers train.
# Base: LCDM taka run4 N500k checkpoint (bs32), input layer padded 15 -> 17
#   with zero columns appended for (w, w0pwa); see pad_lcdm_to_w0wa.py
# Target: w0wa taka run1 (train T500; valid/test T250 cut2p5pct)
# Loops over N_train = 10k, 25k, 50k, 100k
# FREEZE_STRATEGY must be set by the submit script via environment variable
# =============================================================================
set -e

if [ -z "${FREEZE_STRATEGY}" ]; then
    echo "ERROR: FREEZE_STRATEGY not set."
    exit 1
fi

echo "========================================"
echo "TL N_train SWEEP - LCDM->w0wa COLMASK Taka freeze=${FREEZE_STRATEGY} LR=1e-3 BS=32"
echo "Started: $(date)"
echo "Host:    $(hostname)"
echo "========================================"

COCOA="/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa"
TRAIN_SCRIPT="${COCOA}/train_emulator.py"
BASE_YAML="${COCOA}/projects/roman_real/roman_emulator_cs_taka_nvwulf_w0wa_T500.yaml"
DV_DIR="${COCOA}/projects/roman_real/dvs/w0wa/taka"
# Padded 17-input version of the LCDM base (produced by pad_lcdm_to_w0wa.py).
# bs32 base chosen deliberately: same base as the taka2taka TL control sweep.
PRETRAINED_MODEL="${COCOA}/projects/roman_real/chains/taka_scratch_T500_run4_clip90_intdim362_bs32/models/N500000/roman_cs_lcdm_taka_resmlp_T500_padded17.pt"

OUTPUT_BASE="${COCOA}/projects/roman_real/chains/taka_lcdm2w0wa_tl_colmask_T500_run1_clip90_intdim362"
MODEL_DIR="${OUTPUT_BASE}/models"
METRICS_DIR="${OUTPUT_BASE}/metrics"
YAML_DIR="${OUTPUT_BASE}/yamls"
LOG_DIR="${OUTPUT_BASE}/logs"

PROBE="cosmic_shear"
LEARNING_RATE=1e-3
BATCH_SIZE=32
EPOCHS=1500

# Note: the npy stems carry the historical "_parameters_" naming; the txt
# files carry a second "_parameters".
TRAIN_DV="roman_real_w0wa_b_taka_train_parameters_T500_run1.npy"
TRAIN_PAR="roman_real_w0wa_b_taka_train_parameters_T500_run1_parameters.txt"
VALID_DV="roman_real_w0wa_b_taka_valid_parameters_T250_run1_cut2p5pct.npy"
VALID_PAR="roman_real_w0wa_b_taka_valid_parameters_T250_run1_parameters_cut2p5pct.txt"
TEST_DV="roman_real_w0wa_b_taka_test_parameters_T250_run1_cut2p5pct.npy"
TEST_PAR="roman_real_w0wa_b_taka_test_parameters_T250_run1_parameters_cut2p5pct.txt"

SIZES=(10000 25000 50000 100000)

echo ""
echo "=== Pre-flight Checks ==="
check_file() {
    if [ ! -f "$1" ]; then echo "ERROR: Required file not found: $1"; exit 1; fi
    echo "  ✓ $1"
}
check_file "${TRAIN_SCRIPT}"
grep -q "MASK_INPUT_COLS" "${TRAIN_SCRIPT}" || { echo "ERROR: train_emulator.py is missing the MASK_INPUT_COLS patch"; exit 1; }
check_file "${BASE_YAML}"
check_file "${PRETRAINED_MODEL}"
check_file "${PRETRAINED_MODEL%.pt}.h5"
check_file "${DV_DIR}/${TRAIN_DV}"
check_file "${DV_DIR}/${TRAIN_PAR}"
check_file "${DV_DIR}/${VALID_DV}"
check_file "${DV_DIR}/${VALID_PAR}"
check_file "${DV_DIR}/${TEST_DV}"
check_file "${DV_DIR}/${TEST_PAR}"

# Guard: refuse to start from an un-padded (15-input) checkpoint.
python - "${PRETRAINED_MODEL}" <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu')
w = sd['model.0.weight']
assert w.shape[1] == 17, f"pretrained input dim is {w.shape[1]}, expected 17 (padded) — run pad_lcdm_to_w0wa.py first"
print(f"  ✓ pretrained model.0.weight {tuple(w.shape)} (padded)")
PYEOF

echo ""
echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "freeze_strategy : ${FREEZE_STRATEGY}"
echo "LR              : ${LEARNING_RATE}"
echo "BS              : ${BATCH_SIZE}"
echo "Epochs          : ${EPOCHS}"
echo "N_train sizes   : ${SIZES[@]}"
echo "Pretrained      : ${PRETRAINED_MODEL}"
echo "Output          : ${OUTPUT_BASE}"
echo "========================================"

SUCCESS_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)

for idx in "${!SIZES[@]}"; do
    size="${SIZES[$idx]}"
    exp_num=$((idx + 1))

    echo ""
    echo "========================================================================"
    echo "EXPERIMENT ${exp_num}/${#SIZES[@]}: freeze=${FREEZE_STRATEGY}, N_train=${size}"
    echo "========================================================================"
    echo "Started: $(date)"

    mkdir -p "${MODEL_DIR}/N${size}" "${METRICS_DIR}" "${YAML_DIR}" "${LOG_DIR}"

    MODEL_NAME="roman_cs_w0wa_taka_tl_lcdm_resmlp_T500"
    MODEL_SUBDIR="${MODEL_DIR}/N${size}"
    TEMP_YAML="${YAML_DIR}/lcdm2w0wa_${FREEZE_STRATEGY}_N${size}.yaml"

    echo "Generating YAML for N_train=${size}..."
    python3 - << PYEOF
import yaml
with open('${BASE_YAML}', 'r') as f:
    config = yaml.safe_load(f)
ta = config['train_args']
cs = ta['${PROBE}']['extra_args']
cs['extrapar'] = [{'MLA': 'MLP', 'INT_DIM_RES': 362}]
cs['file']  = ['${MODEL_SUBDIR}/${MODEL_NAME}.pt']
cs['extra'] = ['${MODEL_SUBDIR}/${MODEL_NAME}.h5']
ta['training_data_path']     = '/'
ta['train_datavectors_file'] = '${DV_DIR}/${TRAIN_DV}'
ta['train_parameters_file']  = '${DV_DIR}/${TRAIN_PAR}'
ta['n_valid'] = 10000
ta['t_valid'] = 250
ta['valid_datavectors_file'] = '${DV_DIR}/${VALID_DV}'
ta['valid_parameters_file']  = '${DV_DIR}/${VALID_PAR}'
ta['n_test'] = 10000
ta['t_test'] = 250
ta['test_datavectors_file']  = '${DV_DIR}/${TEST_DV}'
ta['test_parameters_file']   = '${DV_DIR}/${TEST_PAR}'
with open('${TEMP_YAML}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('YAML written: ${TEMP_YAML}')
PYEOF

    if [ $? -ne 0 ]; then
        echo "✗ YAML generation failed for size ${size}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "Starting training..."
    TRAIN_START=$(date +%s)

    MASK_INPUT_COLS=15 python "${TRAIN_SCRIPT}" \
        --yaml "${TEMP_YAML}" \
        --probe "${PROBE}" \
        --learning_rate "${LEARNING_RATE}" \
        --batchsize "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --ntrain "${size}" \
        --save_losses True \
        --save_testing_metrics True \
        --transfer_learning True \
        --pretrained_model "${PRETRAINED_MODEL}" \
        --freeze_strategy "${FREEZE_STRATEGY}" \
        2>&1 | tee "${LOG_DIR}/train_${FREEZE_STRATEGY}_N${size}.log"

    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    TRAIN_HOURS=$((TRAIN_DURATION / 3600))
    TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed in ${TRAIN_HOURS}h ${TRAIN_MINUTES}m"

        if [ -f "${MODEL_SUBDIR}/${MODEL_NAME}_metrics.txt" ]; then
            mv "${MODEL_SUBDIR}/${MODEL_NAME}_metrics.txt" "${METRICS_DIR}/metrics_${FREEZE_STRATEGY}_N${size}.txt"
            echo "  ✓ Saved metrics: ${METRICS_DIR}/metrics_${FREEZE_STRATEGY}_N${size}.txt"
        else
            echo "  ⚠ Warning: metrics file not found"
        fi

        if [ -f "${MODEL_SUBDIR}/${MODEL_NAME}.pt" ] && [ -f "${MODEL_SUBDIR}/${MODEL_NAME}.h5" ]; then
            MODEL_SIZE=$(du -h "${MODEL_SUBDIR}/${MODEL_NAME}.pt" | cut -f1)
            echo "  ✓ Model saved (${MODEL_SIZE}): ${MODEL_NAME}.pt"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ ERROR: Model files not found"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "✗ TRAINING FAILED for N_train=${size} (exit code: ${TRAIN_EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo "------------------------------------------------------------------------"

    COMPLETED=$((SUCCESS_COUNT + FAIL_COUNT))
    REMAINING=$((${#SIZES[@]} - COMPLETED))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $COMPLETED -gt 0 ]; then
        AVG_TIME=$((ELAPSED / COMPLETED))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_HOURS=$((EST_REMAINING / 3600))
        EST_MINUTES=$(((EST_REMAINING % 3600) / 60))
        echo "Progress: ${COMPLETED}/${#SIZES[@]} done | ~${EST_HOURS}h ${EST_MINUTES}m remaining"
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "SWEEP COMPLETE: freeze=${FREEZE_STRATEGY}"
echo "========================================"
echo "Finished: $(date)"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "Successful: ${SUCCESS_COUNT}/${#SIZES[@]}"
echo "Failed: ${FAIL_COUNT}/${#SIZES[@]}"
echo "========================================"

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
