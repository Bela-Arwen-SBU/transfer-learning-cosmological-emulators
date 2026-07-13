#!/bin/bash
set -e
echo "========================================"
echo "TL STRATEGY SWEEP - Taka T500 -> Mead T500 (ResMLP)"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "========================================"

PROBE="cosmic_shear"
LEARNING_RATE=1e-5
EPOCHS=1000
BATCH_SIZE=128
N_TRAIN=100000

BASE_YAML="/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/roman_emulator_cs_mead_nvwulf_T500.yaml"
PRETRAINED_MODEL="/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/taka_scratch_T500/models/roman_cs_lcdm_taka_resmlp_T500_N500000.pt"
TRAIN_EMULATOR="/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/train_emulator.py"

OUTPUT_BASE="/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/projects/roman_real/chains/mead_tl_from_taka_T500"
MODEL_DIR="${OUTPUT_BASE}/models"
METRICS_DIR="${OUTPUT_BASE}/metrics"
YAML_DIR="${OUTPUT_BASE}/yamls"

STRATEGIES=(none early_1 early_2 late_1 late_2)

echo ""
echo "=== Pre-flight Checks ==="
if [ ! -f "${BASE_YAML}" ]; then
    echo "ERROR: Base YAML not found: ${BASE_YAML}"
    exit 1
fi
echo "✓ Base YAML found"

if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: Pretrained model not found: ${PRETRAINED_MODEL}"
    exit 1
fi
echo "✓ Pretrained model found"

if [ ! -f "${PRETRAINED_MODEL%.pt}.h5" ]; then
    echo "ERROR: Pretrained .h5 not found: ${PRETRAINED_MODEL%.pt}.h5"
    exit 1
fi
echo "✓ Pretrained .h5 found"

echo ""
echo "=== Creating output directories ==="
mkdir -p "${MODEL_DIR}" "${METRICS_DIR}" "${YAML_DIR}"
echo "✓ Output directories created under ${OUTPUT_BASE}"

echo ""
echo "========================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================"
echo "Architecture:      ResMLP (MLA=MLP, INT_DIM_RES=512)"
echo "Probe:             ${PROBE}"
echo "Source model:      Taka T500 N=500k"
echo "Target:            Mead T500"
echo "N_train:           ${N_TRAIN}"
echo "Learning Rate:     ${LEARNING_RATE}"
echo "Epochs:            ${EPOCHS}"
echo "Batch Size:        ${BATCH_SIZE}"
echo "Strategies:        ${STRATEGIES[@]}"
echo "T_train: 500 | T_valid/test: 250"
echo "========================================"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)

for idx in "${!STRATEGIES[@]}"; do
    strategy="${STRATEGIES[$idx]}"
    exp_num=$((idx + 1))

    echo ""
    echo "========================================================================"
    echo "EXPERIMENT ${exp_num}/${#STRATEGIES[@]}: freeze_strategy = ${strategy}"
    echo "========================================================================"
    echo "Started: $(date)"

    MODEL_NAME="roman_cs_lcdm_mead_resmlp_tl_fromTaka_T500_N${N_TRAIN}_fs_${strategy}"
    TEMP_YAML="${YAML_DIR}/mead_tl_fromTaka_T500_N${N_TRAIN}_fs_${strategy}.yaml"

    echo "Generating YAML..."
    python3 - << EOF
import yaml

with open('${BASE_YAML}', 'r') as f:
    config = yaml.safe_load(f)

# Architecture
config['train_args']['${PROBE}']['extra_args']['extrapar'] = [{'MLA': 'MLP', 'INT_DIM_RES': 512}]

# Model output paths
config['train_args']['${PROBE}']['extra_args']['file']  = ['${MODEL_DIR}/${MODEL_NAME}.pt']
config['train_args']['${PROBE}']['extra_args']['extra'] = ['${MODEL_DIR}/${MODEL_NAME}.h5']

# Data - mead T500 (already correct in base yaml, just set n_train)
config['train_args']['n_train'] = ${N_TRAIN}

with open('${TEMP_YAML}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✓ YAML written: ${TEMP_YAML}")
EOF

    if [ $? -ne 0 ]; then
        echo "✗ YAML generation failed for strategy ${strategy}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "Starting training..."
    TRAIN_START=$(date +%s)

    python ${TRAIN_EMULATOR} \
        --yaml ${TEMP_YAML} \
        --probe ${PROBE} \
        --learning_rate ${LEARNING_RATE} \
        --batchsize ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --ntrain ${N_TRAIN} \
        --save_losses True \
        --save_testing_metrics True \
        --transfer_learning True \
        --pretrained_model ${PRETRAINED_MODEL} \
        --freeze_strategy ${strategy}

    TRAIN_EXIT_CODE=$?
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    TRAIN_HOURS=$((TRAIN_DURATION / 3600))
    TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed in ${TRAIN_HOURS}h ${TRAIN_MINUTES}m"

        if [ -f "${MODEL_DIR}/${MODEL_NAME}_metrics.txt" ]; then
            mv "${MODEL_DIR}/${MODEL_NAME}_metrics.txt" "${METRICS_DIR}/metrics_fs_${strategy}.txt"
            echo "  ✓ Saved metrics: ${METRICS_DIR}/metrics_fs_${strategy}.txt"
        else
            echo "  ⚠ Warning: metrics file not found"
        fi

        if [ -f "${MODEL_DIR}/${MODEL_NAME}.pt" ] && [ -f "${MODEL_DIR}/${MODEL_NAME}.h5" ]; then
            MODEL_SIZE=$(du -h "${MODEL_DIR}/${MODEL_NAME}.pt" | cut -f1)
            echo "  ✓ Model saved (${MODEL_SIZE}): ${MODEL_NAME}.pt"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ ERROR: Model files not found at expected location"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "✗ TRAINING FAILED for strategy=${strategy} (exit code: ${TRAIN_EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo "------------------------------------------------------------------------"

    COMPLETED=$((SUCCESS_COUNT + FAIL_COUNT))
    REMAINING=$((${#STRATEGIES[@]} - COMPLETED))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $COMPLETED -gt 0 ]; then
        AVG_TIME=$((ELAPSED / COMPLETED))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_HOURS=$((EST_REMAINING / 3600))
        EST_MINUTES=$(((EST_REMAINING % 3600) / 60))
        echo "Progress: ${COMPLETED}/${#STRATEGIES[@]} done | ~${EST_HOURS}h ${EST_MINUTES}m remaining"
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"
echo "Finished: $(date)"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "Successful: ${SUCCESS_COUNT}/${#STRATEGIES[@]}"
echo "Failed: ${FAIL_COUNT}/${#STRATEGIES[@]}"
echo ""
echo "Models:"
ls -lh "${MODEL_DIR}"/*.pt 2>/dev/null || echo "  None found"
echo "========================================"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "⚠ WARNING: ${FAIL_COUNT} experiment(s) failed!"
    exit 1
else
    echo "✓ All experiments completed successfully!"
    exit 0
fi
