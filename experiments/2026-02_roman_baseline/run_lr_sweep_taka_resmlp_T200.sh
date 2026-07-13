#!/bin/bash
set -e
echo "========================================"
echo "LEARNING RATE SWEEP - Roman CS LCDM Taka (ResMLP) T200 [N=50k]"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "========================================"

PROBE="cosmic_shear"
EPOCHS=250
BATCH_SIZE=64
N_TRAIN=50000
BASE_YAML="/home/grads/tmp/roman_cs_baseline/roman_emulator_cs_taka.yaml"

OUTPUT_BASE="/home/grads/tmp/roman_cs_baseline/outputs/output_taka_resmlp_T200_lr_sweep"
MODEL_DIR="${OUTPUT_BASE}/models"
METRICS_DIR="${OUTPUT_BASE}/metrics"
YAML_DIR="${OUTPUT_BASE}/yamls"

LEARNING_RATES=(1e-3 1e-2 1e-1)

echo ""
echo "=== Pre-flight Checks ==="
if [ ! -f "${BASE_YAML}" ]; then
    echo "ERROR: Base YAML not found: ${BASE_YAML}"
    exit 1
fi
echo "✓ Base YAML found: ${BASE_YAML}"

echo ""
echo "=== Creating output directories ==="
mkdir -p "${MODEL_DIR}" "${METRICS_DIR}" "${YAML_DIR}"
echo "✓ Output directories created under ${OUTPUT_BASE}"

echo ""
echo "========================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================"
echo "Architecture: ResMLP (MLA=MLP, INT_DIM_RES=512)"
echo "Probe: ${PROBE}"
echo "N_train: ${N_TRAIN}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "T_train: 200 | T_valid/test: 100"
echo "========================================"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)

for idx in "${!LEARNING_RATES[@]}"; do
    lr="${LEARNING_RATES[$idx]}"
    exp_num=$((idx + 1))

    LR_TAG=$(echo "${lr}" | sed 's/e-/em/g' | sed 's/\.//g')

    echo ""
    echo "========================================================================"
    echo "EXPERIMENT ${exp_num}/${#LEARNING_RATES[@]}: LR = ${lr}"
    echo "========================================================================"
    echo "Started: $(date)"

    MODEL_NAME="roman_cs_lcdm_taka_resmlp_T200_N${N_TRAIN}_lr${LR_TAG}"
    TEMP_YAML="${YAML_DIR}/roman_emulator_cs_taka_resmlp_T200_N${N_TRAIN}_lr${LR_TAG}.yaml"

    echo "Generating YAML for LR=${lr}..."
    python3 - << EOF
import yaml

with open('${BASE_YAML}', 'r') as f:
    config = yaml.safe_load(f)

# Architecture
config['train_args']['${PROBE}']['extra_args']['extrapar'] = [{'MLA': 'MLP', 'INT_DIM_RES': 512}]

# Model output paths
config['train_args']['${PROBE}']['extra_args']['file']  = ['${MODEL_DIR}/${MODEL_NAME}.pt']
config['train_args']['${PROBE}']['extra_args']['extra'] = ['${MODEL_DIR}/${MODEL_NAME}.h5']

# training_data_path='/' so all files use full absolute paths
config['train_args']['training_data_path'] = '/'

config['train_args']['t_train'] = 200
config['train_args']['train_datavectors_file'] = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_train_datavectors_T200.npy'
config['train_args']['train_parameters_file']  = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_train_parameters_T200.txt'

config['train_args']['n_valid'] = 10000
config['train_args']['t_valid'] = 100
config['train_args']['valid_datavectors_file'] = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_valid_datavectors_T100.npy'
config['train_args']['valid_parameters_file']  = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_valid_parameters_T100.txt'

config['train_args']['n_test'] = 10000
config['train_args']['t_test'] = 100
config['train_args']['test_datavectors_file']  = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_test_datavectors_T100.npy'
config['train_args']['test_parameters_file']   = 'home/grads/backup/mltraining/yijie/roman_3x2_lcdm_maxcorr0dot15/roman_real_lcdm_b_taka_test_parameters_T100.txt'

with open('${TEMP_YAML}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✓ YAML written: ${TEMP_YAML}")
EOF

    if [ $? -ne 0 ]; then
        echo "✗ YAML generation failed for LR=${lr}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    echo "Starting training..."
    TRAIN_START=$(date +%s)

    python /home/grads/tmp/roman_cs_baseline/train_emulator.py \
        --yaml ${TEMP_YAML} \
        --probe ${PROBE} \
        --learning_rate ${lr} \
        --batchsize ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --ntrain ${N_TRAIN} \
        --save_losses True \
        --save_testing_metrics True

    TRAIN_EXIT_CODE=$?
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    TRAIN_HOURS=$((TRAIN_DURATION / 3600))
    TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed in ${TRAIN_HOURS}h ${TRAIN_MINUTES}m"
        echo "  ✓ Losses saved to: ${MODEL_DIR}/${MODEL_NAME}_losses.txt"

        if [ -f "${MODEL_DIR}/${MODEL_NAME}_metrics.txt" ]; then
            mv "${MODEL_DIR}/${MODEL_NAME}_metrics.txt" "${METRICS_DIR}/metrics_lr${LR_TAG}.txt"
            echo "  ✓ Saved metrics: ${METRICS_DIR}/metrics_lr${LR_TAG}.txt"
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
        echo "✗ TRAINING FAILED for LR=${lr} (exit code: ${TRAIN_EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo "------------------------------------------------------------------------"

    COMPLETED=$((SUCCESS_COUNT + FAIL_COUNT))
    REMAINING=$((${#LEARNING_RATES[@]} - COMPLETED))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $COMPLETED -gt 0 ]; then
        AVG_TIME=$((ELAPSED / COMPLETED))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_HOURS=$((EST_REMAINING / 3600))
        EST_MINUTES=$(((EST_REMAINING % 3600) / 60))
        echo "Progress: ${COMPLETED}/${#LEARNING_RATES[@]} done | ~${EST_HOURS}h ${EST_MINUTES}m remaining"
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
echo "Successful: ${SUCCESS_COUNT}/${#LEARNING_RATES[@]}"
echo "Failed: ${FAIL_COUNT}/${#LEARNING_RATES[@]}"
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
