#!/bin/bash
set -e
echo "========================================"
echo "TRANSFER LEARNING SWEEP - Taka->Mead Roman CS LCDM"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "========================================"
# ============================================================================
# CONFIGURATION
# ============================================================================
PROBE="cosmic_shear"
LEARNING_RATE=1e-3
EPOCHS=250
BATCH_SIZE=256
BASE_YAML="/home/grads/tmp/roman_cs_baseline/roman_emulator_cs_mead.yaml"
PRETRAINED_MODEL="/home/grads/tmp/roman_cs_baseline/output/models/roman_cs_lcdm_taka_N100000.pt"
FREEZE_STRATEGY="none"
# Output directory structure
OUTPUT_BASE="/home/grads/tmp/roman_cs_baseline/output_tl"
MODEL_DIR="${OUTPUT_BASE}/models"
LOSSES_DIR="${OUTPUT_BASE}/losses"
METRICS_DIR="${OUTPUT_BASE}/metrics"
YAML_DIR="${OUTPUT_BASE}/yamls"
# Fine-tuning sizes
SIZES=(10000 25000 50000)
# ============================================================================
# SETUP
# ============================================================================
echo ""
echo "=== Pre-flight Checks ==="
if [ ! -f "${BASE_YAML}" ]; then
    echo "ERROR: Base YAML not found: ${BASE_YAML}"
    exit 1
fi
echo "✓ Base YAML found: ${BASE_YAML}"
if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: Pretrained model not found: ${PRETRAINED_MODEL}"
    exit 1
fi
echo "✓ Pretrained model found: ${PRETRAINED_MODEL}"
echo ""
echo "=== Creating output directories ==="
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOSSES_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${YAML_DIR}"
echo "✓ Output directories created under ${OUTPUT_BASE}"
# ============================================================================
# TRAINING LOOP
# ============================================================================
echo ""
echo "========================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================"
echo "Probe: ${PROBE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Freeze strategy: ${FREEZE_STRATEGY}"
echo "Fine-tuning sizes: ${SIZES[@]}"
echo "========================================"
echo ""
SUCCESS_COUNT=0
FAIL_COUNT=0
START_TIME=$(date +%s)
for idx in "${!SIZES[@]}"; do
    size="${SIZES[$idx]}"
    exp_num=$((idx + 1))
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT ${exp_num}/${#SIZES[@]}: N_finetune = ${size}"
    echo "========================================================================"
    echo "Started: $(date)"
    # Generate per-run YAML with updated output paths
    TEMP_YAML="${YAML_DIR}/roman_emulator_cs_tl_N${size}.yaml"
    echo "Generating YAML for N_finetune=${size}..."
    python3 << EOF
import yaml, sys
with open('${BASE_YAML}', 'r') as f:
    config = yaml.safe_load(f)
# Update output model paths
config['train_args']['${PROBE}']['extra_args']['file'] = ['${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.pt']
config['train_args']['${PROBE}']['extra_args']['extra'] = ['${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.h5']
with open('${TEMP_YAML}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"✓ YAML written: ${TEMP_YAML}")
print(f"  Model: ${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.pt")
EOF
    if [ $? -ne 0 ]; then
        echo "✗ YAML generation failed for size ${size}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    # Run training
    echo "Starting transfer learning fine-tuning..."
    TRAIN_START=$(date +%s)
    python /home/grads/tmp/roman_cs_baseline/train_emulator.py \
        --yaml ${TEMP_YAML} \
        --probe ${PROBE} \
        --learning_rate ${LEARNING_RATE} \
        --batchsize ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --ntrain ${size} \
        --save_losses True \
        --save_testing_metrics True \
        --transfer_learning True \
        --pretrained_model ${PRETRAINED_MODEL} \
        --freeze_strategy ${FREEZE_STRATEGY}
    TRAIN_EXIT_CODE=$?
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    TRAIN_HOURS=$((TRAIN_DURATION / 3600))
    TRAIN_MINUTES=$(((TRAIN_DURATION % 3600) / 60))
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✓ Training completed in ${TRAIN_HOURS}h ${TRAIN_MINUTES}m"
        # Move losses file
        if [ -f "losses.txt" ]; then
            mv losses.txt "${LOSSES_DIR}/losses_TL_N${size}.txt"
            echo "  ✓ Saved losses: ${LOSSES_DIR}/losses_TL_N${size}.txt"
        else
            echo "  ⚠ Warning: losses.txt not found"
        fi
        # Move testing metrics
        if [ -f "${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}_metrics.txt" ]; then
            mv "${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}_metrics.txt" "${METRICS_DIR}/metrics_TL_N${size}.txt"
            echo "  ✓ Saved metrics: ${METRICS_DIR}/metrics_TL_N${size}.txt"
        else
            echo "  ⚠ Warning: metrics file not found"
        fi
        # Verify model files
        if [ -f "${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.pt" ] && \
           [ -f "${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.h5" ]; then
            MODEL_SIZE=$(du -h "${MODEL_DIR}/roman_cs_lcdm_taka2mead_TL_N${size}.pt" | cut -f1)
            echo "  ✓ Model saved (${MODEL_SIZE}): roman_cs_lcdm_taka2mead_TL_N${size}.pt"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ ERROR: Model files not found at expected location"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "✗ TRAINING FAILED for N_finetune=${size} (exit code: ${TRAIN_EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "------------------------------------------------------------------------"
    # Progress estimate
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
# ============================================================================
# SUMMARY
# ============================================================================
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
echo "Successful: ${SUCCESS_COUNT}/${#SIZES[@]}"
echo "Failed: ${FAIL_COUNT}/${#SIZES[@]}"
echo ""
echo "Models:"
ls -lh "${MODEL_DIR}"/*.pt 2>/dev/null || echo "  None found"
echo ""
echo "Losses:"
ls -lh "${LOSSES_DIR}"/*.txt 2>/dev/null || echo "  None found"
echo "========================================"
if [ $FAIL_COUNT -gt 0 ]; then
    echo "⚠ WARNING: ${FAIL_COUNT} experiment(s) failed!"
    exit 1
else
    echo "✓ All experiments completed successfully!"
    exit 0
fi
