#!/bin/bash
set -e  # Exit on any error

echo "========================================"
echo "TRANSFER LEARNING DATA SIZE SWEEP"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "========================================"

# ============================================================================
# CONFIGURATION
# ============================================================================
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=1000
FREEZE_STRATEGY="early_2"
BASE_YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"

# Output directory structure
OUTPUT_BASE="./projects/lsst_y1/emulators/data_size_sweep"
MODEL_DIR="${OUTPUT_BASE}/models"
LOSSES_DIR="${OUTPUT_BASE}/losses"
METRICS_DIR="${OUTPUT_BASE}/metrics"
YAML_DIR="${OUTPUT_BASE}/yamls"

# Training sizes
SIZES=(200 500 1000 1500 2500 5000)

# ============================================================================
# VALIDATION SECTION
# ============================================================================
echo ""
echo "=== Pre-flight Checks ==="

# Check base YAML exists
if [ ! -f "${BASE_YAML}" ]; then
    echo "ERROR: Base YAML not found: ${BASE_YAML}"
    exit 1
fi
echo "✓ Base YAML found: ${BASE_YAML}"

# Check pretrained model exists
if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "ERROR: Pretrained model not found: ${PRETRAINED_MODEL}"
    exit 1
fi
echo "✓ Pretrained model found: ${PRETRAINED_MODEL}"

# Extract and verify data directory
DATA_DIR="./projects/lsst_y1/data/transfer_learning"
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory does not exist: ${DATA_DIR}"
    exit 1
fi
echo "✓ Data directory: ${DATA_DIR}"

# Validate all required data files exist
echo ""
echo "=== Checking data files ==="
MISSING_FILES=0

# Check validation/test files (should be constant across all runs)
for file in "valid_datavectors_high_accuracy.npy" "valid_parameters_high_accuracy.txt" \
            "test_datavectors_high_accuracy.npy" "test_parameters_high_accuracy.txt"; do
    if [ ! -f "${DATA_DIR}/${file}" ]; then
        echo "✗ Missing: ${file}"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        FILE_SIZE=$(du -h "${DATA_DIR}/${file}" | cut -f1)
        echo "✓ Found: ${file} (${FILE_SIZE})"
    fi
done

# Check training files for each size
echo ""
echo "=== Checking training files for each size ==="
for size in "${SIZES[@]}"; do
    train_dv="train_datavectors_high_accuracy_${size}.npy"
    train_params="train_parameters_high_accuracy_${size}.txt"
    
    if [ ! -f "${DATA_DIR}/${train_dv}" ]; then
        echo "✗ Missing: ${train_dv}"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✓ Size ${size}: datavectors found"
    fi
    
    if [ ! -f "${DATA_DIR}/${train_params}" ]; then
        echo "✗ Missing: ${train_params}"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✓ Size ${size}: parameters found"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: ${MISSING_FILES} data files are missing!"
    exit 1
fi

echo ""
echo "✓ All data files verified!"

# ============================================================================
# SETUP
# ============================================================================
echo ""
echo "=== Creating output directories ==="
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOSSES_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${YAML_DIR}"
echo "✓ Output directory: ${OUTPUT_BASE}"
echo "  - Models: ${MODEL_DIR}"
echo "  - Losses: ${LOSSES_DIR}"
echo "  - Metrics: ${METRICS_DIR}"
echo "  - YAMLs: ${YAML_DIR}"

# ============================================================================
# TRAINING LOOP
# ============================================================================
echo ""
echo "========================================"
echo "EXPERIMENT CONFIGURATION"
echo "========================================"
echo "Freeze Strategy: ${FREEZE_STRATEGY}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${EPOCHS}"
echo "Sizes: ${SIZES[@]}"
echo "Pretrained: ${PRETRAINED_MODEL}"
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
    echo "EXPERIMENT ${exp_num}/${#SIZES[@]}: Training with ${size} samples"
    echo "========================================================================"
    echo "Started: $(date)"
    
    # Create temporary YAML for this size
    TEMP_YAML="${YAML_DIR}/xi_emulator_transfer_${size}.yaml"
    
    echo ""
    echo "Generating YAML configuration..."
    python3 << EOF
import yaml
import sys

# Load base YAML
try:
    with open('${BASE_YAML}', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"ERROR: Failed to load base YAML: {e}", file=sys.stderr)
    sys.exit(1)

# Update training data files and size
config['train_args']['n_train'] = ${size}
config['train_args']['train_datavectors_file'] = 'train_datavectors_high_accuracy_${size}.npy'
config['train_args']['train_parameters_file'] = 'train_parameters_high_accuracy_${size}.txt'

# Ensure validation/test files are correct (should already be, but be explicit)
config['train_args']['n_valid'] = 1000
config['train_args']['valid_datavectors_file'] = 'valid_datavectors_high_accuracy.npy'
config['train_args']['valid_parameters_file'] = 'valid_parameters_high_accuracy.txt'

config['train_args']['n_test'] = 1000
config['train_args']['test_datavectors_file'] = 'test_datavectors_high_accuracy.npy'
config['train_args']['test_parameters_file'] = 'test_parameters_high_accuracy.txt'

# Update model output paths - CRITICAL for separate models
config['train_args']['${PROBE}']['extra_args']['file'] = ['${MODEL_DIR}/xi_transfer_${size}']
config['train_args']['${PROBE}']['extra_args']['extra'] = ['${MODEL_DIR}/xi_transfer_${size}.h5']

# Save YAML
try:
    with open('${TEMP_YAML}', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Created YAML: ${TEMP_YAML}")
    print(f"  Training size: ${size}")
    print(f"  Training files: train_*_high_accuracy_${size}.*")
    print(f"  Validation size: 1000 (constant)")
    print(f"  Model output: ${MODEL_DIR}/xi_transfer_${size}")
except Exception as e:
    print(f"ERROR: Failed to save YAML: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo "✗ YAML generation failed for size ${size}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Run training
    echo ""
    echo "Starting training..."
    TRAIN_START=$(date +%s)

    # Determine appropriate batch size based on training size
    if [ $size -le 256 ]; then
        BATCH_SIZE=64
    else
        BATCH_SIZE=256
    fi

    echo "Using batch size: ${BATCH_SIZE}"
    
    python projects/lsst_y1/train_emulator.py \
        --yaml ${TEMP_YAML} \
        --probe ${PROBE} \
        --transfer_learning True \
        --pretrained_model ${PRETRAINED_MODEL} \
        --freeze_strategy ${FREEZE_STRATEGY} \
        --learning_rate ${LEARNING_RATE} \
        --batchsize ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --save_losses True \
        --save_testing_metrics True
    
    TRAIN_EXIT_CODE=$?
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
    TRAIN_MINUTES=$((TRAIN_DURATION / 60))
    TRAIN_SECONDS=$((TRAIN_DURATION % 60))
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Training completed in ${TRAIN_MINUTES}m ${TRAIN_SECONDS}s"
        
        # Move loss and metrics files from CWD to organized directories
        FILES_MOVED=0
        if [ -f "losses.txt" ]; then
            mv losses.txt "${LOSSES_DIR}/losses_${size}.txt"
            echo "  ✓ Saved losses: ${LOSSES_DIR}/losses_${size}.txt"
            FILES_MOVED=$((FILES_MOVED + 1))
        else
            echo "  ⚠ Warning: losses.txt not found in working directory"
        fi
        
        if [ -f "testing_metrics.txt" ]; then
            mv testing_metrics.txt "${METRICS_DIR}/testing_metrics_${size}.txt"
            echo "  ✓ Saved metrics: ${METRICS_DIR}/testing_metrics_${size}.txt"
            FILES_MOVED=$((FILES_MOVED + 1))
        else
            echo "  ⚠ Warning: testing_metrics.txt not found in working directory"
        fi
        
        # Verify model files were created
        if [ -f "${MODEL_DIR}/xi_transfer_${size}" ] && [ -f "${MODEL_DIR}/xi_transfer_${size}.h5" ]; then
            MODEL_SIZE=$(du -h "${MODEL_DIR}/xi_transfer_${size}" | cut -f1)
            H5_SIZE=$(du -h "${MODEL_DIR}/xi_transfer_${size}.h5" | cut -f1)
            echo "  ✓ Model: ${MODEL_SIZE} (${MODEL_DIR}/xi_transfer_${size})"
            echo "  ✓ Preprocessing: ${H5_SIZE} (${MODEL_DIR}/xi_transfer_${size}.h5)"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo ""
            echo "✓✓✓ SUCCESS: Size ${size} completed ✓✓✓"
        else
            echo "  ✗ ERROR: Model files were not created at expected location!"
            echo "    Expected: ${MODEL_DIR}/xi_transfer_${size}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        
    else
        echo ""
        echo "✗✗✗ TRAINING FAILED for size ${size} (exit code: ${TRAIN_EXIT_CODE}) ✗✗✗"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo "------------------------------------------------------------------------"
    
    # Progress update
    COMPLETED=$((SUCCESS_COUNT + FAIL_COUNT))
    REMAINING=$((${#SIZES[@]} - COMPLETED))
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $COMPLETED -gt 0 ]; then
        AVG_TIME=$((ELAPSED / COMPLETED))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_MINUTES=$((EST_REMAINING / 60))
        echo "Progress: ${COMPLETED}/${#SIZES[@]} completed, ~${EST_MINUTES} minutes remaining"
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
echo ""
echo "Results:"
echo "  Successful: ${SUCCESS_COUNT}/${#SIZES[@]}"
echo "  Failed: ${FAIL_COUNT}/${#SIZES[@]}"
echo ""
echo "Output location: ${OUTPUT_BASE}"
echo ""

# Show summary of created files
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Models created:"
    ls -lh "${MODEL_DIR}" | grep "xi_transfer_" || echo "  None found"
    echo ""
    
    echo "Loss files:"
    ls -lh "${LOSSES_DIR}" | grep "losses_" || echo "  None found"
    echo ""
    
    echo "Metrics files:"
    ls -lh "${METRICS_DIR}" | grep "testing_metrics_" || echo "  None found"
fi

echo ""
echo "To analyze results, use these paths in your notebook:"
echo "  model_dir = Path('${MODEL_DIR}')"
echo "  losses_dir = Path('${LOSSES_DIR}')"
echo "  metrics_dir = Path('${METRICS_DIR}')"
echo ""
echo "========================================"

# Exit with error if any runs failed
if [ $FAIL_COUNT -gt 0 ]; then
    echo "⚠ WARNING: ${FAIL_COUNT} experiment(s) failed!"
    exit 1
else
    echo "✓ All experiments completed successfully!"
    exit 0
fi