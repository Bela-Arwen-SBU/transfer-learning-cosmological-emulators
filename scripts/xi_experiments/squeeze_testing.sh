#!/bin/bash
# Covariance squeezing experiments with YAML validation, metrics, and independent model saving
echo "Starting covariance squeezing experiments at $(date)"
echo "=== Covariance Squeezing Tests ==="

EPOCHS=1000  # Adjust this as needed
OUTPUT_DIR="projects/lsst_y1/covariance_squeeze_testing"
YAML="./projects/lsst_y1/xi_emulator_low_accuracy.yaml"

# ============================================================================
# YAML VALIDATION
# ============================================================================

echo ""
echo "========================================"
echo "VALIDATING YAML CONFIGURATION"
echo "========================================"

# Check if YAML exists
if [ ! -f "$YAML" ]; then
    echo "✗ ERROR: YAML file not found: $YAML"
    exit 1
fi
echo "✓ YAML file exists: $YAML"

# Use Python to extract and validate paths from YAML
python3 << END_PYTHON
import yaml
import sys
from pathlib import Path

errors = 0
warnings = 0

try:
    with open('$YAML', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ YAML syntax valid')
    print()
except Exception as e:
    print(f'✗ ERROR: Invalid YAML: {e}')
    sys.exit(1)

print('Checking paths specified in YAML...')
print()

# Check training data path
if 'train_args' in config and 'training_data_path' in config['train_args']:
    data_path = config['train_args']['training_data_path']
    print(f'Training data path: {data_path}')
    if Path(data_path).exists():
        print('  ✓ Directory exists')
        
        # Check specific data files
        data_files = [
            'train_datavectors_file',
            'train_parameters_file',
            'valid_datavectors_file', 
            'valid_parameters_file',
            'test_datavectors_file',
            'test_parameters_file'
        ]
        
        for key in data_files:
            if key in config['train_args']:
                filename = config['train_args'][key]
                filepath = Path(data_path) / filename
                if filepath.exists():
                    print(f'  ✓ {filename}')
                else:
                    print(f'  ✗ Missing: {filename}')
                    errors += 1
    else:
        print('  ✗ Directory does not exist')
        errors += 1
    print()

# Check data covariance file
if 'train_args' in config and 'data_covmat_file' in config['train_args']:
    cov_file = config['train_args']['data_covmat_file']
    print(f'Data covariance: {cov_file}')
    if Path(cov_file).exists():
        print('  ✓ File exists')
    else:
        print('  ✗ File does not exist')
        errors += 1
    print()

# Check model output paths
if 'train_args' in config and 'cosmic_shear' in config['train_args']:
    cs_config = config['train_args']['cosmic_shear']
    if 'extra_args' in cs_config and 'file' in cs_config['extra_args']:
        model_path = cs_config['extra_args']['file'][0]
        model_dir = Path(model_path).parent
        print(f'Model output directory: {model_dir}')
        if model_dir.exists():
            print('  ✓ Directory exists')
        else:
            print('  ⚠ Directory does not exist (will be created)')
            warnings += 1
        print(f'  ℹ Model will be saved as: {Path(model_path).name}')
        print()

# Summary
print('='*50)
if errors > 0:
    print(f'✗ VALIDATION FAILED: {errors} error(s), {warnings} warning(s)')
    sys.exit(1)
elif warnings > 0:
    print(f'⚠ VALIDATION PASSED with {warnings} warning(s)')
    sys.exit(0)
else:
    print('✓ VALIDATION PASSED')
    sys.exit(0)

END_PYTHON

VALIDATION_STATUS=$?

if [ $VALIDATION_STATUS -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "✗ YAML validation failed"
    echo "Please fix the paths in $YAML"
    echo "========================================"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ YAML validation successful"
echo "========================================"

# ============================================================================
# EXPERIMENT SETUP
# ============================================================================

# Create output directories
mkdir -p ${OUTPUT_DIR}/models
mkdir -p ${OUTPUT_DIR}/losses
mkdir -p ${OUTPUT_DIR}/metrics

echo ""
echo "========================================"
echo "COVARIANCE SQUEEZING EXPERIMENTS"
echo "Settings:"
echo "  Epochs: $EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================"
echo ""

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

for squeeze in 2 4 8 16 32; do
    echo "========================================"
    echo "Running squeeze factor ${squeeze}..."
    echo "========================================"
    
    python ./projects/lsst_y1/train_emulator.py \
        --yaml $YAML \
        --probe cosmic_shear \
        --device cpu \
        --save_losses True \
        --save_testing_metrics True \
        --squeeze_factor ${squeeze} \
        --epochs ${EPOCHS}
    
    # Move and rename output files
    mv losses.txt ${OUTPUT_DIR}/losses/losses_squeeze_${squeeze}.txt
    mv testing_metrics.txt ${OUTPUT_DIR}/metrics/metrics_squeeze_${squeeze}.txt
    mv ./projects/lsst_y1/emulators/xi_low_accuracy ${OUTPUT_DIR}/models/xi_low_accuracy_squeeze_${squeeze}.pth
    mv ./projects/lsst_y1/emulators/xi_low_accuracy.h5 ${OUTPUT_DIR}/models/xi_low_accuracy_squeeze_${squeeze}.h5
    
    echo "✓ Completed squeeze factor ${squeeze} at $(date)"
    echo ""
done

echo "========================================"
echo "All covariance squeezing experiments completed at $(date)"
echo "========================================"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Loss files:"
ls -lh ${OUTPUT_DIR}/losses/
echo ""
echo "Metrics files:"
ls -lh ${OUTPUT_DIR}/metrics/
echo ""
echo "Model files:"
ls -lh ${OUTPUT_DIR}/models/