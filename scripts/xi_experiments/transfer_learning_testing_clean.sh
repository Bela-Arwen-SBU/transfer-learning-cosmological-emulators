#!/bin/bash
echo "Starting transfer learning testing at $(date)"

# Define output directory
OUTPUT_DIR="projects/lsst_y1/transfer_learning_testing_1000epochs"

# Consistent parameters
YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=1000

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

# SKIP parameter_covmat_file check - not needed for transfer learning with existing data
print('⚠ Skipping parameter_covmat_file check (not needed for transfer learning)')
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

# Check pretrained model (for transfer learning)
print(f'Pretrained model: $PRETRAINED_MODEL')
if Path('$PRETRAINED_MODEL').exists():
    print('  ✓ Model file exists')
else:
    print('  ✗ Model file does not exist')
    errors += 1

if Path('$PRETRAINED_MODEL.h5').exists():
    print('  ✓ H5 file exists')
else:
    print('  ✗ H5 file does not exist')
    errors += 1
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

# Create results directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "========================================"
echo "TRANSFER LEARNING TESTING"
echo "Consistent parameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Training: 1000 high-accuracy samples"
echo "  Validation: 9000 high-accuracy samples"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================"

# Define all experiments
experiments=(
    "none:Baseline (No Freezing)"
    "early_1:Early Freezing 1 (model.0 only)"
    "early_2:Early Freezing 2 (model.0 + model.1)"
    "early_3:Early Freezing 3 (model.0 + model.1 + model.2)"
    "early_4:Early Freezing 4 (model.0 + model.1 + model.2 + model.3)"
#     "late_1:Late Freezing 1 (model.4 only)"
#     "late_2:Late Freezing 2 (model.4 + model.3)"
#     "late_3:Late Freezing 3 (model.4 + model.3 + model.2)"
#     "late_4:Late Freezing 4 (model.4 + model.3 + model.2 + model.1)"
#     "input_output:Input + Output Freezing (model.0 + model.4)"
#     "resnet_1:ResNet Block 1 Freezing (model.1 only)"
#     "resnet_2:ResNet Block 2 Freezing (model.2 only)"
#     "resnet_3:ResNet Block 3 Freezing (model.3 only)"
#     "resnet_12:ResNet Blocks 1+2 Freezing (model.1 + model.2)"
#     "resnet_13:ResNet Blocks 1+3 Freezing (model.1 + model.3)"
#     "resnet_23:ResNet Blocks 2+3 Freezing (model.2 + model.3)"
#     "resnet_123:ResNet Blocks 1+2+3 Freezing (model.1 + model.2 + model.3)"
)

exp_num=1

for experiment in "${experiments[@]}"; do
    IFS=':' read -r strategy description <<< "$experiment"
    
    echo ""
    echo "=========================================================================="
    echo "EXPERIMENT ${exp_num}: ${description}"
    echo "=========================================================================="
    
    python projects/lsst_y1/train_emulator.py \
        --yaml $YAML \
        --probe $PROBE \
        --transfer_learning True \
        --pretrained_model $PRETRAINED_MODEL \
        --freeze_strategy $strategy \
        --learning_rate $LEARNING_RATE \
        --epochs $EPOCHS \
        --save_losses True \
        --save_testing_metrics True
    
    mv losses.txt ${OUTPUT_DIR}/losses_${strategy}.txt
    mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_${strategy}.txt
    mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_${strategy}.pth
    mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_${strategy}.h5
    
    echo "✓ Completed: ${description}"
    echo "   Files saved as: *_${strategy}.*"
    
    ((exp_num++))
done

echo ""
echo "========================================"
echo "All ${#experiments[@]} experiments completed at $(date)"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh ${OUTPUT_DIR}/