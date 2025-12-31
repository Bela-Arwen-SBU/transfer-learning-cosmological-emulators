#!/bin/bash
# check_yaml.sh - Quick YAML validation for single training runs
# Usage: ./check_yaml.sh <yaml_file>

YAML_FILE=$1

if [ -z "$YAML_FILE" ]; then
    echo "Usage: ./check_yaml.sh <yaml_file>"
    echo "Example: ./check_yaml.sh xi_high_accuracy.yaml"
    exit 1
fi

if [ ! -f "$YAML_FILE" ]; then
    echo "✗ ERROR: YAML file not found: $YAML_FILE"
    exit 1
fi

echo "Validating: $YAML_FILE"
echo ""

python3 << END_PYTHON
import yaml
import sys
from pathlib import Path

try:
    with open('$YAML_FILE', 'r') as f:
        config = yaml.safe_load(f)
    print('✓ YAML syntax valid\n')
except Exception as e:
    print(f'✗ YAML syntax error: {e}')
    sys.exit(1)

errors = 0

# Training data
if 'train_args' in config:
    ta = config['train_args']
    
    if 'training_data_path' in ta:
        data_path = Path(ta['training_data_path'])
        print(f'Training data: {data_path}')
        if data_path.exists():
            files = ['train_datavectors_file', 'train_parameters_file',
                    'valid_datavectors_file', 'valid_parameters_file',
                    'test_datavectors_file', 'test_parameters_file']
            missing = [ta[f] for f in files if f in ta and not (data_path / ta[f]).exists()]
            if missing:
                print(f'  ✗ Missing files: {", ".join(missing)}')
                errors += 1
            else:
                print('  ✓ All data files present')
        else:
            print('  ✗ Directory not found')
            errors += 1
    
    # Covariance
    if 'data_covmat_file' in ta:
        cov = Path(ta['data_covmat_file'])
        print(f'Covariance: {cov}')
        print('  ✓' if cov.exists() else '  ✗ Not found')
        if not cov.exists(): errors += 1
    
    # Output path
    if 'cosmic_shear' in ta and 'extra_args' in ta['cosmic_shear']:
        ea = ta['cosmic_shear']['extra_args']
        if 'file' in ea:
            out_dir = Path(ea['file'][0]).parent
            print(f'Output dir: {out_dir}')
            print('  ✓' if out_dir.exists() else '  ⚠ Will be created')

print()
if errors > 0:
    print(f'✗ FAILED: {errors} error(s)')
    sys.exit(1)
else:
    print('✓ PASSED - Ready to train!')
    sys.exit(0)
END_PYTHON