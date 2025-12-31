#!/bin/bash
# ============================================================================
# TRAIN BEST MODEL FOR MEETING - Early Freezing 2 (model.0 + model.1)
# ============================================================================
# This trains the best-performing strategy (23% frozen, input + ResBlock 1)
# with 1000 epochs for your meeting presentation
# ============================================================================

echo "Starting training of best model at $(date)"
cd ~/data/bela/cocoa/Cocoa

# Configuration
OUTPUT_DIR="projects/lsst_y1/transfer_learning_testing"
YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=1000
STRATEGY="early_2"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Validate YAML first
echo ""
echo "========================================"
echo "VALIDATING YAML"
echo "========================================"
./projects/lsst_y1/check_yaml.sh ${YAML}

if [ $? -ne 0 ]; then
    echo "✗ YAML validation failed. Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "TRAINING BEST MODEL FOR MEETING"
echo "========================================"
echo "Strategy: Early Freezing 2 (model.0 + model.1)"
echo "Frozen layers: 23.0% (input + ResBlock 1)"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${EPOCHS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"
echo ""

# Run training
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy $STRATEGY \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ TRAINING COMPLETED SUCCESSFULLY!"
    echo "========================================"
    echo ""
    echo "Moving output files..."
    
    # Move files with descriptive names
    if [ -f "losses.txt" ]; then
        mv losses.txt ${OUTPUT_DIR}/losses_${STRATEGY}.txt
        echo "✓ Saved: losses_${STRATEGY}.txt"
    else
        echo "⚠ Warning: losses.txt not found"
    fi
    
    if [ -f "testing_metrics.txt" ]; then
        mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_${STRATEGY}.txt
        echo "✓ Saved: testing_metrics_${STRATEGY}.txt"
    else
        echo "⚠ Warning: testing_metrics.txt not found"
    fi
    
    if [ -f "./projects/lsst_y1/emulators/xi_transfer_learning" ]; then
        mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_${STRATEGY}.pth
        echo "✓ Saved: model_${STRATEGY}.pth"
    else
        echo "⚠ Warning: model file not found"
    fi
    
    if [ -f "./projects/lsst_y1/emulators/xi_transfer_learning.h5" ]; then
        mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_${STRATEGY}.h5
        echo "✓ Saved: model_${STRATEGY}.h5"
    else
        echo "⚠ Warning: h5 file not found"
    fi
    
    echo ""
    echo "========================================"
    echo "FILES READY FOR ANALYSIS"
    echo "========================================"
    ls -lh ${OUTPUT_DIR}/*${STRATEGY}*
    
    echo ""
    echo "The model is saved at:"
    echo "  ${OUTPUT_DIR}/model_${STRATEGY}.pth"
    echo "  ${OUTPUT_DIR}/model_${STRATEGY}.h5"
    echo ""
    
else
    echo ""
    echo "========================================"
    echo "✗ TRAINING FAILED"
    echo "========================================"
    echo "Check error messages above"
    exit 1
fi

echo ""
echo "Completed at $(date)"
echo ""