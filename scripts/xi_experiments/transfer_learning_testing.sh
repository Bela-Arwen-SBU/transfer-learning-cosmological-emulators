#!/bin/bash
echo "Starting transfer learning testing at $(date)"

# Create results directory
mkdir -p projects/lsst_y1/transfer_learning_testing

# Define output directory
OUTPUT_DIR="projects/lsst_y1/transfer_learning_testing"

# Consistent parameters
YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=1000

echo "========================================"
echo "TRANSFER LEARNING TESTING"
echo "Consistent parameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Training: 1000 high-accuracy samples"
echo "  Validation: 1000 high-accuracy samples"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================"

# ============================================================================
# YAML VALIDATION
# ============================================================================

echo ""
echo "========================================"
echo "VALIDATING YAML CONFIGURATION"
echo "========================================"

./projects/lsst_y1/check_yaml.sh ${YAML}

if [ $? -ne 0 ]; then
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
# EXPERIMENT 1: Baseline (No Freezing)
# ============================================================================
echo ""
echo "=== EXPERIMENT 1: Baseline (No Freezing) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy none \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_none.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_none.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_none.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_none.h5

# ============================================================================
# EXPERIMENT 2: Early Freezing 1 (model.0 only)
# ============================================================================
echo ""
echo "=== EXPERIMENT 2: Early Freezing 1 (model.0 only) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_1 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_early_1.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_early_1.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_early_1.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_early_1.h5

# ============================================================================
# EXPERIMENT 3: Early Freezing 2 (model.0 + model.1)
# ============================================================================
echo ""
echo "=== EXPERIMENT 3: Early Freezing 2 (model.0 + model.1) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_early_2.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_early_2.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_early_2.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_early_2.h5

# ============================================================================
# EXPERIMENT 4: Early Freezing 3 (model.0 + model.1 + model.2)
# ============================================================================
echo ""
echo "=== EXPERIMENT 4: Early Freezing 3 (model.0 + model.1 + model.2) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_3 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_early_3.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_early_3.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_early_3.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_early_3.h5

# ============================================================================
# EXPERIMENT 5: Early Freezing 4 (model.0 + model.1 + model.2 + model.3)
# ============================================================================
echo ""
echo "=== EXPERIMENT 5: Early Freezing 4 (model.0 + model.1 + model.2 + model.3) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_4 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_early_4.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_early_4.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_early_4.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_early_4.h5

# ============================================================================
# EXPERIMENT 6: Late Freezing 1 (model.4 only)
# ============================================================================
echo ""
echo "=== EXPERIMENT 6: Late Freezing 1 (model.4 only) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_1 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_late_1.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_late_1.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_late_1.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_late_1.h5

# ============================================================================
# EXPERIMENT 7: Late Freezing 2 (model.4 + model.3)
# ============================================================================
echo ""
echo "=== EXPERIMENT 7: Late Freezing 2 (model.4 + model.3) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_late_2.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_late_2.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_late_2.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_late_2.h5

# ============================================================================
# EXPERIMENT 8: Late Freezing 3 (model.4 + model.3 + model.2)
# ============================================================================
echo ""
echo "=== EXPERIMENT 8: Late Freezing 3 (model.4 + model.3 + model.2) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_3 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_late_3.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_late_3.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_late_3.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_late_3.h5

# ============================================================================
# EXPERIMENT 9: Late Freezing 4 (model.4 + model.3 + model.2 + model.1)
# ============================================================================
echo ""
echo "=== EXPERIMENT 9: Late Freezing 4 (model.4 + model.3 + model.2 + model.1) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_4 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_late_4.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_late_4.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_late_4.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_late_4.h5

# ============================================================================
# EXPERIMENT 10: Input + Output Freezing (model.0 + model.4)
# ============================================================================
echo ""
echo "=== EXPERIMENT 10: Input + Output Freezing (model.0 + model.4) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy input_output \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_input_output.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_input_output.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_input_output.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_input_output.h5

# ============================================================================
# EXPERIMENT 11: ResNet Block 1 Freezing (model.1 only)
# ============================================================================
echo ""
echo "=== EXPERIMENT 11: ResNet Block 1 Freezing (model.1 only) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_1 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_1.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_1.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_1.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_1.h5

# ============================================================================
# EXPERIMENT 12: ResNet Block 2 Freezing (model.2 only)
# ============================================================================
echo ""
echo "=== EXPERIMENT 12: ResNet Block 2 Freezing (model.2 only) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_2.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_2.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_2.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_2.h5

# ============================================================================
# EXPERIMENT 13: ResNet Block 3 Freezing (model.3 only)
# ============================================================================
echo ""
echo "=== EXPERIMENT 13: ResNet Block 3 Freezing (model.3 only) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_3 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_3.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_3.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_3.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_3.h5

# ============================================================================
# EXPERIMENT 14: ResNet Blocks 1+2 Freezing (model.1 + model.2)
# ============================================================================
echo ""
echo "=== EXPERIMENT 14: ResNet Blocks 1+2 Freezing (model.1 + model.2) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_12 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_12.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_12.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_12.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_12.h5

# ============================================================================
# EXPERIMENT 15: ResNet Blocks 2+3 Freezing (model.2 + model.3)
# ============================================================================
echo ""
echo "=== EXPERIMENT 15: ResNet Blocks 2+3 Freezing (model.2 + model.3) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_23 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_23.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_23.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_23.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_23.h5

# ============================================================================
# EXPERIMENT 16: ResNet Blocks 1+2+3 Freezing (model.1 + model.2 + model.3)
# ============================================================================
echo ""
echo "=== EXPERIMENT 16: ResNet Blocks 1+2+3 Freezing (model.1 + model.2 + model.3) ==="
python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_123 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt ${OUTPUT_DIR}/losses_resnet_123.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_resnet_123.txt
mv ./projects/lsst_y1/emulators/xi_transfer_learning ${OUTPUT_DIR}/model_resnet_123.pth
mv ./projects/lsst_y1/emulators/xi_transfer_learning.h5 ${OUTPUT_DIR}/model_resnet_123.h5

echo ""
echo "========================================"
echo "All experiments completed at $(date)"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Generated files:"
ls -lh ${OUTPUT_DIR}/
