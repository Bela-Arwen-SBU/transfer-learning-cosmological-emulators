#!/bin/bash

OUTPUT_DIR="projects/lsst_y1/transfer_learning_testing_1000epochs"

echo "Training high accuracy baseline (10k samples, 1000 epochs) at $(date)"

python projects/lsst_y1/train_emulator.py \
    --yaml ./projects/lsst_y1/xi_emulator_high_accuracy_baseline_1000ep.yaml \
    --probe cosmic_shear \
    --learning_rate 1e-3 \
    --epochs 1000 \
    --save_losses True \
    --save_testing_metrics True

# Move outputs
mv losses.txt ${OUTPUT_DIR}/losses_high_acc_baseline.txt
mv testing_metrics.txt ${OUTPUT_DIR}/testing_metrics_high_acc_baseline.txt
mv ./projects/lsst_y1/emulators/xi_high_accuracy_baseline_1000ep ${OUTPUT_DIR}/model_high_acc_baseline.pth
mv ./projects/lsst_y1/emulators/xi_high_accuracy_baseline_1000ep.h5 ${OUTPUT_DIR}/model_high_acc_baseline.h5

echo "✓ Done at $(date)"
