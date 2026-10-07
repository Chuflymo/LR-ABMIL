#!/bin/bash

# grid search
LR_LIST=("8e-5" "1e-4" "1e-6" "8e-6" "2e-4" "5e-4" "8e-4" "1e-3" "2e-3" "5e-3")
WD_LIST=("0." "1e-5" "1e-4" "1e-3" "1e-2")

OUTPUT_DIR="/data/ysli/Johnbo/Focus_High_ABMIL/results"

RESULTS_FILE="${OUTPUT_DIR}/lr_experiment_summary.txt"
echo "Learning Rate,Weight Decay,Test AUC,Test Acc,Train Loss" > "${RESULTS_FILE}"

for lr in "${LR_LIST[@]}"; do
    for wd in "${WD_LIST[@]}"; do
        echo "================================================================================"
        echo "Start: lr=${lr}, wd=${wd}"
        echo "================================================================================"

        bash /data/ysli/Johnbo/Focus_High_ABMIL/run_single_exp.sh "${lr}" "${wd}" "${OUTPUT_DIR}"

        EXP_ID="lr_${lr}_wd_${wd}"
        LOG_FILE="${OUTPUT_DIR}/${EXP_ID}/training.log"

        if [ -f "${LOG_FILE}" ]; then
            TEST_AUC=$(grep "Epoch" "${LOG_FILE}" | tail -1 | sed 's/.*auc_value:\([0-9.]*\).*/\1/')
            TEST_ACC=$(grep "Epoch" "${LOG_FILE}" | tail -1 | sed 's/.*accuracy: \([0-9.]*\).*/\1/')
            TRAIN_LOSS=$(grep "Epoch" "${LOG_FILE}" | tail -1 | sed 's/.*train loss: \([0-9.eE-+]*\).*/\1/')

            echo "Result: lr=${lr}, wd=${wd}, AUC=${TEST_AUC}, ACC=${TEST_ACC}, LOSS=${TRAIN_LOSS}"
            echo "${lr},${wd},${TEST_AUC},${TEST_ACC},${TRAIN_LOSS}" >> "${RESULTS_FILE}"
        else
            echo "Error ${LOG_FILE} not found"
            echo "${lr},${wd},ERROR,ERROR,ERROR" >> "${RESULTS_FILE}"
        fi

        echo "================================================================================"
        echo "Experiment lr=${lr}, wd=${wd} Completed."
        echo "================================================================================"
        echo ""
    done
done

echo "================================================================================"
echo "All Experiments Completed！"
echo "================================================================================"
echo "Results are saved: ${RESULTS_FILE}"
cat "${RESULTS_FILE}"
