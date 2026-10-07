#!/bin/bash
# Usage: ./run_single_exp.sh <lr> <weight_decay> <output_dir>

LR=$1
WD=$2
OUTPUT_DIR=$3

# Create experiment ID
EXP_ID="lr_${LR}_wd_${WD}"

# Create output directory
EXP_DIR="${OUTPUT_DIR}/${EXP_ID}"
mkdir -p "${EXP_DIR}"

# Update config file
CONFIG_FILE="/data/ysli/Johnbo/Focus_High_ABMIL/config/e2e/r18_brca_subtyping.yaml"
sed -i "s/^lr: .*/lr: ${LR}/" "${CONFIG_FILE}"
sed -i "s/^weight_decay: .*/weight_decay: ${WD}/" "${CONFIG_FILE}"

# Print experiment info
echo "================================================================================"
echo "Start ${EXP_ID}"
echo "================================================================================"
echo "lr: ${LR}"
echo "wd: ${WD}"
echo "config: ${CONFIG_FILE}"
echo "exp_dir: ${EXP_DIR}"
echo "================================================================================"

# Run training
source /data/ysli/anaconda3/bin/activate chuflymo
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=1,4
bash /data/ysli/Johnbo/Focus_High_ABMIL/train_dist.sh 2 \
    --datasets=brca \
    --project=subtyping \
    --dataset_root=/data/ysli/Johnbo/ABMILX/cpath_image/brca/224_80_level1.lmdb \
    --dataset_sub_root=/data/ysli/Johnbo/ABMILX/cpath_image/brca/224_80_level1_512.lmdb \
    --csv_path=/data/ysli/Johnbo/Focus_High_ABMIL/labels/subtyping_tcga_brca_resample.csv \
    --model=r18_fhabmil_flash \
    --output_path="${EXP_DIR}" \
    --config=/data/ysli/Johnbo/Focus_High_ABMIL/config/e2e/r18_brca_subtyping.yaml \
    --title="${EXP_ID}" \
    2>&1 | tee "${EXP_DIR}/training.log"

echo "================================================================================"
echo "${EXP_ID} Completed"
echo "================================================================================"
