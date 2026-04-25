#!/bin/bash
#=================================================================
# Ablation Study: Dynamic GVNN vs Subject-Conditioned Layer
# Runs all 6 conditions x 2 datasets x 3 seeds as a SLURM array.
#
# Array layout:
#   Each task = one (condition, dataset, seed) combination.
#   Total tasks = 6 conditions x 2 datasets x 3 seeds = 36
#
# Submit:
#   sbatch run_ablation.sh
#
# To run a single condition for debugging:
#   sbatch --array=0-0 run_ablation.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=08:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=gvnn_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --array=0-35

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/dynamic_gvnn

mkdir -p logs

# ── define all conditions, datasets, seeds ───────────────────────────────────
CONDITIONS=("vanilla" "static_lora" "gvnn_pop" "gvnn_subject" "gvnn_additive" "full")
DATASETS=("BCI2a" "BCI2b")
SEEDS=(1 2 3)

N_CONDITIONS=${#CONDITIONS[@]}    # 6
N_DATASETS=${#DATASETS[@]}        # 2
N_SEEDS=${#SEEDS[@]}              # 3

# Map SLURM_ARRAY_TASK_ID -> (condition, dataset, seed)
TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( TASK_ID % N_SEEDS ))
DATASET_IDX=$(( (TASK_ID / N_SEEDS) % N_DATASETS ))
CONDITION_IDX=$(( TASK_ID / (N_SEEDS * N_DATASETS) ))

CONDITION=${CONDITIONS[$CONDITION_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo " Condition: ${CONDITION}"
echo " Dataset:   ${DATASET}"
echo " Seed:      ${SEED}"
echo "========================================================"

python ablation_study.py \
    --condition   "${CONDITION}" \
    --dataset     "${DATASET}" \
    --seed        "${SEED}" \
    --epochs      100 \
    --batch_size  64 \
    --lr          1e-3 \
    --rank        4 \
    --alpha       1.0 \
    --node_fn     combined

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
