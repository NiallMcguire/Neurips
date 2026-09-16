#!/bin/bash
#=================================================================
# Cheap non-meta baselines (reviewer request 1 — decisive item)
# under identical LOSO plumbing as run_reptile.sh.
#
# Array layout:
#   4 conditions x 2 datasets x 9 subjects x 5 seeds = 360 tasks
#
# Conditions:
#   average_adapters — mean of trained adapters in ΔW space (SVD refactored)
#   shared_adapter   — held-out slot init from slot 0
#   donor_random     — random training subject's adapter
#   donor_nearest    — nearest training subject (log-Euclidean on mean cov)
#
#   Factor order (innermost -> outermost):
#     seed      (5)
#     condition (4)
#     subject   (9)
#     dataset   (2)
#
#   BCI2a only:   sbatch --array=0-179   run_baselines.sh
#   BCI2b only:   sbatch --array=180-359 run_baselines.sh
#   Debug single: sbatch --array=0-0     run_baselines.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=06:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=baseline_inits
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --array=0-359

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
CONDITIONS=("average_adapters" "shared_adapter" "donor_random" "donor_nearest")
SEEDS=(1 2 3 4 5)

N_CONDITIONS=${#CONDITIONS[@]}   # 4
N_SEEDS=${#SEEDS[@]}             # 5
TASKS_PER_DATASET=$(( 9 * N_CONDITIONS * N_SEEDS ))  # 180

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASET_IDX=$(( TASK_ID / TASKS_PER_DATASET ))
REMAINDER=$(( TASK_ID % TASKS_PER_DATASET ))
SUBJECT_IDX=$(( REMAINDER / (N_CONDITIONS * N_SEEDS) ))
REMAINDER2=$(( REMAINDER % (N_CONDITIONS * N_SEEDS) ))
CONDITION_IDX=$(( REMAINDER2 / N_SEEDS ))
SEED_IDX=$(( REMAINDER2 % N_SEEDS ))

DATASET=${DATASETS[$DATASET_IDX]}
HELD_OUT=${SUBJECTS[$SUBJECT_IDX]}
CONDITION=${CONDITIONS[$CONDITION_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo " Dataset:    ${DATASET}"
echo " Condition:  ${CONDITION}"
echo " Held-out:   subject ${HELD_OUT}"
echo " Seed:       ${SEED}"
echo "========================================================"

python run_experiment.py \
    --condition "${CONDITION}" \
    --dataset   "${DATASET}" \
    --held_out  "${HELD_OUT}" \
    --seed      "${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
