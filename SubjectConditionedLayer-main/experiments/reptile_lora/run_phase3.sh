#!/bin/bash
#=================================================================
# Phase 3 arms: MAML, hypernetwork, compute-matched control.
#
# Array layout:
#   3 conditions x 2 datasets x 9 subjects x 5 seeds = 270 tasks
#
# Conditions: maml, hypernet, compute_matched
#   Factor order (innermost -> outermost): seed, condition, subject, dataset
#
#   BCI2a only:   sbatch --array=0-134   run_phase3.sh
#   BCI2b only:   sbatch --array=135-269 run_phase3.sh
#   Debug single: sbatch --array=0-0     run_phase3.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=08:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=phase3_arms
#SBATCH --output=logs/phase3_%A_%a.out
#SBATCH --array=0-269

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
CONDITIONS=("maml" "hypernet" "compute_matched")
SEEDS=(1 2 3 4 5)

N_CONDITIONS=${#CONDITIONS[@]}   # 3
N_SEEDS=${#SEEDS[@]}             # 5
TASKS_PER_DATASET=$(( 9 * N_CONDITIONS * N_SEEDS ))  # 135

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
