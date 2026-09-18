#!/bin/bash
#=================================================================
# Cheap baselines WITH Euclidean Alignment (Fix 1).
# The original run_baselines.sh (8098352) ran WITHOUT --ea; this provides
# the aligned cell of the 2x2 (aligned vs not x MetaSub vs baseline)
# for average_adapters and the other donor/shared baselines.
#
# ea flag is now logged in wandb config and run name gets _ea suffix.
#
# Array: 4 conditions x 2 datasets x 9 subjects x 5 seeds = 360 tasks
#   Factor order (innermost -> outermost): seed, condition, subject, dataset
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=06:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=baselines_ea
#SBATCH --output=logs/baselines_ea_%A_%a.out
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

N_CONDITIONS=${#CONDITIONS[@]}
N_SEEDS=${#SEEDS[@]}
TASKS_PER_DATASET=$(( 9 * N_CONDITIONS * N_SEEDS ))

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

echo "=== Task ${SLURM_ARRAY_TASK_ID}: ${CONDITION} ${DATASET} heldout${HELD_OUT} seed${SEED} (EA) ==="

python run_experiment.py \
    --condition "${CONDITION}" \
    --dataset   "${DATASET}" \
    --held_out  "${HELD_OUT}" \
    --seed      "${SEED}" \
    --ea

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
