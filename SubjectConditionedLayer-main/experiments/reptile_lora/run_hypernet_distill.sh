#!/bin/bash
#=================================================================
# Hypernetwork with gated distillation of per-subject teachers into
# rank-8 adapter targets (Phase 3 improvement path).
#
# Array: 1 condition x 2 datasets x 9 subjects x 5 seeds = 90 tasks
#   Factor order (innermost -> outermost): seed, subject, dataset
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=08:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=hypernet_distill
#SBATCH --output=logs/hndistill_%A_%a.out
#SBATCH --array=0-89

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate
cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora
mkdir -p logs

DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEEDS=(1 2 3 4 5)

N_SEEDS=${#SEEDS[@]}
TASKS_PER_DATASET=$(( 9 * N_SEEDS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASET_IDX=$(( TASK_ID / TASKS_PER_DATASET ))
REMAINDER=$(( TASK_ID % TASKS_PER_DATASET ))
SUBJECT_IDX=$(( REMAINDER / N_SEEDS ))
SEED_IDX=$(( REMAINDER % N_SEEDS ))

DATASET=${DATASETS[$DATASET_IDX]}
HELD_OUT=${SUBJECTS[$SUBJECT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Task ${SLURM_ARRAY_TASK_ID}: hypernet_distill ${DATASET} heldout${HELD_OUT} seed${SEED} ==="

python run_experiment.py \
    --condition hypernet_distill \
    --dataset   "${DATASET}" \
    --held_out  "${HELD_OUT}" \
    --seed      "${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
