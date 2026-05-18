#!/bin/bash
#=================================================================
# Grassmann MetaLoRA vs Euclidean MetaLoRA — LOSO Comparison
# BCI Competition IV 2a + 2b
#
# Runs reptile_lora_grassmann alongside reptile_lora for a direct
# head-to-head comparison. baseline_lora results already exist from
# run_reptile.sh and can be compared via wandb.
#
# Array layout:
#   2 conditions x 2 datasets x 9 subjects x 3 seeds = 108 tasks
#
#   Factor order (innermost -> outermost):
#     seed      (3)  — innermost, cycles fastest
#     condition (2)  — reptile_lora then reptile_lora_grassmann
#     subject   (9)
#     dataset   (2)  — outermost
#
#   This interleaving means after every 6 tasks you have both
#   conditions fully evaluated on one subject/dataset across all
#   seeds — meaningful comparisons appear as early as possible
#   on a single GPU running tasks sequentially.
#
#   Example mapping:
#     Task   0: BCI2a, reptile_lora,           subject 1, seed 1
#     Task   1: BCI2a, reptile_lora,           subject 1, seed 2
#     Task   2: BCI2a, reptile_lora,           subject 1, seed 3
#     Task   3: BCI2a, reptile_lora_grassmann, subject 1, seed 1
#     Task   4: BCI2a, reptile_lora_grassmann, subject 1, seed 2
#     Task   5: BCI2a, reptile_lora_grassmann, subject 1, seed 3
#     Task   6: BCI2a, reptile_lora,           subject 2, seed 1
#     ...
#     Task  53: BCI2a, reptile_lora_grassmann, subject 9, seed 3
#     Task  54: BCI2b, reptile_lora,           subject 1, seed 1
#     ...
#     Task 107: BCI2b, reptile_lora_grassmann, subject 9, seed 3
#
# Useful partial submissions:
#   BCI2a only:                  sbatch --array=0-53   run_grassmann.sh
#   BCI2b only:                  sbatch --array=54-107 run_grassmann.sh
#   Grassmann only (BCI2a):      sbatch --array=3-53:6 run_grassmann.sh
#   Euclidean only (BCI2a):      sbatch --array=0-53:6 run_grassmann.sh
#   Single debug task:           sbatch --array=0-0    run_grassmann.sh
#   Grassmann subject 1 (both):  sbatch --array=3,4,5,57,58,59 run_grassmann.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=06:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=grassmann_lora
#SBATCH --output=logs/grassmann_%A_%a.out
#SBATCH --array=0-107

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

# ── Factor arrays ──────────────────────────────────────────────────────────────
DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
CONDITIONS=("reptile_lora" "reptile_lora_grassmann")
SEEDS=(1 2 3)

N_DATASETS=${#DATASETS[@]}       # 2
N_SUBJECTS=${#SUBJECTS[@]}       # 9
N_CONDITIONS=${#CONDITIONS[@]}   # 2
N_SEEDS=${#SEEDS[@]}             # 3

# Tasks per dataset = N_SUBJECTS * N_CONDITIONS * N_SEEDS = 9 * 2 * 3 = 54
TASKS_PER_DATASET=$(( N_SUBJECTS * N_CONDITIONS * N_SEEDS ))

# Map SLURM_ARRAY_TASK_ID -> (dataset, subject, condition, seed)
# Factor order innermost -> outermost: seed, condition, subject, dataset
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
