#!/bin/bash
#=================================================================
# Reptile Hyperparameter Ablation — Full Sweep
#
# Full factorial:
#   4 ablation types x 4 values x 2 datasets x 9 subjects x 3 seeds
#   = 864 individual runs
#
# Organised as 216 SLURM array tasks.
# Each task runs one (ablation, dataset, subject, seed) combination
# sequentially across all 4 values of that ablation variable.
#
# Factor order (innermost -> outermost):
#   seed      (3)  — cycles fastest
#   subject   (9)
#   dataset   (2)
#   ablation  (4)  — cycles slowest
#
# Task mapping examples:
#   Task   0: K,              BCI2a, subject 1, seed 1
#   Task   1: K,              BCI2a, subject 1, seed 2
#   Task   2: K,              BCI2a, subject 1, seed 3
#   Task   3: K,              BCI2a, subject 2, seed 1
#   ...
#   Task  53: K,              BCI2b, subject 9, seed 3
#   Task  54: inner_lr,       BCI2a, subject 1, seed 1
#   ...
#   Task 215: update_freq_matched, BCI2b, subject 9, seed 3
#
# Useful partial submissions:
#   K only:                sbatch --array=0-53    run_ablation.sh
#   inner_lr only:         sbatch --array=54-107  run_ablation.sh
#   update_freq only:      sbatch --array=108-161 run_ablation.sh
#   update_freq_matched:   sbatch --array=162-215 run_ablation.sh
#   Single debug task:     sbatch --array=0-0     run_ablation.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=reptile_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --array=0-215

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

# ── Factor arrays ──────────────────────────────────────────────────────────────
ABLATIONS=("K" "inner_lr" "update_freq" "update_freq_matched")
DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEEDS=(1 2 3)

N_ABLATIONS=${#ABLATIONS[@]}   # 4
N_DATASETS=${#DATASETS[@]}     # 2
N_SUBJECTS=${#SUBJECTS[@]}     # 9
N_SEEDS=${#SEEDS[@]}           # 3

# Tasks per ablation = N_DATASETS * N_SUBJECTS * N_SEEDS = 54
TASKS_PER_ABLATION=$(( N_DATASETS * N_SUBJECTS * N_SEEDS ))

TASK_ID=${SLURM_ARRAY_TASK_ID}

ABLATION_IDX=$(( TASK_ID / TASKS_PER_ABLATION ))
REMAINDER=$(( TASK_ID % TASKS_PER_ABLATION ))

DATASET_IDX=$(( REMAINDER / (N_SUBJECTS * N_SEEDS) ))
REMAINDER2=$(( REMAINDER % (N_SUBJECTS * N_SEEDS) ))

SUBJECT_IDX=$(( REMAINDER2 / N_SEEDS ))
SEED_IDX=$(( REMAINDER2 % N_SEEDS ))

ABLATION=${ABLATIONS[$ABLATION_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
HELD_OUT=${SUBJECTS[$SUBJECT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo " Ablation:  ${ABLATION}  (runs all 4 values sequentially)"
echo " Dataset:   ${DATASET}"
echo " Held-out:  subject ${HELD_OUT}"
echo " Seed:      ${SEED}"
echo "========================================================"

python run_ablation.py \
    --ablation  "${ABLATION}" \
    --dataset   "${DATASET}" \
    --held_out  "${HELD_OUT}" \
    --seed      "${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}