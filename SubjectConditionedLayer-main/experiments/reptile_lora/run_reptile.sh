#!/bin/bash
#=================================================================
# Reptile LoRA vs Baseline LoRA — LOSO Evaluation
# BCI Competition IV 2a (9 subjects, 4-class motor imagery)
#
# Array layout:
#   2 conditions x 9 subjects x 3 seeds = 54 tasks
#
#   Task ID mapping:
#     seed_idx      = TASK_ID % 3
#     subject_idx   = (TASK_ID / 3) % 9
#     condition_idx = TASK_ID / 27
#
# Submit all 54 jobs:
#   sbatch run_reptile.sh
#
# Submit a single job for debugging (task 0 = baseline, subject 1, seed 1):
#   sbatch --array=0-0 run_reptile.sh
#
# Submit only reptile condition (tasks 27-53):
#   sbatch --array=27-53 run_reptile.sh
#
# Submit only baseline condition (tasks 0-26):
#   sbatch --array=0-26 run_reptile.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=06:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=reptile_lora
#SBATCH --output=logs/reptile_%A_%a.out
#SBATCH --array=0-53

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

# ── Factor arrays ─────────────────────────────────────────────────────────────
CONDITIONS=("baseline_lora" "reptile_lora")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEEDS=(1 2 3)

N_CONDITIONS=${#CONDITIONS[@]}   # 2
N_SUBJECTS=${#SUBJECTS[@]}       # 9
N_SEEDS=${#SEEDS[@]}             # 3

# Map SLURM_ARRAY_TASK_ID -> (condition, subject, seed)
TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( TASK_ID % N_SEEDS ))
SUBJECT_IDX=$(( (TASK_ID / N_SEEDS) % N_SUBJECTS ))
CONDITION_IDX=$(( TASK_ID / (N_SEEDS * N_SUBJECTS) ))

CONDITION=${CONDITIONS[$CONDITION_IDX]}
HELD_OUT=${SUBJECTS[$SUBJECT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo " Condition:  ${CONDITION}"
echo " Held-out:   subject ${HELD_OUT}"
echo " Seed:       ${SEED}"
echo "========================================================"

python run_experiment.py \
    --condition "${CONDITION}" \
    --held_out  "${HELD_OUT}" \
    --seed      "${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
