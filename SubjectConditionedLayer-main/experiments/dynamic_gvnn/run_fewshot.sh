#!/bin/bash
#=================================================================
# Few-Shot Adaptation Speed Experiment.
#
# For each (condition, target subject, seed):
#   - Pre-train on all 9 subjects (full session T data)
#   - Fine-tune subject-specific params on N = 0,5,10,20,50,100 trials
#   - Evaluate on held-out half of session E data
#
# Conditions that have subject-specific params:
#   static_lora, gvnn_subject, gvnn_additive, full
#
# Array layout:
#   4 conditions x 9 subjects x 3 seeds = 108 tasks
#
# Submit:
#   sbatch run_fewshot.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=10:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=gvnn_fewshot
#SBATCH --output=logs/fewshot_%A_%a.out
#SBATCH --array=0-107

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/dynamic_gvnn

mkdir -p logs

# ── define factors ────────────────────────────────────────────────────────────
CONDITIONS=("static_lora" "gvnn_subject" "gvnn_additive" "full")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEEDS=(1 2 3)

N_CONDITIONS=${#CONDITIONS[@]}    # 4
N_SUBJECTS=${#SUBJECTS[@]}        # 9
N_SEEDS=${#SEEDS[@]}              # 3

TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( TASK_ID % N_SEEDS ))
SUBJECT_IDX=$(( (TASK_ID / N_SEEDS) % N_SUBJECTS ))
CONDITION_IDX=$(( TASK_ID / (N_SEEDS * N_SUBJECTS) ))

CONDITION=${CONDITIONS[$CONDITION_IDX]}
TARGET_SUBJECT=${SUBJECTS[$SUBJECT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:         ${SLURM_ARRAY_TASK_ID}"
echo " Condition:       ${CONDITION}"
echo " Target subject:  ${TARGET_SUBJECT}"
echo " Seed:            ${SEED}"
echo "========================================================"

python few_shot_eval.py \
    --condition       "${CONDITION}" \
    --target_subject  "${TARGET_SUBJECT}" \
    --seed            "${SEED}" \
    --pretrain_epochs 100 \
    --batch_size      64 \
    --lr              1e-3 \
    --rank            4 \
    --alpha           1.0 \
    --node_fn         combined

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
