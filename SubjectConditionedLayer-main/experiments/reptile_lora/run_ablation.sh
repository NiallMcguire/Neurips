#!/bin/bash
#=================================================================
# Reptile Hyperparameter Ablation
#
# Explores three variables:
#   K            {1, 5, 10, 20}            4 values
#   inner_lr     {0.001, 0.005, 0.01, 0.05} 4 values
#   update_freq  {0, 1, 10, 50}             4 values
#   update_freq_matched {0, 1, 10, 50}      4 values
#
# Total ablations: 4 types x 3 seeds = 12 tasks
# Each task runs all 4 values of its ablation variable sequentially.
#
# Factor order:
#   seed_idx      = TASK_ID % 3
#   ablation_idx  = TASK_ID / 3
#
# Mapping:
#   Task  0: K              seed 1
#   Task  1: K              seed 2
#   Task  2: K              seed 3
#   Task  3: inner_lr       seed 1
#   ...
#   Task 11: update_freq_matched  seed 3
#
# Submit all:
#   sbatch run_ablation.sh
#
# Submit single debug task (K ablation, seed 1):
#   sbatch --array=0-0 run_ablation.sh
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=08:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=reptile_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --array=0-11

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

mkdir -p logs

# ── Factor arrays ──────────────────────────────────────────────────────────────
ABLATIONS=("K" "inner_lr" "update_freq" "update_freq_matched")
SEEDS=(1 2 3)

N_ABLATIONS=${#ABLATIONS[@]}   # 4
N_SEEDS=${#SEEDS[@]}           # 3

TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( TASK_ID % N_SEEDS ))
ABLATION_IDX=$(( TASK_ID / N_SEEDS ))

ABLATION=${ABLATIONS[$ABLATION_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================================"
echo " Task ID:   ${SLURM_ARRAY_TASK_ID}"
echo " Ablation:  ${ABLATION}"
echo " Seed:      ${SEED}"
echo " Dataset:   BCI2a"
echo " Held-out:  subject 1 (single fold for ablation efficiency)"
echo "========================================================"

python run_ablation.py \
    --ablation  "${ABLATION}" \
    --dataset   BCI2a \
    --held_out  1 \
    --seed      "${SEED}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"

/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
