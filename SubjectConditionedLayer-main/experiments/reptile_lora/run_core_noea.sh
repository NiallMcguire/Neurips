#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=06:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=core_noea
#SBATCH --output=logs/core_noea_%A_%a.out
#SBATCH --array=0-179

module purge
module load nvidia/sdk/23.3
source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate
cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora
mkdir -p logs checkpoints

DATASETS=("BCI2a" "BCI2b")
SUBJECTS=(1 2 3 4 5 6 7 8 9)
CONDITIONS=("baseline_lora" "reptile_lora")
SEEDS=(1 2 3 4 5)
N_CONDITIONS=${#CONDITIONS[@]}
N_SEEDS=${#SEEDS[@]}
TASKS_PER_DATASET=$((9 * N_CONDITIONS * N_SEEDS))
TASK_ID=${SLURM_ARRAY_TASK_ID}
DATASET_IDX=$((TASK_ID / TASKS_PER_DATASET))
REMAINDER=$((TASK_ID % TASKS_PER_DATASET))
SUBJECT_IDX=$((REMAINDER / (N_CONDITIONS * N_SEEDS)))
REMAINDER2=$((REMAINDER % (N_CONDITIONS * N_SEEDS)))
CONDITION_IDX=$((REMAINDER2 / N_SEEDS))
SEED_IDX=$((REMAINDER2 % N_SEEDS))

python run_experiment.py \
  --condition "${CONDITIONS[$CONDITION_IDX]}" \
  --dataset "${DATASETS[$DATASET_IDX]}" \
  --held_out "${SUBJECTS[$SUBJECT_IDX]}" \
  --seed "${SEEDS[$SEED_IDX]}"
EXIT_CODE=$?
/opt/software/scripts/job_epilogue.sh
exit $EXIT_CODE
