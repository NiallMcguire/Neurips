#!/bin/bash
# NOT SUBMITTED YET — staged per instructions. Submit only after the fold-1
# sanity check (python hypernet_objective.py --sanity) has been reviewed.
#
# 9 folds x 3 seeds x 2 EA settings = 54 tasks.

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=08:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=hn_objective
#SBATCH --output=logs/hnobj_%A_%a.out
#SBATCH --array=0-53

module purge
module load nvidia/sdk/23.3
source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate
cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora
mkdir -p logs

SUBJECTS=(1 2 3 4 5 6 7 8 9)
SEEDS=(1 2 3)
EAS=(0 1)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_SEEDS=${#SEEDS[@]}
N_EAS=${#EAS[@]}
SUBJECT_IDX=$(( TASK_ID / (N_SEEDS * N_EAS) ))
REM=$(( TASK_ID % (N_SEEDS * N_EAS) ))
SEED_IDX=$(( REM / N_EAS ))
EA_IDX=$(( REM % N_EAS ))

HELD_OUT=${SUBJECTS[$SUBJECT_IDX]}
SEED=${SEEDS[$SEED_IDX]}
EA=${EAS[$EA_IDX]}

ARGS="--dataset BCI2a --held_out ${HELD_OUT} --seed ${SEED} --out hypernet_objective_results.csv"
[ "${EA}" = "1" ] && ARGS="${ARGS} --ea"

echo "=== Task ${TASK_ID}: held_out=${HELD_OUT} seed=${SEED} ea=${EA} ==="
python hypernet_objective.py ${ARGS}
EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}
