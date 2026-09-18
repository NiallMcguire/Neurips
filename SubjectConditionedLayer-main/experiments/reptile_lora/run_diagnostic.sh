#!/bin/bash
#=================================================================
# ΔW heterogeneity diagnostic driver (Fix 2): computes between/within
# subject distance ratio on effective deltas, raw and (optionally)
# distilled, before and after EA. Reads checkpoints produced by the
# baseline/reptile sweeps.
#
# Array of 8: dataset(2) x distill(2) x ea(2). condition=baseline_lora
# (per-subject adapters exist for all training subjects).
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=delta_w_diag
#SBATCH --output=logs/diag_%A_%a.out
#SBATCH --array=0-7

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate
cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora
mkdir -p logs diagnostics

DATASETS=("BCI2a" "BCI2b")
T=${SLURM_ARRAY_TASK_ID}
DS=${DATASETS[$(( T / 4 ))]}
REM=$(( T % 4 ))
DISTILL=$(( REM / 2 ))
EA=$(( REM % 2 ))

ARGS="--dataset ${DS} --condition baseline_lora --seeds 1 2 3 4 5"
[ ${DISTILL} -eq 1 ] && ARGS="${ARGS} --distill"
[ ${EA} -eq 1 ] && ARGS="${ARGS} --ea"
OUT="diagnostics/diag_${DS}_distill${DISTILL}_ea${EA}.json"
ARGS="${ARGS} --out ${OUT}"

echo "=== Task ${T}: ${DS} distill=${DISTILL} ea=${EA} -> ${OUT} ==="
python delta_w_diagnostics.py ${ARGS}

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
/opt/software/scripts/job_epilogue.sh
exit ${EXIT_CODE}

