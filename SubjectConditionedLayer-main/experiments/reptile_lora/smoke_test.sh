#!/bin/bash
#=================================================================
# Smoke test: 5 epochs of each condition on each dataset.
# Run interactively BEFORE submitting run_reptile.sh.
# Tests 4 combinations: 2 conditions x 2 datasets.
#=================================================================

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

mkdir -p logs

run_smoke() {
    local CONDITION=$1
    local DATASET=$2
    echo "========================================================"
    echo " Smoke test: ${CONDITION} | ${DATASET} | subject 1 | seed 1"
    echo "========================================================"

    python - <<EOF
import sys
sys.path.insert(0, '../EEGNex')
import run_experiment
run_experiment.DEFAULT_CONFIG['epochs'] = 5
import argparse
args = argparse.Namespace(condition='${CONDITION}', dataset='${DATASET}',
                          held_out=1, seed=1)
run_experiment.main(args)
EOF

    if [ $? -ne 0 ]; then
        echo "SMOKE TEST FAILED: ${CONDITION} on ${DATASET}"
        exit 1
    fi
    echo "PASSED: ${CONDITION} on ${DATASET}"
    echo ""
}

run_smoke baseline_lora BCI2a
run_smoke reptile_lora  BCI2a
run_smoke baseline_lora BCI2b
run_smoke reptile_lora  BCI2b

echo "========================================================"
echo " All smoke tests PASSED."
echo " Safe to submit: sbatch run_reptile.sh"
echo "========================================================"