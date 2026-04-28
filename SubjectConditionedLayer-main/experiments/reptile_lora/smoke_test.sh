#!/bin/bash
#=================================================================
# Smoke test: runs 5 epochs of each condition on subject 1, seed 1.
# Run this interactively BEFORE submitting run_reptile.sh.
# Should complete in a few minutes on CPU.
# If this passes, submit the full array.
#=================================================================

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

mkdir -p logs

echo "========================================================"
echo " Smoke test: baseline_lora, subject 1, seed 1, 5 epochs"
echo "========================================================"

# Temporarily patch epochs to 5 for smoke test
python - <<'EOF'
import sys
sys.path.insert(0, '../EEGNex')

# Patch config before importing run_experiment
import run_experiment
run_experiment.DEFAULT_CONFIG['epochs'] = 5

import argparse
args = argparse.Namespace(condition='baseline_lora', held_out=1, seed=1)
run_experiment.main(args)
EOF

if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED: baseline_lora"
    exit 1
fi

echo ""
echo "========================================================"
echo " Smoke test: reptile_lora, subject 1, seed 1, 5 epochs"
echo "========================================================"

python - <<'EOF'
import sys
sys.path.insert(0, '../EEGNex')

import run_experiment
run_experiment.DEFAULT_CONFIG['epochs'] = 5

import argparse
args = argparse.Namespace(condition='reptile_lora', held_out=1, seed=1)
run_experiment.main(args)
EOF

if [ $? -ne 0 ]; then
    echo "SMOKE TEST FAILED: reptile_lora"
    exit 1
fi

echo ""
echo "========================================================"
echo " Smoke tests PASSED. Safe to submit run_reptile.sh"
echo " sbatch run_reptile.sh"
echo "========================================================"
