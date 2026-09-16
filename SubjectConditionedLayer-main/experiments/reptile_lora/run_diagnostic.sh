#!/bin/bash
#=================================================================
# ΔW heterogeneity diagnostic.
#
# Prerequisites: multi-seed LOSO runs already completed (checkpoints/).
# The diagnostic needs per-subject adapters across seeds, so run the
# full sweeps first (run_reptile.sh and run_ea_ablation.sh).
#
# This script is a placeholder driver — the diagnostic logic lives in
# delta_w_diagnostics.py. Extend that script's main() to reconstruct
# models from checkpoints and call spread_metrics on pre/post-EA runs.
#=================================================================

set -e
cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/reptile_lora
source ../../venv_sulora/bin/activate

echo "Checkpoints available:"
ls checkpoints/*.pt 2>/dev/null | head -20 || echo "  (none yet — run the sweeps first)"

python delta_w_diagnostics.py \
    --checkpoints checkpoints/*.pt \
    --out delta_w_diagnostic.json

echo "Diagnostic written to delta_w_diagnostic.json"
