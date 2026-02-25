#!/bin/bash
#=================================================================
# CONDITION 4: EEG-EEG Alignment — Multi-Positive + Dynamic Resample
#              + Weighted Positives (Full Proposed System)
# Query:  Subject A EEG → Document: Subject B EEG (same sentence)
# Loss:   Weighted multi-positive CE
# Flags:  --multi_positive_train --multi_positive_eval
#         --dynamic_resample --weighted_multi_positive
#
# Purpose: Full proposed training strategy combining both improvements.
#          Dynamic resampling exposes the encoder to the complete
#          cross-subject positive distribution each epoch, while
#          weighted multi-positive ensures the gradient within each
#          batch is dominated by the most neurologically reliable
#          positive pairs. Together they address the two core failure
#          modes of naive cross-subject multi-positive training:
#          memorisation of fixed subject pairings, and equal-weight
#          gradients from noisy co-positives.
#          Compare against Cond1 (neither), Cond2 (resample only),
#          and Cond3 (weighting only) to attribute improvements.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond4_eeg_eeg_mp_dyn_wtd
#SBATCH --output=slurm-%j_cond4_eeg_eeg_mp_dyn_wtd.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type eeg \
    --pooling_strategy max \
    --eeg_arch transformer \
    --hidden_dim 768 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --dropout 0.2 \
    --patience 10 \
    --epochs 50 \
    --batch_size 64 \
    --max_eeg_len 50 \
    --seed 42 \
    --subject_mode cross-subject \
    --multi_positive_eval \
    --multi_positive_train \
    --dynamic_resample \
    --weighted_multi_positive

/opt/software/scripts/job_epilogue.sh
