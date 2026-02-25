#!/bin/bash
#=================================================================
# CONDITION 1: EEG-EEG Alignment — Multi-Positive (Baseline)
# Query:  Subject A EEG → Document: Subject B EEG (same sentence)
# Loss:   Multi-positive CE (all subjects reading same sentence = positives)
# Flags:  --multi_positive_train --multi_positive_eval
#
# Purpose: Cross-subject EEG-EEG retrieval baseline. Static doc-subject
#          pairing (fixed at preprocessing). No positive weighting.
#          Serves as the direct comparison point for Cond2, 3, and 4.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond1_eeg_eeg_mp
#SBATCH --output=slurm-%j_cond1_eeg_eeg_mp.out

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
    --dropout 0.3 \
    --patience 10 \
    --epochs 50 \
    --batch_size 64 \
    --max_eeg_len 50 \
    --seed 42 \
    --subject_mode cross-subject \
    --multi_positive_eval \
    --multi_positive_train

/opt/software/scripts/job_epilogue.sh
