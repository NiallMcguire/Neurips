#!/bin/bash
#=================================================================
# CONDITION 3: EEG-EEG Alignment — Multi-Positive + Weighted Positives
# Query:  Subject A EEG → Document: Subject B EEG (same sentence)
# Loss:   Weighted multi-positive CE (raw EEG cosine similarity weights)
# Flags:  --multi_positive_train --multi_positive_eval
#         --weighted_multi_positive
#
# Purpose: Tests whether weighting co-positives by their cosine
#          similarity to the query in raw (pre-encoder) EEG space
#          improves gradient quality during cross-subject training.
#          Neurologically similar cross-subject pairs contribute more
#          gradient; noisy pairs are down-weighted automatically,
#          approximating a natural curriculum without manual scheduling.
#          Doc-subject pairing remains static (fixed at preprocessing),
#          isolating the contribution of positive weighting alone.
#          Compare directly against Cond1 to measure its effect.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond3_eeg_eeg_mp_wtd
#SBATCH --output=slurm-%j_cond3_eeg_eeg_mp_wtd.out

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
    --lr 1e-4 \
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
    --weighted_multi_positive

/opt/software/scripts/job_epilogue.sh
