#!/bin/bash
#=================================================================
# CONDITION 4: EEG-Text Alignment (Multi-Positive CE)
# Query: Subject A EEG → Document: Tokenized text of same sentence
# Loss:  Multi-positive CE + uniformity loss
#
# Fixes applied (v2.3):
#   --temperature 0.2       : same rationale as Cond1. EEG-Text similarities
#                             also live in the high range at initialisation.
#   --uniformity_weight 0.5 : uniformity loss on query (EEG) vectors.
#   per-channel norm        : dataloader per-electrode normalization.
#
# Note: EEG-Text serves as the cross-modal ceiling. These fixes should
#       benefit it too but the effect will be smaller since text docs
#       have no subject fingerprint to exploit.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond4_eeg_text_mp
#SBATCH --output=slurm-%j_cond4_eeg_text_mp.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type text \
    --pooling_strategy max \
    --eeg_arch transformer \
    --hidden_dim 768 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --dropout 0.3 \
    --temperature 0.07 \
    --patience 10 \
    --epochs 50 \
    --batch_size 64 \
    --max_eeg_len 50 \
    --seed 42 \
    --subject_mode cross-subject \
    --text_loss_mode multi_positive

/opt/software/scripts/job_epilogue.sh