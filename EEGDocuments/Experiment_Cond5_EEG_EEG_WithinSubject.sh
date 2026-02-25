#!/bin/bash
#=================================================================
# CONDITION 5: EEG-EEG Alignment — Within-Subject (Upper Bound)
# Query:  Subject A query EEG → Document: Subject A doc EEG (same sentence)
# Loss:   Standard CE (within-subject: only one positive per query)
# Flags:  --subject_mode within-subject
#
# Purpose: Establishes the ceiling for EEG-EEG retrieval performance.
#          Within-subject pairing removes all inter-subject variance
#          from the retrieval signal — the encoder only needs to
#          discriminate between sentences for the same person.
#          Any cross-subject condition (Cond1-4) that approaches
#          this score has successfully learned subject-invariant
#          semantic representations. Conds 1-4 should be interpreted
#          relative to both this ceiling and the EEG-Text baseline
#          (Cond6).
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond5_eeg_eeg_within
#SBATCH --output=slurm-%j_cond5_eeg_eeg_within.out

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
    --subject_mode within-subject

/opt/software/scripts/job_epilogue.sh
