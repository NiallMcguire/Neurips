#!/bin/bash
#=================================================================
# CONDITION 2: EEG-EEG Alignment — Multi-Positive + Dynamic Resample
# Query:  Subject A EEG → Document: Subject B EEG (same sentence)
# Loss:   Multi-positive CE
# Flags:  --multi_positive_train --multi_positive_eval --dynamic_resample
#
# Purpose: Tests whether preventing memorisation of a fixed query→doc
#          subject pairing improves cross-subject generalisation.
#          On every training step the document subject is re-sampled
#          uniformly from all other subjects who read the same sentence,
#          forcing the encoder to learn signal that is invariant across
#          the full cross-subject positive distribution rather than
#          overfitting to a single subject pairing per sentence.
#          Val and test splits use static sampling for reproducibility.
#          Isolates the contribution of dynamic resampling from weighting.
#          Compare directly against Cond1 to measure its effect.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond2_eeg_eeg_mp_dynresample
#SBATCH --output=slurm-%j_cond2_eeg_eeg_mp_dynresample.out

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
    --dynamic_resample

/opt/software/scripts/job_epilogue.sh
