#!/bin/bash
#=================================================================
# CONDITION 8: EEG-Text, EEG-style document encoder, Multi-Positive CE
#
# Query:    EEG → EEGEncoder (trainable)
# Document: text tokens → frozen BERT embeddings → EEGEncoder (trainable)
# Loss:     Multi-positive CE — same-sentence batch duplicates treated
#           as co-positives (mirrors EEG-EEG loss formulation exactly)
#
# Purpose: Combines the fairest loss formulation (Cond 4) with the
#          EEG-style document encoder (Cond 5). Both query and document
#          now use the same EEGEncoder architecture AND the same
#          multi-positive loss as EEG-EEG (Cond 1), making this the
#          most direct apples-to-apples comparison against Cond 1:
#
#   Cond 1 vs Cond 8:  EEG doc modality  vs  text doc modality
#                       Same encoder architecture, same loss
#                       → cleanest possible modality effect
#   Cond 4 vs Cond 8:  BERT doc encoder  vs  EEG doc encoder
#                       Same loss (multi-positive), different doc arch
#                       → architecture effect under matched loss
#   Cond 5 vs Cond 8:  standard CE       vs  multi-positive CE
#                       Same EEG doc encoder, different loss
#                       → loss formulation effect with EEG doc encoder
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond8_eeg_text_eegdoc_mp
#SBATCH --output=slurm-%j_cond8_eeg_text_eegdoc_mp.out

module purge
module load nvidia/sdk/23.3
source /users/gxb18167/venv_clean/bin/activate

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type text \
    --doc_encoder_type eeg \
    --text_loss_mode multi_positive \
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
    --subject_mode cross-subject

/opt/software/scripts/job_epilogue.sh
