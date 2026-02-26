#!/bin/bash
#=================================================================
# TWO-SUBJECT EEG-TEXT ALIGNMENT
#
# Query:    Subject A (glas005allseg) EEG → sentence_i
# Document: Text of sentence_i        (BERT tokenized)
#
# SAME query subject as the EEG-EEG condition above.
# This is the ceiling — how well does Subject A's EEG align to
# the ground-truth text representation?
#
# The gap between this and EEG-EEG quantifies the cost of
# replacing the text anchor with another subject's EEG.
#
# Note: --doc_subject has no effect for EEG-Text (doc is always
# text). Only --query_subject matters here for comparability.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=two_subj_eeg_text
#SBATCH --output=slurm-%j_two_subj_eeg_text.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type text \
    --subject_mode cross-subject \
    --query_subject glas005allseg \
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
    --text_loss_mode standard

/opt/software/scripts/job_epilogue.sh
