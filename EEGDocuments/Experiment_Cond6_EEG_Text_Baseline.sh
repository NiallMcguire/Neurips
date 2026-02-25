#!/bin/bash
#=================================================================
# CONDITION 6: EEG-Text Alignment — Cross-Modal Baseline
# Query:  Subject A EEG → Document: Text (same sentence)
# Loss:   Standard CE
# Flags:  --document_type text
#
# Purpose: The primary performance ceiling for the full experiment.
#          Text documents provide a clean, subject-invariant semantic
#          anchor via LoRA-BERT, so the EEG encoder only needs to
#          solve the EEG-to-semantics mapping — not inter-subject
#          variance.  All cross-subject EEG-EEG conditions (Cond1-4)
#          are evaluated against this score: closing the gap between
#          Cond4 and Cond6 is the central research question.
#          If any EEG-EEG condition exceeds Cond6, it would suggest
#          EEG documents carry retrieval-relevant signal beyond text.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond6_eeg_text_baseline
#SBATCH --output=slurm-%j_cond6_eeg_text_baseline.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type text \
    --colbert_model_name colbert-ir/colbertv2.0 \
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
    --max_text_len 256 \
    --seed 42 \
    --subject_mode cross-subject

/opt/software/scripts/job_epilogue.sh
