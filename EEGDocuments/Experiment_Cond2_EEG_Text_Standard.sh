#!/bin/bash
#=================================================================
# CONDITION 2: EEG-Text Alignment (Standard CE)
# Query: Subject A EEG → Document: Tokenized text of same sentence
# Loss: Standard CE — same-sentence batch duplicates treated as hard negatives
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=cond2_eeg_text_standard
#SBATCH --output=slurm-%j_cond2_eeg_text_standard.out

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
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dropout 0.2 \
    --patience 10 \
    --epochs 50 \
    --batch_size 64 \
    --max_eeg_len 50 \
    --seed 42 \
    --subject_mode cross-subject \
    --text_loss_mode standard

/opt/software/scripts/job_epilogue.sh
