#!/bin/bash
#=================================================================
#
# Job script for running a job on a single GPU (any available GPU)
#
#=================================================================
#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the gpu partition (queue) with any GPU
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#
# Specify project account (replace as required)
#SBATCH --account=moshfeghi-pmwc
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=2:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END

# Job name
#SBATCH --job-name=brain_text_dynamic_mask

# Output file
#SBATCH --output=slurm-%j.out
#======================================================
#--enable_multi_masking_validation \
#    --validation_masking_levels 0 25 50 75 90 100 \
#    --multi_masking_frequency 1 \
#    --primary_masking_level 90
#--data_paths /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy  \
 #                 /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/alice_ict_pairs_RUNTIME_MASKING.npy  \

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

# TEXT BASELINE: Same architecture but using text queries instead of EEG push
python controller.py \
    --data_paths /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/alice_ict_pairs_RUNTIME_MASKING.npy  \
                /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_types original \
    --encoder_type dual \
    --pooling_strategy max \
    --eeg_arch transformer \
    --query_type text \
    --use_pretrained_text \
    --colbert_model_name colbert-ir/colbertv2.0 \
    --lr 1e-4 \
    --patience 5 \
    --epochs 1 \
    --batch_size 32 \
    --max_text_len 256 \
    --training_masking_level 90 \

/opt/software/scripts/job_epilogue.sh