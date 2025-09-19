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
#SBATCH --time=10:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END

# Job name
#SBATCH --job-name=brain_labram_sequence_concat

# Output file
#SBATCH --output=slurm-%j.out

#======================================================

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

# LABRAM SETUP with SEQUENCE CONCATENATION
python controller.py \
  --data_paths /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
               /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/alice_ict_pairs_RUNTIME_MASKING.npy \
  --dataset_types nieuwland nieuwland \
  --encoder_type dual \
  --eeg_arch transformer \
  --pooling_strategy max \
  --query_type eeg \
  --no_lora \
  --colbert_model_name colbert-ir/colbertv2.0 \
  --lr 1e-4 \
  --patience 10 \
  --epochs 50 \
  --batch_size 32 \
  --max_text_len 256 \
  --enable_multi_masking_validation \
  --validation_masking_levels 90 \
  --multi_masking_frequency 3 \
  --primary_masking_level 90 \
  --training_masking_level 90 \
  --use_cnn_preprocessing \
  --use_sequence_concat

/opt/software/scripts/job_epilogue.sh