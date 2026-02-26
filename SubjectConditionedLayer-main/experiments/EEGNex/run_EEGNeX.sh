#!/bin/bash
#=================================================================
# Job script for SuLoRA EEGNeX Motor Imagery Experiment
# BCI Competition IV-2a Dataset
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=sulora_eegnex
#SBATCH --output=slurm-%j.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

source /users/gxb18167/Neurips/SubjectConditionedLayer-main/venv_sulora/bin/activate

cd /users/gxb18167/Neurips/SubjectConditionedLayer-main/experiments/EEGNex

python EEGNeX.py

/opt/software/scripts/job_epilogue.sh
