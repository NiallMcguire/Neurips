#!/bin/bash
#=================================================================
# TWO-SUBJECT EEG-EEG ALIGNMENT
#
# Query:    Subject A (glas005allseg) EEG → sentence_i
# Document: Subject B (edin30)        EEG → sentence_i
#
# Subject A is fixed as query, Subject B fixed as document.
# Standard CE loss — no multi-positive needed because there is
# exactly one positive doc per query (the fixed doc subject's
# recording of the same sentence).
#
# Purpose: Show that cross-subject EEG-EEG alignment is possible
#          in a maximally controlled setting. Comparison baseline
#          for the EEG-Text ceiling condition below.
#
# To use different subjects, change --query_subject and --doc_subject.
# Available subjects can be found by running with --inspect_only.
#=================================================================

#SBATCH --export=ALL
#SBATCH --partition=gpu --gpus=1 --mem-per-cpu=36000
#SBATCH --account=moshfeghi-pmwc
#SBATCH --time=12:00:00
#SBATCH --mail-user=niall.mcguire@strath.ac.uk
#SBATCH --mail-type=END
#SBATCH --job-name=two_subj_eeg_eeg
#SBATCH --output=slurm-%j_two_subj_eeg_eeg.out

module purge
module load nvidia/sdk/23.3
module load anaconda/python-3.9.7/2021.11

python controller.py \
    --data_path /users/gxb18167/ECIR2026/SpatialTemporalDecompositionMultiDataset/Dataset/nieuwland_ict_pairs_RUNTIME_MASKING.npy \
    --dataset_type nieuwland \
    --document_type eeg \
    --subject_mode cross-subject \
    --query_subject glas005allseg \
    --doc_subject edin30 \
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
    --seed 42

/opt/software/scripts/job_epilogue.sh
