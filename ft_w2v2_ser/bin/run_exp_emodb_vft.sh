#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --export=NONE

# set TF_ENABLE_ONEDNN_OPTS environment variable against floating-point round-off errors
export TF_ENABLE_ONEDNN_OPTS=0

# find the Python executable dynamically
PYTHON_EXEC=$(which python)

FOLDS=$1
NUM_EXPS=$2
EPOCHS=$3
LEARNING_RATE=$4

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --max_epochs $EPOCHS \
    --num_exps $NUM_EXPS \
    --lr $LEARNING_RATE \
    --datadir  ../Dataset/emodb/wav \
    --labeldir ../Dataset/emodb/labels \
    --saving_path downstream/checkpoints/VFT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_lr${LEARNING_RATE} \
    --outputfile downstream/checkpoints/VFT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_lr${LEARNING_RATE}/metrics_folds${FOLDS}_numexps${NUM_EXPS}_lr${LEARNING_RATE}.txt
