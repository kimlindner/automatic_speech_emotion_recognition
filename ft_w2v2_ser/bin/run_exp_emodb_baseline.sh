#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --export=NONE

# set TF_ENABLE_ONEDNN_OPTS environment variable against floating-point round-off errors
export TF_ENABLE_ONEDNN_OPTS=0

# find the Python executable dynamically
PYTHON_EXEC=$(which python)

FOLDS=$1
NUM_EXPS=$2
EPOCHS=$3
BATCH_SIZE=$4
TRAINING_STEP=$5

srun $PYTHON_EXEC ../run_baseline_continueFT.py \
   --saving_path pretrain/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_batchsize${BATCH_SIZE}_trainingstep${TRAINING_STEP} \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath ../Dataset/emodb/emodb_info.json \
   --training_step $TRAINING_STEP \
   --warmup_step 100 \
   --save_top_k 1 \
   --lr 1e-4 \
   --batch_size $BATCH_SIZE \
   --use_bucket_sampler \

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --max_epochs $EPOCHS \
    --num_exps $NUM_EXPS \
    --datadir  ../Dataset/emodb/wav \
    --labeldir ../Dataset/emodb/labels \
    --batch_size $BATCH_SIZE \
    --pretrained_path pretrain/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_batchsize${BATCH_SIZE}_trainingstep${TRAINING_STEP}/last.ckpt \
    --saving_path downstream/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_batchsize${BATCH_SIZE}_trainingstep${TRAINING_STEP} \
    --outputfile downstream/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_batchsize${BATCH_SIZE}_trainingstep${TRAINING_STEP}/metrics_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_batchsize${BATCH_SIZE}.txt
