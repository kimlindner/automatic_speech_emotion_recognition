#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --export=NONE

# set TF_ENABLE_ONEDNN_OPTS environment variable against floating-point round-off errors
export TF_ENABLE_ONEDNN_OPTS=0

# find the Python executable dynamically
PYTHON_EXEC=$(which python)

FOLDS=$1
NUM_EXPS=$2
EPOCHS=$3
BATCH_SIZE=$4

srun $PYTHON_EXEC ../run_pretrain.py \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath ../Dataset/emodb/emodb_info.json \
   --labeling_method hard \
   --saving_path pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100 \
   --training_step 1000 \
   --save_top_k 1 \
   --wav2vecpath ../model/wav2vec_large.pt \
   --batch_size ${BATCH_SIZE} \

srun $PYTHON_EXEC ../cluster.py \
   --datadir ../Dataset/emodb/wav \
   --outputdir pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/clusters \
   --model_path pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/last.ckpt \
   --labelpath ../Dataset/emodb/emodb_info.json \
   --model_type wav2vec \
   --sample_ratio 1.0 \
   --num_clusters "64,512,4096" \
   --wav2vecpath ../model/wav2vec_large.pt \

srun $PYTHON_EXEC ../run_second.py \
   --saving_path pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/second \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/clusters/all-clus.json \
   --training_step 1000 \
   --warmup_step 100 \
   --save_top_k 1 \
   --lr 1e-4 \
   --batch_size 64 \
   --num_clusters "64,512,4096" \
   --batch_size ${BATCH_SIZE} \
   --use_bucket_sampler \
   --dynamic_batch \

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
   --precision 16 \
   --num_exps $NUM_EXPS \
   --max_epochs $EPOCHS \
   --datadir ../Dataset/emodb/wav \
   --labeldir ../Dataset/emodb/labels \
   --batch_size ${BATCH_SIZE} \
   --pretrained_path pretrain/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/second/last.ckpt \
   --saving_path downstream/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100 \
   --outputfile downstream/checkpoints/PTAPT/wav2vec_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}_${BATCH_SIZE}_training_step1000_warmup100/metrics_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}.txt

