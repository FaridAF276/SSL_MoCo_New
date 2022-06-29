#!/bin/bash
#shell script
pip install pandas matplotlib tensorboard gdown gdrive
#Create directories for train et eval models
mkdir MoCo_train_checkpoints
mkdir MoCo_eval_checkpoints
#Launch training process
time python pre_train.py \
--epochs 2 \
--batch-size 512 \
--results-dir "MoCo_train_checkpoints/"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 2 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "cifar10" \
--results_dir "MoCo_eval_checkpoints/" \
-pt-ssl
