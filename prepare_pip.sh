#!/bin/bash
#shell script
git clone https://github.com/FaridAF276/SSL_MoCo_New.git
cd SSL_MoCo_New
pip install pandas matplotlib tensorboard gdown gdrive
gdown --fuzzy https://drive.google.com/file/d/1Y9TGam8gf_SnmdMIicaDVXJF9TIDo7ke/view?usp=sharing
mkdir MoCo_train_checkpoints
mkdir MoCo_eval_checkpoints

time python pre_train.py \
--epochs 2 \
--batch-size 512 \
--results-dir "MoCo_train_checkpoints/"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 2 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "cifar10" \
-pt-ssl
