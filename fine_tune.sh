#!/bin/bash

cd SSL_MoCo_New
##Download pretrained model
gdown --fuzzy -O temp.zip https://drive.google.com/file/d/1U0Kdy1N4DyzY6p3B2gODdSJJoMyTuFlz/view?usp=sharing 
unzip temp.zip
##Download model

time python linear_eval.py \
--epochs 10 \
--batch_size 256 \
--lr 0.01 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "stl10" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 10 \
-pt-ssl