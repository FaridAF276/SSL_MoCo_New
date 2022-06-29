#!/bin/bash
#shell script
#git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x prepare_pip.sh && ./prepare_pip.sh
pip install pandas matplotlib tensorboard gdown gdrive
#Download and install gdrive
wget https://docs.google.com/uc?id=0B3X9GlR6EmbnWksyTEtCM0VfaFE&export=download
mv uc\?id\=0B3X9GlR6EmbnWksyTEtCM0VfaFE gdrive
chmod +x gdrive
cp gdrive /usr/local/bin/gdrive
gdrive list
#Create directories for train et eval models
mkdir MoCo_train_checkpoints
mkdir MoCo_eval_checkpoints
#Launch training process
time python pre_train.py \
--epochs 2 \
--batch-size 512 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "cifar10"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 2 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "cifar10" \
--results_dir "MoCo_eval_checkpoints/" \
-pt-ssl

zip cifar10_pretext.zip MoCo_train_checkpoints
zip cifar10_dowstr.zip MoCo_eval_checkpoints
gdrive upload cifar10_pretext.zip
gdrive upload cifar10_dowstr.zip

time python pre_train.py \
--epochs EPOCH \
--batch-size 512 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "stl10"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs EPOCH \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "stl10" \
--results_dir "MoCo_eval_checkpoints/" \
-pt-ssl

zip stl10_pretext.zip MoCo_train_checkpoints
zip stl10_dowstr.zip MoCo_eval_checkpoints
gdrive upload stl10_pretext.zip
gdrive upload stl10_dowstr.zip