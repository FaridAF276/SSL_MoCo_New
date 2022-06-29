#!/bin/bash
#shell script
apt-get install zip unzip
#git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x prepare_pip.sh && ./prepare_pip.sh
pip install pandas matplotlib tensorboard gdown gdrive
#Download and install gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about

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

zip -r cifar10_pretext.zip MoCo_train_checkpoints
zip -r cifar10_dowstr.zip MoCo_eval_checkpoints
./gdrive cifar10_pretext.zip
./gdrive cifar10_pretext.zip

time python pre_train.py \
--epochs 2 \
--batch-size 512 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "stl10"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 2 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "stl10" \
--results_dir "MoCo_eval_checkpoints/" \
-pt-ssl

zip -r stl10_pretext.zip MoCo_train_checkpoints
zip -r stl10_dowstr.zip MoCo_eval_checkpoints
./gdrive stl10_pretext.zip
./gdrive stl10_pretext.zip
