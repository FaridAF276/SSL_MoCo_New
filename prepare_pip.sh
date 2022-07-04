#!/bin/bash
#shell script
#apt-get install -y git zip unzip && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x prepare_pip.sh && ./prepare_pip.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow gdown
#Download and connect with gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
gdown --fuzzy https://drive.google.com/file/d/1NeBMqfrgLPJcb6_w9-2QZ7ZgYeSzG__u/view?usp=sharing && unzip tiny_imagenet_200.zip
cd SSL_MoCo_New
#Create directories for train et eval models
mkdir MoCo_train_checkpoints && \
mkdir MoCo_eval_checkpoints

# #Launch training process
time python pre_train.py \
--epochs 200 \
--batch_size 512 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "tiny_imagenet_200/train"
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 200 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "cifar10" \
--results_dir "MoCo_eval_checkpoints/" \
-pt-ssl

# #Zip the result and upload them to drive
zip -r cifar10_pretext.zip MoCo_train_checkpoints
zip -r cifar10_dowstr.zip MoCo_eval_checkpoints
./gdrive upload cifar10_pretext.zip
./gdrive upload cifar10_dowstr.zip
rm -r MoCo_train_checkpoints
rm -r MoCo_eval_checkpoints

#Lets do that again!

# time python pre_train.py \
# --epochs 200 \
# --batch-size 512 \
# --results-dir "MoCo_train_checkpoints/" \
# --dataset "stl10"
# touch MoCo_eval_checkpoints/linear_eval.log
# time python linear_eval.py \
# --epochs 200 \
# --model-dir "MoCo_train_checkpoints/" \
# --dataset-ft "stl10" \
# --results_dir "MoCo_eval_checkpoints/" \
# -pt-ssl
# zip -r stl10_pretext.zip MoCo_train_checkpoints
# zip -r stl10_dowstr.zip MoCo_eval_checkpoints
# ./gdrive upload stl10_pretext.zip
# ./gdrive upload stl10_dowstr.zip
# rm -r MoCo_train_checkpoints
# rm -r MoCo_eval_checkpoints
# cd ~
# rm -r SSL_MoCo_New