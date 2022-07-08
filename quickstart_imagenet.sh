#!/bin/bash
#Around 5 GB used
#shell script
# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x quickstart_imagenet.sh && ./quickstart_imagenet.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow gdown
#Download and connect with gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
#Download ImageNet dataset
gdown --fuzzy https://drive.google.com/file/d/1NeBMqfrgLPJcb6_w9-2QZ7ZgYeSzG__u/view?usp=sharing && unzip tiny_imagenet_200.zip
#Create directories for train et eval models
mkdir MoCo_train_checkpoints && \
mkdir MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
#to start from here type this command : tail -n +17 quickstart_chestxray.sh | bash
# #Launch training process
time python pre_train.py \
--epochs 200 \
--batch_size 1 \
--lr 0.6 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "ChestX" \
--cos \
--knn-k 4000 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 200 \
--batch_size 1 \
--lr 0.6 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "ChestX" \
--cos \
--num_classes 3 \
-pt-ssl

# #Zip the result and upload them to drive
zip -r imagenet_pretext.zip MoCo_train_checkpoints
zip -r imagenet_dowstr.zip MoCo_eval_checkpoints
./gdrive upload imagenet_pretext.zip
./gdrive upload imagenet_dowstr.zip
rm -rf MoCo_train_checkpoints
rm -rf MoCo_eval_checkpoints

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
# rm -rf MoCo_train_checkpoints
# rm -rf MoCo_eval_checkpoints
# cd ~
# rm -r SSL_MoCo_New
