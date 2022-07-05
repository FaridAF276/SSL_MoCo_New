#!/bin/bash
#Around 5 GB used
#shell script
# apt-get install -y git zip unzip && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x quickstart_imagenet.sh && ./quickstart_imagenet.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders
#Download and connect with gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
#Download ImageNet dataset
wget https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/jctsfj2sfn-1.zip && \
unzip jctsfj2sfn-1.zip && \
unzip covid19-pneumonia-normal-chest-xraypa-dataset.zip
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset


#Create directories for train et eval models
mkdir MoCo_train_checkpoints && \
mkdir MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
# #Launch training process
time python pre_train.py \
--epochs 1 \
--batch_size 8 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "ChestX" \
--knn-k 4000 
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 1 \
--batch_size 2048 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "ChestX" \
-pt-ssl

# #Zip the result and upload them to drive
zip -r imagenet_pretext.zip MoCo_train_checkpoints
zip -r imagenet_dowstr.zip MoCo_eval_checkpoints
./gdrive upload imagenet_pretext.zip
./gdrive upload imagenet_dowstr.zip
rm -rf MoCo_train_checkpoints
rm -rf MoCo_eval_checkpoints