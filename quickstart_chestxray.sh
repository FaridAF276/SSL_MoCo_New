#!/bin/bash
#Around 5 GB used
apt-get install -y wget; wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
./vast start instance ${VAST_CONTAINERLABEL:2}
#shell script
# tail -n +17 quickstart_chestxray.sh | bash
#cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key;
# apt-get install -y git zip vim unzip fastjar && git clone https://github.com/FaridAF276/SSL_MoCo_New.git && cd SSL_MoCo_New && chmod +x quickstart_chestxray.sh && ./quickstart_chestxray.sh
apt update -y
pip install pandas matplotlib tensorboard Pillow split-folders
#Download and connect with gdrive
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_2.1.1_linux_386.tar.gz
tar -xvf gdrive_2.1.1_linux_386.tar.gz
./gdrive about
#Download ImageNet dataset
wget https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded && \
unzip file_downloaded
# splitfolders --output ChestX --ratio .8 .1 .1 --move \
# -- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset

time python dataset_preparation.py \
--dataset_dir COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset \
--percentage 0.2 \
--split_train_test
#To start from here type this command : 
# tail -n +17 quickstart_chestxray.sh | bash

#Create directories for train et eval models
mkdir MoCo_train_checkpoints && \
mkdir MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
# #Launch training process
time python pre_train.py \
--epochs 200 \
--batch_size 16 \
--lr 0.6 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "pretext" \
--cos \
--knn-k 4000 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log

time python linear_eval.py \
--epochs 200 \
--batch_size 16 \
--lr 0.6 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 3 \
-pt-ssl

# #Zip the result and upload them to drive
zip -r chest_pretext.zip MoCo_train_checkpoints
zip -r chest_dowstr.zip MoCo_eval_checkpoints
./gdrive upload chest_pretext.zip
./gdrive upload chest_dowstr.zip
rm -rf MoCo_train_checkpoints
rm -rf MoCo_eval_checkpoints
./vast stop instance ${VAST_CONTAINERLABEL:2}