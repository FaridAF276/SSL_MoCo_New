#!/bin/bash


# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_chestxray.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}

cd SSL_MoCo_New
wget -nc https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/898720e7-9fcd-49f0-87ba-08c979e6f35e -O temp.zip && unzip -n temp.zip
#Around 5 GB used
splitfolders --output ChestX --ratio .8 .1 .1 --move \
-- COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset

time python dataset_preparation.py \
--dataset_dir ChestX \
--percentage 0.2 \
--split_train_test
#To start from here type this command : 
# tail -n +17 quickstart_chestxray.sh | bash

#Create directories for train et eval models
mkdir -p MoCo_train_checkpoints && \
mkdir -p MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
# #Launch training process
time python pre_train.py \
--epochs 400 \
--batch_size 128 \
--lr 0.06 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "pretext" \
--moco-dim 256 \
--cos \
--knn-k 4000 \
--patience 7 \
--chest_aug \
--size_crop 224 \
--moco-k 16384 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 1 \
--batch_size 64 \
--lr 0.012 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 3 \
-pt-ssl

# #Zip the result and upload them to drive
zip -r chest_pretext_8192.zip MoCo_train_checkpoints
zip -r chest_dowstr_8192.zip MoCo_eval_checkpoints
cd
./gdrive upload SSL_MoCo_New/chest_dowstr_8192.zip
./gdrive upload SSL_MoCo_New/chest_pretext_8192.zip
