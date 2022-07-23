#!/bin/bash
# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_cifar10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}



cd SSL_MoCo_New
gdown --fuzzy https://drive.google.com/file/d/1ny6vBH54X0qV07EsNhgddHFGYgoROswy/view?usp=sharing && unzip -n cifar10.zip
time python dataset_preparation.py \
--dataset_dir cifar10 \
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
--epochs 1 \
--batch_size 256 \
--lr 0.01 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "pretext" \
--cos \
--knn-k 4000 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
zip -r cifar10_pretext.zip MoCo_train_checkpoints
cd 
./gdrive upload cd SSL_MoCo_New/cifar10_pretext.zip
cd SSL_MoCo_New
time python linear_eval.py \
--epochs 1 \
--batch_size 256 \
--lr 0.01 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 10 \
-pt-ssl

# #Zip the result and upload them to drive

zip -r cifar10_dowstr.zip MoCo_eval_checkpoints
cd
./gdrive upload cifar10_dowstr.zip