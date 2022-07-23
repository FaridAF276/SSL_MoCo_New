#!/bin/bash
set -e

# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; && \
# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; && \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_imagenet.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
#shell script
#Download ImageNet dataset
gdown --fuzzy https://drive.google.com/file/d/1NeBMqfrgLPJcb6_w9-2QZ7ZgYeSzG__u/view?usp=sharing && unzip -n tiny_imagenet_200.zip
#Create directories for train et eval models
mkdir -p MoCo_train_checkpoints && \
mkdir -p MoCo_eval_checkpoints
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
#to start from here type this command : tail -n +17 quickstart_chestxray.sh | bash
# #Launch training process
#launch one epoch to see if everything is working fine
time python pre_train.py \
--epochs 1 \
--batch_size 16 \
--lr 1.0 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "imagenet" \
--cos \
--knn-k 4000 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 1 \
--batch_size 16 \
--lr 1.0 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "imagenet" \
--cos \
--num_classes 200 \
-pt-ssl

# #Zip the result and upload them to drive
zip -r imagenet_pretext.zip MoCo_train_checkpoints
zip -r imagenet_dowstr.zip MoCo_eval_checkpoints
cd
./gdrive upload SSL_MoCo_New/imagenet_pretext.zip
./gdrive upload SSL_MoCo_New/imagenet_dowstr.zip