#!/bin/bash
# cd && wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_cifar10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
##Download pretrained model
gdown --fuzzy -O temp.zip https://drive.google.com/file/d/10mLWQpGox9YJwa6P2qIqp6XpyalxwL7W/view?usp=sharing
unzip -n temp.zip
gdown --fuzzy -O imagenet.zip https://drive.google.com/file/d/1_dRbJEpMH7436l8aU4xrGHcFIE9i5TX7/view?usp=sharing && unzip -n imagenet.zip
mkdir -p MoCo_eval_checkpoints

time python dataset_preparation.py \
--dataset_dir imagenet \
--percentage 0.05 \
--split_train_test
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
##Download model
mkdir -p MoCo_eval_checkpoints
time python linear_eval.py \
--epochs 10 \
--batch_size 256 \
--lr 0.5 \
--model-dir "MoCo_train_checkpoints/" \
--dataset-ft "folder" \
--results_dir "MoCo_eval_checkpoints/" \
--root_folder "downstream" \
--cos \
--num_classes 200 \
-pt-ssl