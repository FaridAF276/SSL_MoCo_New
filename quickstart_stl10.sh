#!/bin/bash

# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_stl10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
gdown --fuzzy https://drive.google.com/file/d/1ny6vBH54X0qV07EsNhgddHFGYgoROswy/view?usp=sharing && unzip stl10.zip
time python dataset_preparation.py \
--dataset_dir stl10 \
--percentage 0.2 \
--split_train_test

mkdir -p MoCo_train_checkpoints && \
mkdir -p MoCo_eval_checkpoints
#To start from here type this command : 
# tail -n +17 quickstart_stl10xray.sh | bash
python -c "import torch; import torchvision; print('\n Torch version:\t', torch.__version__, '\n Torchvision version:\t', torchvision.__version__)"
# #Launch training process
time python pre_train.py \
--epochs 20 \
--batch_size 256 \
--lr 0.01 \
--results-dir "MoCo_train_checkpoints/" \
--dataset "folder" \
--root_folder "pretext" \
--cos \
--knn-k 4000 \
--size_crop 32 \
--moco-k 4096 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
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
zip -r stl10_pretext_4096.zip MoCo_train_checkpoints
zip -r stl10_dowstr_4096.zip MoCo_eval_checkpoints
cd
./gdrive upload stl10_pretext_4096.zip
./gdrive upload stl10_dowstr_4096.zip