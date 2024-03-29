#!/bin/bash

# wget -nc https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast; \
# cat ~/.ssh/authorized_keys | md5sum | awk '{print $1}' > ssh_key_hv; echo -n $VAST_CONTAINERLABEL | md5sum | awk '{print $1}' > instance_id_hv; head -c -1 -q ssh_key_hv instance_id_hv > ~/.vast_api_key; \
# ./vast start instance ${VAST_CONTAINERLABEL:2} && \
# bash -e SSL_MoCo_New/quickstart_stl10.sh && \
# ./vast stop instance ${VAST_CONTAINERLABEL:2}
cd SSL_MoCo_New
gdown --fuzzy https://drive.google.com/file/d/1B0GLPjsXgtWLhV5SvBT1Kti1QHKWql2t/view?usp=sharing && unzip -n stl10.zip -d stl10
rm -rf stl10/unlabelled/
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
--epochs 50 \
--batch_size 512 \
--lr 0.06 \
--cos \
--results-dir "MoCo_train_checkpoints/" \
--dataset "stl10" \
--aug_plus \
--root_folder "pretext" \
--patience 7 \
--knn-k 4000 \
--resume 'MoCo_train_checkpoints/model.pth' \
--size_crop 32 \
--moco-k 4096 \
--bn-splits 1
touch MoCo_train_checkpoints/linear_eval.log
time python linear_eval.py \
--epochs 50 \
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
./gdrive upload SSL_MoCo_New/stl10_pretext_4096.zip
./gdrive upload SSL_MoCo_New/stl10_dowstr_4096.zip